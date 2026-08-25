"""
Entraînement, normalisation intra-course, calibration.

Trois idées, dans cet ordre d'importance.

1. LE DÉCOUPAGE EST CHRONOLOGIQUE, JAMAIS ALÉATOIRE.
   Un `train_test_split(shuffle=True)` sur des courses met le futur dans
   l'entraînement. La validation croisée en k-plis classique fait pareil.
   Ici : entraînement < calibration < test, trois fenêtres disjointes qui
   se suivent dans le temps.

2. UNE PROBABILITÉ SEULE NE VEUT RIEN DIRE, IL FAUT NORMALISER PAR COURSE.
   Le classifieur note chaque partant indépendamment ; rien ne garantit que
   les probabilités d'une course somment à 1. Or il y a exactement un
   gagnant par course. On divise donc par la somme de la course.

3. LA CALIBRATION EST LE LIVRABLE, PAS L'ACCESSOIRE.
   « Degré de confiance » n'a de sens que si, quand le modèle annonce 20 %,
   l'événement arrive vraiment une fois sur cinq. C'est une propriété
   mesurable (courbe de calibration, score de Brier), et elle ne s'obtient
   pas gratuitement : d'où la régression isotonique, ajustée sur une
   fenêtre que le modèle n'a jamais vue.

Deux variantes entraînées systématiquement :
  - `sans_marche` : la seule qui puisse révéler un écart exploitable.
  - `avec_marche` : la cote en entrée. Beaucoup plus performante, et sans
    intérêt pratique — elle recopie le consensus. Elle sert de plafond :
    elle dit ce qui est prévisible au mieux, donc ce qui reste à gagner.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression

from . import features as ft

log = logging.getLogger("pmu.train")

# LightGBM fait mieux sur ce type de données (catégorielles natives,
# apprentissage de rang). On reste sur scikit-learn par défaut pour que le
# dépôt tourne sans dépendance lourde ; bascule automatique s'il est présent.
try:  # pragma: no cover
    import lightgbm as lgb
    LIGHTGBM = True
except ImportError:  # pragma: no cover
    LIGHTGBM = False


@dataclass
class Decoupage:
    """Trois fenêtres temporelles disjointes et ordonnées."""
    fin_train: pd.Timestamp
    fin_calib: pd.Timestamp

    @classmethod
    def par_proportions(cls, dates: pd.Series, p_train=0.6, p_calib=0.2) -> "Decoupage":
        bornes = dates.quantile([p_train, p_train + p_calib])
        return cls(fin_train=bornes.iloc[0], fin_calib=bornes.iloc[1])

    def masques(self, dates: pd.Series):
        return (
            dates <= self.fin_train,
            (dates > self.fin_train) & (dates <= self.fin_calib),
            dates > self.fin_calib,
        )


@dataclass
class ModelePmu:
    cible: str = "y_gagnant"
    avec_marche: bool = False
    colonnes: list[str] = field(default_factory=list)
    modele: object = None
    calibrateur: IsotonicRegression | None = None

    # -- interne ------------------------------------------------------

    def _matrice(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[self.colonnes].apply(pd.to_numeric, errors="coerce")

    def _nouveau_modele(self):
        if LIGHTGBM:  # pragma: no cover
            return lgb.LGBMClassifier(
                objective="binary", n_estimators=600, learning_rate=0.04,
                num_leaves=63, min_child_samples=40, subsample=0.85,
                subsample_freq=1, colsample_bytree=0.75, reg_lambda=1.0,
                random_state=0, verbose=-1,
            )
        return HistGradientBoostingClassifier(
            max_iter=500, learning_rate=0.05, max_leaf_nodes=63,
            min_samples_leaf=40, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.1, random_state=0,
        )

    @staticmethod
    def _normaliser_par_course(p: np.ndarray, courses: pd.Series) -> np.ndarray:
        """
        Ramène les scores à une distribution de probabilité par course.
        Il y a un gagnant et un seul : la somme doit valoir 1.
        """
        s = pd.Series(p, index=courses.index)
        somme = s.groupby(courses).transform("sum")
        return (s / somme.replace(0, np.nan)).fillna(0.0).to_numpy()

    # -- API ----------------------------------------------------------

    def entrainer(self, df: pd.DataFrame, decoupage: Decoupage) -> "ModelePmu":
        # `est_cible`, pas `est_exploitable` : les lignes importées ont
        # servi à bâtir l'historique, elles ne doivent pas devenir des
        # exemples — leurs colonnes de cote, gains et musique sont vides.
        df = df[df["est_cible"]].copy()
        self.colonnes = ft.colonnes_features(df, avec_marche=self.avec_marche)

        m_train, m_calib, _ = decoupage.masques(df["heure_depart"])
        if m_train.sum() == 0 or m_calib.sum() == 0:
            raise ValueError("fenêtres d'entraînement ou de calibration vides")

        X, y = self._matrice(df), df[self.cible]
        self.modele = self._nouveau_modele()
        self.modele.fit(X[m_train], y[m_train])
        log.info("entraîné sur %d partants, %d features", m_train.sum(), len(self.colonnes))

        # Calibration sur une fenêtre POSTÉRIEURE, jamais vue à l'entraînement.
        brut = self.modele.predict_proba(X[m_calib])[:, 1]
        norm = self._normaliser_par_course(brut, df.loc[m_calib, "course_id"])
        self.calibrateur = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self.calibrateur.fit(norm, y[m_calib])
        log.info("calibré sur %d partants", m_calib.sum())
        return self

    def predire(self, df: pd.DataFrame) -> pd.DataFrame:
        """Renvoie course_id, num_pmu, proba, rang — trié par course puis proba."""
        X = self._matrice(df)
        brut = self.modele.predict_proba(X)[:, 1]
        norm = self._normaliser_par_course(brut, df["course_id"])
        proba = self.calibrateur.predict(norm) if self.calibrateur is not None else norm

        out = pd.DataFrame({
            "course_id": df["course_id"].to_numpy(),
            "num_pmu": df["num_pmu"].to_numpy(),
            "proba_brute": norm,
            "proba": proba,
        }, index=df.index)
        # La calibration isotonique casse la somme à 1 : on renormalise après.
        out["proba"] = self._normaliser_par_course(out["proba"].to_numpy(), out["course_id"])
        out["rang"] = out.groupby("course_id")["proba"].rank(ascending=False, method="first")
        # Indice de confiance : écart entre le 1er et le 2e choix. Une course
        # où deux chevaux sont à 22 % n'est pas la même qu'une à 45 % / 8 %.
        top2 = out.groupby("course_id")["proba"].transform(
            lambda s: s.nlargest(2).min() if len(s) > 1 else 0.0
        )
        top1 = out.groupby("course_id")["proba"].transform("max")
        out["ecart_top2"] = top1 - top2
        return out.sort_values(["course_id", "proba"], ascending=[True, False])

    # -- persistance --------------------------------------------------

    def sauver(self, dossier: str | Path) -> None:
        import joblib
        dossier = Path(dossier)
        dossier.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"modele": self.modele, "calibrateur": self.calibrateur,
             "colonnes": self.colonnes, "cible": self.cible,
             "avec_marche": self.avec_marche},
            dossier / "modele.joblib",
        )
        (dossier / "colonnes.json").write_text(
            json.dumps(self.colonnes, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def charger(cls, dossier: str | Path) -> "ModelePmu":
        import joblib
        d = joblib.load(Path(dossier) / "modele.joblib")
        obj = cls(cible=d["cible"], avec_marche=d["avec_marche"], colonnes=d["colonnes"])
        obj.modele, obj.calibrateur = d["modele"], d["calibrateur"]
        return obj


# =====================================================================
# Modèle ordinal — exploiter l'ordre d'arrivée, pas seulement le gagnant
# =====================================================================
#
# LE CONSTAT
#
# Une course de quatorze partants n'apprend qu'une seule chose au
# modèle binaire : qui a gagné. Les treize autres places, qui disent
# pourtant quelque chose du niveau de chacun, sont jetées. À volume de
# données constant, c'est le gaspillage le plus coûteux du projet.
#
# LA MÉTHODE
#
# Décomposition ordinale : au lieu d'un modèle « gagnant / pas
# gagnant », on en entraîne un par seuil — dans les 1, les 2, les 3,
# les 5 premiers. Chacun voit une cible différente, donc une découpe
# différente de l'ordre d'arrivée. Un empileur apprend ensuite à les
# combiner en une probabilité de victoire.
#
# POURQUOI PAS UN MODÈLE DE RANG (LambdaRank)
#
# Il faudrait LightGBM, absent de l'image par défaut, et il rend un
# SCORE, pas une probabilité. Or la probabilité calibrée est le
# livrable : « 20 % » doit vouloir dire une fois sur cinq. La
# décomposition garde cette propriété et tourne partout.
#
# L'HYGIÈNE DU DÉCOUPAGE, qui est le point délicat
#
#   train    → les quatre modèles de seuil
#   calib A  → l'empileur (il voit les sorties des modèles de seuil,
#              jamais vues à l'entraînement)
#   calib B  → la calibration isotonique ET l'arbitrage
#   test     → jamais touché
#
# Sans cette coupure de `calib` en deux, l'empileur et le calibrateur
# apprendraient sur les mêmes lignes, et la calibration annoncerait une
# justesse qu'elle n'a pas.


@dataclass
class ModeleOrdinal:
    """Un modèle par seuil d'arrivée, plus un empileur qui les combine."""
    cible: str = "y_gagnant"
    avec_marche: bool = False
    colonnes: list = field(default_factory=list)
    seuils: list = field(default_factory=list)
    modeles: dict = field(default_factory=dict)
    empileur: object = None
    calibrateur: IsotonicRegression | None = None

    _matrice = ModelePmu._matrice
    _nouveau_modele = ModelePmu._nouveau_modele
    _normaliser_par_course = staticmethod(ModelePmu._normaliser_par_course)

    def _scores(self, df: pd.DataFrame) -> np.ndarray:
        """Une colonne par seuil, normalisée à l'intérieur de la course."""
        X = self._matrice(df)
        cols = []
        for nom in self.seuils:
            brut = self.modeles[nom].predict_proba(X)[:, 1]
            cols.append(self._normaliser_par_course(brut, df["course_id"]))
        return np.column_stack(cols)

    def entrainer(self, df: pd.DataFrame, decoupage: Decoupage) -> "ModeleOrdinal":
        from sklearn.linear_model import LogisticRegression

        df = df[df["est_cible"]].copy()
        self.colonnes = ft.colonnes_features(df, avec_marche=self.avec_marche)
        self.seuils = [n for n, _ in ft.SEUILS_ORDINAUX if n in df.columns
                       and df[n].nunique() > 1]
        if self.cible not in self.seuils:
            self.seuils = [self.cible] + self.seuils

        m_train, m_calib, _ = decoupage.masques(df["heure_depart"])
        if m_train.sum() == 0 or m_calib.sum() == 0:
            raise ValueError("fenêtres d'entraînement ou de calibration vides")

        X = self._matrice(df)
        for nom in self.seuils:
            modele = self._nouveau_modele()
            modele.fit(X[m_train], df.loc[m_train, nom])
            self.modeles[nom] = modele
        log.info("ordinal : %d seuils sur %d partants", len(self.seuils), m_train.sum())

        # Coupure de la fenêtre de calibration en deux moitiés
        # chronologiques — A pour l'empileur, B pour l'isotonie.
        calib = df[m_calib]
        milieu = calib["heure_depart"].quantile(0.5)
        a = calib[calib["heure_depart"] <= milieu]
        b = calib[calib["heure_depart"] > milieu]
        if len(a) < 50 or len(b) < 50 or a[self.cible].nunique() < 2:
            raise ValueError("fenêtre de calibration trop courte pour l'empilement")

        self.empileur = LogisticRegression(max_iter=1000)
        self.empileur.fit(self._scores(a), a[self.cible])

        brut = self.empileur.predict_proba(self._scores(b))[:, 1]
        norm = self._normaliser_par_course(brut, b["course_id"])
        self.calibrateur = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self.calibrateur.fit(norm, b[self.cible])
        return self

    def predire(self, df: pd.DataFrame) -> pd.DataFrame:
        brut = self.empileur.predict_proba(self._scores(df))[:, 1]
        norm = self._normaliser_par_course(brut, df["course_id"])
        proba = self.calibrateur.predict(norm) if self.calibrateur is not None else norm
        out = pd.DataFrame({
            "course_id": df["course_id"].to_numpy(),
            "num_pmu": df["num_pmu"].to_numpy(),
            "proba_brute": norm, "proba": proba,
        }, index=df.index)
        out["proba"] = self._normaliser_par_course(out["proba"].to_numpy(), out["course_id"])
        out["rang"] = out.groupby("course_id")["proba"].rank(ascending=False, method="first")
        top2 = out.groupby("course_id")["proba"].transform(
            lambda s: s.nlargest(2).min() if len(s) > 1 else 0.0)
        top1 = out.groupby("course_id")["proba"].transform("max")
        out["ecart_top2"] = top1 - top2
        return out.sort_values(["course_id", "proba"], ascending=[True, False])

    def sauver(self, dossier: str | Path) -> None:
        import joblib
        dossier = Path(dossier)
        dossier.mkdir(parents=True, exist_ok=True)
        joblib.dump({"modeles": self.modeles, "empileur": self.empileur,
                     "calibrateur": self.calibrateur, "colonnes": self.colonnes,
                     "seuils": self.seuils, "cible": self.cible,
                     "avec_marche": self.avec_marche},
                    dossier / "ordinal.joblib")

    @classmethod
    def charger(cls, dossier: str | Path) -> "ModeleOrdinal":
        import joblib
        d = joblib.load(Path(dossier) / "ordinal.joblib")
        o = cls(cible=d["cible"], avec_marche=d["avec_marche"],
                colonnes=d["colonnes"], seuils=d["seuils"])
        o.modeles, o.empileur, o.calibrateur = d["modeles"], d["empileur"], d["calibrateur"]
        return o


# =====================================================================
# Modèles par famille de discipline
# =====================================================================
#
# CE QUI A MOTIVÉ CE DÉCOUPAGE
#
# Sur 60 jours de données réelles, le gain du modèle face au marché
# variait du simple au quintuple selon la discipline : +0,137 d'AUC en
# monté, +0,061 en attelé, +0,029 seulement en plat — alors que le plat
# représente la moitié des partants. Un modèle unique arbitre ces
# régimes en les moyennant, donc les dessert tous.
#
# CE QUE CE CODE NE FAIT PAS
#
# Il ne suppose pas que scinder aide. Un modèle spécialisé voit moins
# d'exemples ; en dessous d'un certain volume il perd plus en effectif
# qu'il ne gagne en homogénéité, et ça ne se devine pas. Chaque famille
# est donc mise en concurrence avec le modèle global, et le spécialisé
# n'est retenu que s'il gagne — sur une fenêtre qu'AUCUN des deux
# modèles n'a vue à l'entraînement.
#
# POURQUOI L'ARBITRAGE SE FAIT SUR LA CALIBRATION, PAS SUR LE TEST
#
# Choisir sur le test, c'est le consommer : le score rapporté ensuite
# n'est plus une mesure hors échantillon, il est optimiste de tout ce
# qu'a rapporté le choix. La fenêtre de calibration convient, et le
# critère retenu est l'AUC — la régression isotonique étant monotone,
# elle ne change aucun classement, donc l'AUC lue sur cette fenêtre
# reflète le seul modèle de base, entraîné sur `train` uniquement.

# Volumes en dessous desquels on ne tente même pas la spécialisation.
MIN_TRAIN_FAMILLE = 3000
MIN_CALIB_FAMILLE = 800
# Décomposition ordinale active par défaut ; `PMU_ORDINAL=0` revient à la
# cible binaire seule, pour comparer les deux sur ta propre base.
ORDINAL_ACTIF = os.environ.get("PMU_ORDINAL", "1").strip() not in ("0", "non", "false")
# Marge exigée pour préférer le spécialisé. Un écart d'AUC de 0,002 sur
# quelques milliers de lignes est du bruit ; on ne complique pas la pile
# pour du bruit.
MARGE_AUC = 0.005


@dataclass
class ModeleParDiscipline:
    """
    Un modèle global, plus un modèle spécialisé par famille — mais
    seulement là où le spécialisé fait ses preuves. À la prédiction,
    chaque course part vers le modèle retenu pour sa famille.
    """
    cible: str = "y_gagnant"
    avec_marche: bool = False
    global_: ModelePmu | None = None
    par_famille: dict = field(default_factory=dict)
    arbitrage: dict = field(default_factory=dict)
    arbitrage_global: dict = field(default_factory=dict)

    # -- interne ------------------------------------------------------

    @staticmethod
    def _familles(df: pd.DataFrame) -> pd.Series:
        if "famille" in df.columns:
            return df["famille"]
        return ft.famille(df["discipline"])

    def _auc(self, modele, sous_ensemble: pd.DataFrame):
        """AUC du modèle sur un sous-ensemble, ou None si non calculable."""
        from sklearn.metrics import roc_auc_score
        y = sous_ensemble[self.cible]
        if len(sous_ensemble) < 50 or y.nunique() < 2:
            return None
        p = modele.predire(sous_ensemble)["proba"].reindex(sous_ensemble.index)
        return float(roc_auc_score(y, p))

    def _binaire_ou_ordinal(self, sub: pd.DataFrame, decoupage: Decoupage,
                            fiche: dict):
        """
        Entraîne les deux approches et garde la meilleure — mesurée, pas
        supposée. La comparaison porte sur la SECONDE moitié de la
        fenêtre de calibration : l'empileur du modèle ordinal n'a vu que
        la première, et les modèles de base des deux n'ont vu que
        l'entraînement. Le test reste intact.
        """
        binaire = ModelePmu(cible=self.cible,
                            avec_marche=self.avec_marche).entrainer(sub, decoupage)
        if not ORDINAL_ACTIF:
            fiche["cible"] = "binaire"
            return binaire

        _, m_calib, _ = decoupage.masques(sub["heure_depart"])
        calib = sub[m_calib]
        b = calib[calib["heure_depart"] > calib["heure_depart"].quantile(0.5)]
        try:
            ordinal = ModeleOrdinal(cible=self.cible,
                                    avec_marche=self.avec_marche).entrainer(sub, decoupage)
        except (ValueError, KeyError) as exc:
            fiche["cible"] = "binaire"
            fiche["motif_cible"] = f"ordinal impossible : {exc}"
            return binaire

        a_ord, a_bin = self._auc(ordinal, b), self._auc(binaire, b)
        fiche["auc_ordinal"] = None if a_ord is None else round(a_ord, 4)
        fiche["auc_binaire"] = None if a_bin is None else round(a_bin, 4)
        if a_ord is not None and a_bin is not None and a_ord > a_bin + MARGE_AUC:
            fiche["cible"] = "ordinale"
            fiche["gain_cible"] = round(a_ord - a_bin, 4)
            return ordinal
        fiche["cible"] = "binaire"
        if a_ord is not None and a_bin is not None:
            fiche["gain_cible"] = round(a_ord - a_bin, 4)
        return binaire

    # -- API ----------------------------------------------------------

    def entrainer(self, df: pd.DataFrame, decoupage: Decoupage) -> "ModeleParDiscipline":
        df = df[df["est_cible"]].copy()
        df["famille"] = self._familles(df)

        # Fiche du modèle global tenue À PART : `arbitrage` ne contient
        # que des familles, et tout ce qui le lit compte dessus.
        self.arbitrage_global = {"n_total": int(len(df))}
        self.global_ = self._binaire_ou_ordinal(df, decoupage, self.arbitrage_global)

        self.par_famille, self.arbitrage = {}, {}
        for fam, sub in df.groupby("famille", sort=True):
            m_train, m_calib, _ = decoupage.masques(sub["heure_depart"])
            n_tr, n_ca = int(m_train.sum()), int(m_calib.sum())
            fiche = {"n_total": int(len(sub)), "n_train": n_tr, "n_calib": n_ca}

            if n_tr < MIN_TRAIN_FAMILLE or n_ca < MIN_CALIB_FAMILLE:
                fiche["decision"] = "global"
                fiche["motif"] = (f"volume insuffisant ({n_tr} entraînement, "
                                  f"{n_ca} calibration)")
                self.arbitrage[str(fam)] = fiche
                continue

            try:
                spec = self._binaire_ou_ordinal(sub, decoupage, fiche)
            except ValueError as exc:          # fenêtre vide malgré le comptage
                fiche["decision"] = "global"
                fiche["motif"] = f"entraînement impossible : {exc}"
                self.arbitrage[str(fam)] = fiche
                continue

            calib = sub[m_calib]
            auc_spec, auc_glob = self._auc(spec, calib), self._auc(self.global_, calib)
            fiche["auc_specialise"] = None if auc_spec is None else round(auc_spec, 4)
            fiche["auc_global"] = None if auc_glob is None else round(auc_glob, 4)

            if auc_spec is None or auc_glob is None:
                fiche["decision"] = "global"
                fiche["motif"] = "AUC non calculable sur la calibration"
            elif auc_spec > auc_glob + MARGE_AUC:
                fiche["decision"] = "specialise"
                fiche["gain_auc"] = round(auc_spec - auc_glob, 4)
                self.par_famille[str(fam)] = spec
            else:
                fiche["decision"] = "global"
                fiche["gain_auc"] = round(auc_spec - auc_glob, 4)
                fiche["motif"] = f"gain insuffisant (marge exigée {MARGE_AUC})"
            self.arbitrage[str(fam)] = fiche

        log.info("arbitrage par famille\n%s",
                 resumer_arbitrage(self.arbitrage, self.arbitrage_global))
        return self

    def predire(self, df: pd.DataFrame) -> pd.DataFrame:
        familles = self._familles(df)
        morceaux = []
        for fam, sub in df.groupby(familles, sort=False):
            modele = self.par_famille.get(str(fam)) or self.global_
            morceaux.append(modele.predire(sub))
        out = pd.concat(morceaux)
        return out.sort_values(["course_id", "proba"], ascending=[True, False])

    # -- persistance --------------------------------------------------

    def sauver(self, dossier: str | Path) -> None:
        dossier = Path(dossier)
        dossier.mkdir(parents=True, exist_ok=True)
        self.global_.sauver(dossier / "global")
        for fam, m in self.par_famille.items():
            m.sauver(dossier / "familles" / fam)
        (dossier / "meta.json").write_text(
            json.dumps({"type": "par_discipline", "cible": self.cible,
                        "avec_marche": self.avec_marche,
                        "familles": sorted(self.par_famille),
                        "arbitrage": self.arbitrage,
                        "arbitrage_global": self.arbitrage_global},
                       indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def charger(cls, dossier: str | Path) -> "ModeleParDiscipline":
        dossier = Path(dossier)
        meta = json.loads((dossier / "meta.json").read_text(encoding="utf-8"))
        obj = cls(cible=meta["cible"], avec_marche=meta["avec_marche"],
                  arbitrage=meta.get("arbitrage", {}),
                  arbitrage_global=meta.get("arbitrage_global", {}))
        obj.global_ = _charger_un(dossier / "global")
        obj.par_famille = {
            fam: _charger_un(dossier / "familles" / fam)
            for fam in meta.get("familles", [])
        }
        return obj


def _charger_un(dossier: Path):
    """
    Recharge un sous-modèle sans savoir lequel des deux il est.
    L'arbitrage peut avoir retenu une cible ordinale pour l'attelé et
    binaire pour le plat : le format n'est donc pas uniforme d'un
    dossier à l'autre.
    """
    dossier = Path(dossier)
    if (dossier / "ordinal.joblib").exists():
        return ModeleOrdinal.charger(dossier)
    return ModelePmu.charger(dossier)


def resumer_arbitrage(arb: dict, global_: dict | None = None) -> str:
    """Tableau lisible : qui a été spécialisé, qui ne l'a pas été, pourquoi."""
    if not arb and not global_:
        return "  (aucune famille)"
    L = []
    g = global_
    if g:
        L.append(f"  cible retenue au global : {g.get('cible', '?')}"
                 + (f" (AUC {g['auc_ordinal']:.4f} ordinale contre "
                    f"{g['auc_binaire']:.4f} binaire)"
                    if g.get("auc_ordinal") is not None
                    and g.get("auc_binaire") is not None else ""))
        L.append("")
    L.append(f"  {'famille':<10} {'partants':>9} {'AUC spéc.':>10} {'AUC glob.':>10} "
             f"{'gain':>8}  décision")
    for fam, f in sorted(arb.items()):
        spec = f.get("auc_specialise")
        glob = f.get("auc_global")
        gain = f.get("gain_auc")
        L.append(
            f"  {fam:<10} {f['n_total']:>9} "
            f"{('—' if spec is None else f'{spec:.4f}'):>10} "
            f"{('—' if glob is None else f'{glob:.4f}'):>10} "
            f"{('—' if gain is None else f'{gain:+.4f}'):>8}  "
            f"{'SPÉCIALISÉ' if f['decision'] == 'specialise' else 'global'}"
            + (f" · cible {f['cible']}" if f.get("cible") else "")
            + (f" ({f['motif']})" if f.get("motif") else "")
        )
    L.append("  → « global » n'est pas un échec : c'est la mesure qui dit que")
    L.append("    scinder cette famille coûterait plus d'exemples qu'il ne")
    L.append("    rapporterait d'homogénéité.")
    return "\n".join(L)


def charger_modele(dossier: str | Path):
    """
    Recharge un modèle sans savoir d'avance s'il est unique ou scindé.
    Les modèles d'avant l'étape 4 n'ont pas de `meta.json` : ils se
    rechargent comme avant, sans rien casser.
    """
    dossier = Path(dossier)
    meta = dossier / "meta.json"
    if meta.exists():
        try:
            if json.loads(meta.read_text(encoding="utf-8")).get("type") == "par_discipline":
                return ModeleParDiscipline.charger(dossier)
        except (ValueError, OSError) as exc:
            log.warning("meta.json illisible dans %s (%s) — repli sur le modèle unique",
                        dossier, exc)
    return _charger_un(dossier)


def modele_present(dossier: str | Path) -> bool:
    """Vrai si un modèle rechargeable existe à cet endroit."""
    dossier = Path(dossier)
    return any((dossier / n).exists() for n in ("modele.joblib", "ordinal.joblib")) or \
        any((dossier / "global" / n).exists() for n in ("modele.joblib", "ordinal.joblib"))


def importances(modele: ModelePmu, df: pd.DataFrame, n: int = 30) -> pd.DataFrame:
    """
    Importances par permutation, sur la fenêtre de test.

    Les importances natives d'un arbre gonflent les variables à forte
    cardinalité ; la permutation mesure ce qu'on perd RÉELLEMENT en cassant
    la variable. Plus lent, mais c'est la seule lecture honnête.
    """
    from sklearn.inspection import permutation_importance

    X = modele._matrice(df)
    r = permutation_importance(
        modele.modele, X, df[modele.cible], n_repeats=5,
        random_state=0, scoring="neg_log_loss",
    )
    return (
        pd.DataFrame({"feature": modele.colonnes,
                      "importance": r.importances_mean,
                      "ecart_type": r.importances_std})
        .sort_values("importance", ascending=False)
        .head(n)
        .reset_index(drop=True)
    )
