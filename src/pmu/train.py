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
        df = df[df["est_exploitable"]].copy()
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
