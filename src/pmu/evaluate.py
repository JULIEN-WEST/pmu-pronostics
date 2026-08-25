"""
Évaluation.

L'AUC ne suffit pas. Elle mesure un pouvoir de discrimination, pas la
justesse des probabilités annoncées, et c'est cette justesse qui a été
demandée : « selon des vérités ou des degrés de confiance ».

Quatre lectures, de la plus technique à la plus décisive :

  1. Justesse       Brier, log-loss, AUC.
  2. Calibration    quand le modèle dit 20 %, ça arrive-t-il 20 % du temps ?
  3. Le marché      bat-on le favori du public ? C'est LA référence.
  4. Rentabilité    l'écart survit-il au prélèvement ? C'est LE juge de paix.

Sur le point 4, un avertissement qui n'est pas de la prudence de façade mais
une contrainte arithmétique : le PMU est un pari MUTUEL. Les mises sont
réparties entre gagnants après prélèvement, donc :

  - la cote n'est pas fixée par un bookmaker qu'on pourrait prendre en
    défaut : elle EST la répartition des mises du public, c'est-à-dire le
    consensus de tous les autres parieurs, dont beaucoup sont outillés ;
  - le prélèvement (~15 % sur le Simple, davantage sur les paris combinés)
    est retiré AVANT répartition. Un modèle qui égale le marché perd donc
    ce pourcentage à chaque tour ;
  - vos propres mises font baisser la cote que vous venez de viser.

Conséquence pratique : le seuil de réussite n'est pas « mieux que le
hasard », ni même « mieux que le favori ». C'est « mieux que le marché,
d'une marge supérieure au prélèvement, et de façon stable sur plusieurs
centaines de courses ». Ce harnais est fait pour mesurer ça honnêtement,
y compris quand la réponse est non.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

# Prélèvement moyen. À VÉRIFIER sur les rapports réels avant toute
# conclusion : il dépend du type de pari (le Simple est le moins taxé,
# les paris combinés le sont beaucoup plus).
PRELEVEMENT_DEFAUT = 0.15


# ---------------------------------------------------------------------
# 1. Justesse
# ---------------------------------------------------------------------

def justesse(y: pd.Series, p: pd.Series) -> dict:
    y, p = np.asarray(y, float), np.clip(np.asarray(p, float), 1e-9, 1 - 1e-9)
    base = float(y.mean())
    return {
        "n": int(len(y)),
        "taux_base": round(base, 4),
        "brier": round(float(brier_score_loss(y, p)), 5),
        # Skill score : gain relatif face au modèle « tout le monde à la
        # moyenne ». Négatif = le modèle fait pire que ne rien savoir.
        "brier_skill": round(1 - brier_score_loss(y, p) / (base * (1 - base) or 1e-9), 4),
        "log_loss": round(float(log_loss(y, p, labels=[0, 1])), 5),
        "auc": round(float(roc_auc_score(y, p)), 4) if 0 < base < 1 else None,
    }


# ---------------------------------------------------------------------
# 2. Calibration
# ---------------------------------------------------------------------

def table_calibration(y: pd.Series, p: pd.Series, n_bins: int = 10) -> pd.DataFrame:
    """
    Une ligne par tranche de probabilité annoncée.
    `ecart` proche de 0 partout = modèle calibré, la confiance a un sens.
    """
    df = pd.DataFrame({"y": np.asarray(y, float), "p": np.asarray(p, float)})
    df["tranche"] = pd.qcut(df["p"], q=n_bins, duplicates="drop")
    out = df.groupby("tranche", observed=True).agg(
        n=("y", "size"), predit=("p", "mean"), observe=("y", "mean")
    ).reset_index()
    out["ecart"] = (out["observe"] - out["predit"]).round(4)
    out["predit"] = out["predit"].round(4)
    out["observe"] = out["observe"].round(4)
    return out


def erreur_calibration(y: pd.Series, p: pd.Series, n_bins: int = 10) -> float:
    """ECE : écart moyen pondéré entre annoncé et observé. Plus bas, mieux."""
    t = table_calibration(y, p, n_bins)
    return round(float((t["ecart"].abs() * t["n"]).sum() / t["n"].sum()), 5)


# ---------------------------------------------------------------------
# 3. Comparaison au marché
# ---------------------------------------------------------------------

def face_au_marche(df: pd.DataFrame, col_proba="proba",
                   col_marche="mkt_proba_implicite", cible="y_gagnant") -> dict:
    """
    Le seul comparatif qui compte. Battre le hasard est facile ;
    battre le public l'est beaucoup moins.
    """
    res: dict = {}
    par_course = df.groupby("course_id")

    def top1(colonne: str) -> float:
        idx = par_course[colonne].idxmax()
        return float(df.loc[idx, cible].mean())

    res["top1_modele"] = round(top1(col_proba), 4)
    res["n_courses"] = int(df["course_id"].nunique())

    if col_marche in df.columns and df[col_marche].notna().any():
        sub = df[df[col_marche].notna()]
        if len(sub):
            idx = sub.groupby("course_id")[col_marche].idxmax()
            res["top1_favori"] = round(float(sub.loc[idx, cible].mean()), 4)
            res["ecart_top1"] = round(res["top1_modele"] - res["top1_favori"], 4)
            res["brier_modele"] = round(
                float(brier_score_loss(sub[cible], sub[col_proba].clip(1e-9, 1 - 1e-9))), 5)
            res["brier_marche"] = round(
                float(brier_score_loss(sub[cible], sub[col_marche].clip(1e-9, 1 - 1e-9))), 5)
            res["modele_bat_marche"] = bool(res["brier_modele"] < res["brier_marche"])
            # Corrélation des désaccords : si elle est très haute, le modèle
            # ne fait que redire le marché et n'apportera jamais rien.
            res["correlation_au_marche"] = round(
                float(sub[col_proba].corr(sub[col_marche])), 4)
    return res


# ---------------------------------------------------------------------
# 4. Rentabilité
# ---------------------------------------------------------------------

def simulation(df: pd.DataFrame, *, col_proba="proba", col_cote="mkt_cote",
               cible="y_gagnant", seuils_valeur=(0.0, 0.10, 0.20, 0.30, 0.50),
               prelevement: float = PRELEVEMENT_DEFAUT,
               mise: float = 1.0, col_reel: str | None = None) -> pd.DataFrame:
    """
    Mise à plat sur les partants dont la « valeur » dépasse un seuil.

        valeur = proba_modele × cote − 1

    C'est l'espérance de gain par euro misé. Positive en théorie = pari
    intéressant. En pratique il faut qu'elle dépasse le prélèvement ET que
    l'échantillon soit assez grand pour que ce ne soit pas du bruit.

    La colonne `n_paris` est à lire en premier : un retour de +40 % sur
    23 paris ne veut strictement rien dire. En dessous de ~500 paris,
    aucune conclusion.
    """
    if col_cote not in df.columns or df[col_cote].isna().all():
        return pd.DataFrame()

    d = df[df[col_cote].notna() & (df[col_cote] > 1)].copy()
    d["valeur"] = d[col_proba] * d[col_cote] - 1.0

    # ── Le retour : estimé, ou mesuré ────────────────────────────────
    #
    # Par défaut on ESTIME le retour à partir de la cote relevée avant
    # le départ, corrigée du prélèvement. Deux approximations s'y
    # cachent, et elles vont en sens CONTRAIRE :
    #
    #   — la cote est peut-être déjà nette (le ×(1−t) serait de trop) ;
    #   — la cote pré-départ n'est pas le rapport payé : mesuré sur 284
    #     gagnants, le rapport vaut 0,894 fois la cote annoncée, parce
    #     que l'argent des dernières minutes raccourcit le prix.
    #
    # Quand `col_reel` est fourni ET renseigné, on ne suppose plus rien :
    # on paie ce qui a été réellement payé. C'est la seule mesure qui
    # n'ait aucun paramètre.
    reel = bool(col_reel and col_reel in d.columns and d[col_reel].notna().any())
    if reel:
        # Une course sans rapport connu doit être ÉCARTÉE, pas comptée
        # perdante : sinon chaque rapport manquant devient une défaite
        # imaginaire et le ROI s'effondre pour rien.
        connues = set(d.loc[d[col_reel].notna(), "course_id"].unique())
        avant = len(d)
        d = d[d["course_id"].isin(connues)].copy()
        ecartes = avant - len(d)
        d["_paye"] = d[col_reel].fillna(0.0)
    else:
        ecartes = 0
        d["_paye"] = d[col_cote] * (1.0 - prelevement)

    lignes = []
    for seuil in seuils_valeur:
        paris = d[d["valeur"] >= seuil]
        n = len(paris)
        if n == 0:
            lignes.append({"seuil_valeur": seuil, "n_paris": 0})
            continue
        gagnes = paris[cible].sum()
        gain = paris[cible].mul(paris["_paye"]).mul(mise)   # 0 si le pari perd
        retour = float(gain.sum())
        engage = n * mise
        lignes.append({
            "seuil_valeur": seuil,
            "n_paris": n,
            "n_gagnants": int(gagnes),
            "taux_reussite": round(float(gagnes / n), 4),
            "cote_moyenne": round(float(paris[col_cote].mean()), 2),
            "engage": round(engage, 2),
            "retour": round(retour, 2),
            "roi_pct": round(float((retour - engage) / engage * 100), 2),
            # Écart-type du ROI : sans lui, le ROI n'est pas interprétable.
            "roi_ecart_type_pct": round(
                float(gain.sub(mise).div(mise).std() / np.sqrt(n) * 100), 2),
            "mesure": bool(reel),
            "courses_ecartees": int(ecartes) if seuil == seuils_valeur[0] else None,
        })
    return pd.DataFrame(lignes)


# ---------------------------------------------------------------------
# Rapport complet
# ---------------------------------------------------------------------

def rapport(df: pd.DataFrame, cible="y_gagnant", prelevement=PRELEVEMENT_DEFAUT,
            col_reel: str = "rapport_reel") -> dict:
    """
    `df` = fenêtre de TEST, avec au minimum : course_id, proba, la cible,
    et si possible mkt_cote / mkt_proba_implicite.
    """
    out = {
        "justesse": justesse(df[cible], df["proba"]),
        "ece": erreur_calibration(df[cible], df["proba"]),
        "calibration": table_calibration(df[cible], df["proba"]).to_dict("records"),
        "marche": face_au_marche(df, cible=cible),
    }
    sim = simulation(df, cible=cible, prelevement=prelevement)
    # `_json_sur` n'est pas cosmétique ici : pandas rend des np.bool_ et
    # des np.int64 que `json.dumps` refuse, et un rapport illisible en
    # JSON casse l'endpoint au moment précis où on vient l'inspecter.
    out["rentabilite"] = _json_sur(sim.to_dict("records")) if len(sim) else []

    # La même simulation, mais payée aux rapports RÉELLEMENT versés.
    # Elle ne remplace pas l'autre : les deux côte à côte montrent de
    # combien l'estimation se trompait, et dans quel sens.
    if col_reel and col_reel in df.columns and df[col_reel].notna().any():
        reelle = simulation(df, cible=cible, prelevement=prelevement,
                            col_reel=col_reel)
        out["rentabilite_reelle"] = (
            _json_sur(reelle.to_dict("records")) if len(reelle) else [])

    # Stratification par discipline : c'est le contrôle qui dira si le
    # modèle unique tient, ou s'il faut le scinder. On ne le devine pas,
    # on le mesure.
    if "discipline" in df.columns:
        par_disc = {}
        for disc, sub in df.groupby("discipline"):
            if len(sub) < 200 or sub[cible].nunique() < 2:
                continue
            par_disc[str(disc)] = {
                **justesse(sub[cible], sub["proba"]),
                "ece": erreur_calibration(sub[cible], sub["proba"], n_bins=5),
            }
        out["par_discipline"] = par_disc
    return out


# ---------------------------------------------------------------------
# 4 bis. Abstention — savoir se taire
# ---------------------------------------------------------------------
#
# Un modèle qui pronostique les quarante courses du jour se trompe sur
# les trois quarts, et c'est normal : il y a une seule bonne réponse
# parmi quatorze. Mais toutes les courses ne se valent pas. Quand le
# modèle détache nettement un partant, il a vu quelque chose ; quand il
# donne 9 %, 9 % et 8 % aux trois premiers, il n'a rien vu du tout et
# le dire serait plus honnête que de désigner quelqu'un.
#
# `ecart_top2` — la distance entre le 1ᵉʳ et le 2ᵉ choix — mesure
# exactement ça. Reste à savoir à partir de quelle valeur le modèle
# mérite d'être écouté. On ne le décide pas : on le mesure.

# En dessous, une bande n'a pas assez de courses pour qu'on lui fasse
# dire quoi que ce soit.
MIN_COURSES_BANDE = 60


def bandes_confiance(df: pd.DataFrame, *, cible="y_gagnant", n_bandes=5) -> pd.DataFrame:
    """
    Réussite du favori du modèle, tranche de confiance par tranche.

    `df` doit contenir course_id, proba, ecart_top2 et la cible — donc
    une fenêtre de test déjà notée.
    """
    if "ecart_top2" not in df.columns or df.empty:
        return pd.DataFrame()

    par_course = df.groupby("course_id")
    favoris = df.loc[par_course["proba"].idxmax()].copy()
    if len(favoris) < n_bandes * 2:
        return pd.DataFrame()

    favoris["bande"] = pd.qcut(favoris["ecart_top2"], q=n_bandes, duplicates="drop")
    lignes = []
    for bande, sub in favoris.groupby("bande", observed=True):
        n = len(sub)
        reussites = int(sub[cible].sum())
        ligne = {
            "seuil_bas": round(float(bande.left), 5),
            "n_courses": n,
            "reussite": round(reussites / n, 4),
        }
        # Le favori du public sur les mêmes courses : c'est la seule
        # comparaison qui dise si écouter le modèle apporte quelque chose.
        if "mkt_proba_implicite" in df.columns:
            memes = df[df["course_id"].isin(sub["course_id"])]
            memes = memes[memes["mkt_proba_implicite"].notna()]
            if len(memes):
                idx = memes.groupby("course_id")["mkt_proba_implicite"].idxmax()
                ligne["reussite_marche"] = round(float(memes.loc[idx, cible].mean()), 4)
        lignes.append(ligne)
    return pd.DataFrame(lignes)


def seuil_abstention(bandes: pd.DataFrame) -> float | None:
    """
    Plus petit écart 1ᵉʳ/2ᵉ à partir duquel le modèle fait au moins aussi
    bien que le favori du public, ET s'y tient sur toutes les bandes
    supérieures.

    La seconde condition est celle qui compte : une bande isolée qui
    dépasse le marché est probablement du bruit. Ce qu'on cherche, c'est
    un régime — « au-delà de tel écart, ça tient ».

    Renvoie None quand aucun seuil ne convient : le modèle doit alors se
    taire partout, et c'est une réponse valable.
    """
    if bandes.empty or "reussite_marche" not in bandes.columns:
        return None
    b = bandes[bandes["n_courses"] >= MIN_COURSES_BANDE].sort_values("seuil_bas")
    if b.empty:
        return None
    for i in range(len(b)):
        reste = b.iloc[i:]
        if (reste["reussite"] >= reste["reussite_marche"]).all():
            return float(reste.iloc[0]["seuil_bas"])
    return None


def afficher_bandes(bandes: pd.DataFrame, seuil: float | None) -> str:
    if bandes.empty:
        return "── Abstention " + "─" * 45 + "\n  pas assez de courses pour trancher"
    L = ["── Abstention " + "─" * 45,
         f"  {'écart ≥':>9} {'courses':>8} {'modèle':>9} {'marché':>9}"]
    for r in bandes.itertuples():
        marche = getattr(r, "reussite_marche", None)
        L.append(f"  {r.seuil_bas:>9.3f} {r.n_courses:>8} {r.reussite:>9.1%} "
                 f"{('—' if marche is None or pd.isna(marche) else f'{marche:.1%}'):>9}")
    if seuil is None:
        L.append("  → aucun seuil ne tient : le modèle n'égale le marché sur")
        L.append("    aucun régime de confiance. Rien n'est filtré — on ne")
        L.append("    masque pas ce qu'on n'a pas su départager — mais aucune")
        L.append("    course ne doit être tenue pour fiable pour autant.")
    else:
        n = int(bandes.loc[bandes["seuil_bas"] >= seuil, "n_courses"].sum())
        total = int(bandes["n_courses"].sum())
        L.append(f"  → seuil retenu : écart ≥ {seuil:.3f}, soit {n} courses "
                 f"sur {total} ({n / total:.0%})")
        L.append("    En dessous, le modèle n'apporte rien face au public :")
        L.append("    mieux vaut ne rien annoncer que d'annoncer au hasard.")
    return "\n".join(L)


# ---------------------------------------------------------------------
# 5. Bilan de PRODUCTION
# ---------------------------------------------------------------------
#
# Les mesures ci-dessus portent sur la fenêtre de test d'un
# entraînement. Celle-ci porte sur les pronostics RÉELLEMENT PUBLIÉS,
# relus dans la table `pronostic` et confrontés aux arrivées. C'est le
# seul chiffre qui compte, et le seul qui ne puisse pas être flatté par
# un choix de découpage.
#
# Elle répond aussi à la question qu'on se pose en regardant un tableau
# de bord : « tous mes favoris sont battus, le modèle est-il cassé ? »
# Sur dix courses, non — on ne peut rien conclure. L'intervalle de
# confiance affiché le dit à la place de l'intuition.

def _intervalle_binomial(succes: int, n: int) -> tuple[float, float]:
    """
    Intervalle de Wilson à 95 %. Plus honnête que l'approximation
    normale sur les petits effectifs, précisément le cas où l'on est
    tenté de conclure trop vite.
    """
    if n == 0:
        return (0.0, 1.0)
    z, p = 1.96, succes / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    demi = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - demi), min(1.0, centre + demi))


SQL_BILAN = """
SELECT c.course_id, c.discipline, c.date_reunion,
       pr.num_pmu, pr.proba, pr.rang, pr.cote,
       px.rang_expert,
       p.ordre_arrivee,
       (SELECT count(*) FROM partant px
         WHERE px.course_id = c.course_id AND px.ordre_arrivee = 1) AS a_un_gagnant
  FROM pronostic pr
  JOIN course  c ON c.course_id = pr.course_id
  JOIN partant p ON p.course_id = pr.course_id AND p.num_pmu = pr.num_pmu
  LEFT JOIN pronostic_expert px ON px.course_id = pr.course_id
                                AND px.num_pmu = pr.num_pmu
 WHERE pr.modele = %(modele)s
   AND c.date_reunion BETWEEN %(depuis)s AND %(jusqua)s
   AND c.ordre_arrivee IS NOT NULL
"""


def _json_sur(valeur):
    """
    Rend une valeur encodable en JSON.

    NaN et ±inf sont des flottants parfaitement légitimes en Python et
    en NumPy, mais ILLÉGAUX en JSON : `json.dumps` lève dessus. Le
    calibrage rend NaN quand une tranche est vide, ce qui suffit à faire
    répondre 500 à l'endpoint `/bilan` — c'est-à-dire précisément quand
    on vient voir si quelque chose ne va pas.
    """
    if isinstance(valeur, dict):
        return {k: _json_sur(v) for k, v in valeur.items()}
    if isinstance(valeur, (list, tuple)):
        return [_json_sur(v) for v in valeur]
    if isinstance(valeur, (float, np.floating)):
        v = float(valeur)
        return None if (np.isnan(v) or np.isinf(v)) else v
    if isinstance(valeur, (np.integer,)):
        return int(valeur)
    if isinstance(valeur, (np.bool_,)):
        return bool(valeur)
    return valeur


def bilan_production(conn, *, modele: str, depuis, jusqua) -> dict:
    """
    Le tableau de bord honnête : ce que les pronostics publiés ont donné.

    Renvoie aussi un champ `anomalies`, et il faut le lire EN PREMIER.
    Une course arrivée dont aucun partant n'a `ordre_arrivee = 1` est un
    défaut de collecte, pas une contre-performance : elle compterait
    comme un échec du modèle alors qu'elle ne prouve rien.
    """
    import psycopg.rows

    with conn.cursor(row_factory=psycopg.rows.tuple_row) as cur:
        cur.execute(SQL_BILAN, {"modele": modele, "depuis": depuis, "jusqua": jusqua})
        colonnes = [d.name for d in cur.description]
        df = pd.DataFrame(cur.fetchall(), columns=colonnes)

    if df.empty:
        return {"n_courses": 0, "message": "aucun pronostic confronté à une arrivée"}

    df["proba"] = pd.to_numeric(df["proba"], errors="coerce")
    df["cote"] = pd.to_numeric(df["cote"], errors="coerce")
    df["y"] = (pd.to_numeric(df["ordre_arrivee"], errors="coerce") == 1).astype(float)

    # ── Anomalies de collecte, isolées AVANT toute conclusion ──────
    sans_gagnant = df.loc[df["a_un_gagnant"] == 0, "course_id"].nunique()
    df = df[df["a_un_gagnant"] > 0]
    if df.empty:
        return {"n_courses": 0, "anomalies": {"courses_sans_gagnant": int(sans_gagnant)},
                "message": "aucune course exploitable : les arrivées ne sont pas "
                           "renseignées au niveau des partants"}

    par_course = df.groupby("course_id")
    choix = df.loc[par_course["proba"].idxmax()]
    n = int(choix["course_id"].nunique())
    succes = int(choix["y"].sum())
    bas, haut = _intervalle_binomial(succes, n)

    out = {
        "modele": modele,
        "n_courses": n,
        "n_partants": int(len(df)),
        "top1_reussites": succes,
        "top1_taux": round(succes / n, 4),
        "top1_ic95": [round(bas, 4), round(haut, 4)],
        "brier": round(float(brier_score_loss(df["y"], df["proba"].clip(1e-9, 1 - 1e-9))), 5),
        "ece": erreur_calibration(df["y"], df["proba"]),
        "anomalies": {"courses_sans_gagnant": int(sans_gagnant)},
    }

    # Le favori du public : la cote la plus basse.
    avec_cote = df[df["cote"].notna() & (df["cote"] > 1)]
    if len(avec_cote):
        fav = avec_cote.loc[avec_cote.groupby("course_id")["cote"].idxmin()]
        n_m = int(fav["course_id"].nunique())
        s_m = int(fav["y"].sum())
        out["marche"] = {
            "n_courses": n_m, "top1_reussites": s_m,
            "top1_taux": round(s_m / n_m, 4) if n_m else None,
            "top1_ic95": [round(x, 4) for x in _intervalle_binomial(s_m, n_m)],
        }

    # Troisième repère : l'analyste. Le modèle n'a pas à battre le
    # hasard, il a à battre ce que n'importe qui peut lire gratuitement
    # avant la course.
    if "rang_expert" in df.columns and df["rang_expert"].notna().any():
        exp = df[df["rang_expert"].notna()]
        idx = exp.groupby("course_id")["rang_expert"].idxmin()
        n_e, s_e = int(exp["course_id"].nunique()), int(exp.loc[idx, "y"].sum())
        out["expert"] = {
            "n_courses": n_e, "top1_reussites": s_e,
            "top1_taux": round(s_e / n_e, 4) if n_e else None,
            "top1_ic95": [round(x, 4) for x in _intervalle_binomial(s_e, n_e)],
        }

    out["par_discipline"] = {}
    for disc, sub in choix.groupby("discipline"):
        k = int(len(sub))
        if k < 20:
            continue
        r = int(sub["y"].sum())
        out["par_discipline"][str(disc)] = {
            "n_courses": k, "top1_taux": round(r / k, 4),
            "top1_ic95": [round(x, 4) for x in _intervalle_binomial(r, k)],
        }
    # Dernier passage : plus aucun NaN ni infini ne doit sortir d'ici,
    # sinon la sérialisation JSON de l'API lève une 500.
    return _json_sur(out)


def _ou_tiret(x, gabarit: str = ".5f") -> str:
    """Un nombre formaté, ou un tiret quand il n'y en a pas."""
    if x is None:
        return "—"
    try:
        return format(float(x), gabarit)
    except (TypeError, ValueError):
        return "—"


def afficher_bilan(b: dict) -> str:
    if not b.get("n_courses"):
        L = ["── Bilan de production " + "─" * 37, "  " + b.get("message", "rien à mesurer")]
        a = (b.get("anomalies") or {}).get("courses_sans_gagnant")
        if a:
            L.append(f"  ⚠ {a} courses arrivées sans aucun partant classé 1ᵉʳ — "
                     "défaut de collecte")
        return "\n".join(L)

    L = ["── Bilan de production " + "─" * 37,
         f"  modèle             {b['modele']:>16}",
         f"  courses jugées     {b['n_courses']:>16}",
         f"  favori gagnant     {b['top1_reussites']:>6} / {b['n_courses']:<7}"
         f"  {b['top1_taux']:>6.1%}",
         f"  intervalle à 95 %  [{b['top1_ic95'][0]:>5.1%} ; {b['top1_ic95'][1]:>5.1%}]",
         # `_json_sur` transforme les NaN en None : le format numérique
         # lèverait dessus, et c'est précisément quand une tranche est
         # vide qu'on vient lire ce tableau.
         f"  Brier              {_ou_tiret(b.get('brier')):>16}",
         f"  ECE                {_ou_tiret(b.get('ece')):>16}"]
    m = b.get("marche")
    if m and m.get("top1_taux") is not None:
        L.append(f"  favori du public   {m['top1_reussites']:>6} / {m['n_courses']:<7}"
                 f"  {m['top1_taux']:>6.1%}")
        L.append("  → si les deux intervalles se chevauchent, l'écart n'est pas établi")
    e = b.get("expert")
    if e and e.get("top1_taux") is not None:
        L.append(f"  favori analyste    {e['top1_reussites']:>6} / {e['n_courses']:<7}"
                 f"  {e['top1_taux']:>6.1%}")
    a = (b.get("anomalies") or {}).get("courses_sans_gagnant", 0)
    if a:
        L.append(f"  ⚠ {a} courses écartées : arrivée connue mais aucun partant "
                 "classé 1ᵉʳ (défaut de collecte, pas une contre-performance)")
    if b.get("par_discipline"):
        L.append("\n  " + f"{'discipline':<14} {'courses':>8} {'réussite':>9} {'IC 95 %':>18}")
        for d, s in sorted(b["par_discipline"].items()):
            L.append(f"  {d:<14} {s['n_courses']:>8} {s['top1_taux']:>9.1%} "
                     f"  [{s['top1_ic95'][0]:>5.1%} ; {s['top1_ic95'][1]:>5.1%}]")
    L.append("\n  Rappel : sur 10 courses, un modèle à 25 % de réussite finit")
    L.append("  bredouille une fois sur 18. Dix courses ne prouvent rien.")
    return "\n".join(L)


def afficher(rap: dict) -> str:
    """Rendu texte lisible dans un terminal ou un rapport de mémoire."""
    L = []
    j = rap["justesse"]
    L.append("── Justesse " + "─" * 48)
    L.append(f"  partants évalués   {j['n']:>10}")
    L.append(f"  taux de base       {j['taux_base']:>10.2%}")
    L.append(f"  Brier              {j['brier']:>10.5f}   (plus bas = mieux)")
    L.append(f"  Brier skill        {j['brier_skill']:>10.4f}   (>0 = utile)")
    L.append(f"  AUC                {str(j['auc']):>10}")
    L.append(f"  ECE                {rap['ece']:>10.5f}   (écart moyen de calibration)")

    L.append("\n── Calibration " + "─" * 45)
    L.append(f"  {'annoncé':>10} {'observé':>10} {'écart':>9} {'n':>8}")
    for r in rap["calibration"]:
        L.append(f"  {r['predit']:>10.3f} {r['observe']:>10.3f} "
                 f"{r['ecart']:>+9.3f} {r['n']:>8}")

    m = rap.get("marche") or {}
    if m:
        L.append("\n── Face au marché " + "─" * 42)
        L.append(f"  courses            {m.get('n_courses', 0):>10}")
        L.append(f"  top-1 modèle       {m.get('top1_modele', 0):>10.2%}")
        if "top1_favori" in m:
            L.append(f"  top-1 favori public{m['top1_favori']:>10.2%}")
            L.append(f"  écart              {m['ecart_top1']:>+10.2%}")
            L.append(f"  Brier modèle       {m['brier_modele']:>10.5f}")
            L.append(f"  Brier marché       {m['brier_marche']:>10.5f}")
            L.append(f"  corrélation        {m['correlation_au_marche']:>10.4f}"
                     "   (proche de 1 = le modèle recopie le public)")
            verdict = "OUI" if m.get("modele_bat_marche") else "NON"
            L.append(f"  bat le marché ?    {verdict:>10}")

    if rap.get("rentabilite"):
        L.append("\n── Rentabilité " + "─" * 45)
        L.append(f"  {'seuil':>7} {'paris':>8} {'réussite':>10} {'ROI':>9} {'±1σ':>8}")
        for r in rap["rentabilite"]:
            if not r.get("n_paris"):
                continue
            L.append(f"  {r['seuil_valeur']:>7.2f} {r['n_paris']:>8} "
                     f"{r['taux_reussite']:>10.2%} {r['roi_pct']:>+8.1f}% "
                     f"{r['roi_ecart_type_pct']:>7.1f}%")
        L.append("  ⚠ en dessous de ~500 paris, le ROI n'est pas interprétable")
        L.append("  ⚠ retour ESTIMÉ : cote relevée avant le départ, moins le"
                 " prélèvement supposé")

    if rap.get("rentabilite_reelle"):
        ecartees = next((r.get("courses_ecartees") for r in rap["rentabilite_reelle"]
                         if r.get("courses_ecartees") is not None), None)
        L.append("\n── Rentabilité mesurée (rapports réellement payés) " + "─" * 9)
        L.append(f"  {'seuil':>7} {'paris':>8} {'réussite':>10} {'ROI':>9} {'±1σ':>8}")
        for r in rap["rentabilite_reelle"]:
            if not r.get("n_paris"):
                continue
            L.append(f"  {r['seuil_valeur']:>7.2f} {r['n_paris']:>8} "
                     f"{r['taux_reussite']:>10.2%} {r['roi_pct']:>+8.1f}% "
                     f"{r['roi_ecart_type_pct']:>7.1f}%")
        L.append("  → aucun paramètre : on paie ce qui a été payé.")
        if ecartees:
            L.append(f"  → {ecartees} partants écartés (rapport de la course inconnu),"
                     " et non comptés perdants.")
        L.append("  ⚠ un ROI ne se lit qu'à partir de 2σ. Sous 2σ, c'est du bruit.")

    if rap.get("par_discipline"):
        L.append("\n── Par discipline " + "─" * 42)
        L.append(f"  {'discipline':<14} {'n':>8} {'Brier':>9} {'skill':>8} {'AUC':>7}")
        for disc, s in sorted(rap["par_discipline"].items()):
            L.append(f"  {disc:<14} {s['n']:>8} {s['brier']:>9.5f} "
                     f"{s['brier_skill']:>+8.4f} {str(s['auc']):>7}")
        L.append("  → si les scores divergent fortement, scinder le modèle par discipline")
    return "\n".join(L)

# ---------------------------------------------------------------------
# 4 ter. L'échelle de confiance, de 1 à 5
# ---------------------------------------------------------------------
#
# CE QUE LA NOTE DIT, ET CE QU'ELLE NE DIT PAS
#
# Elle dit : « dans cette course, le modèle détache son favori plus
# nettement que dans X % des courses ». C'est une mesure de TRANCHANT,
# pas une promesse de gain.
#
# Les seuils ne sont pas choisis à la main : ce sont les quintiles de
# `ecart_top2` observés sur la fenêtre de test. Chaque niveau porte donc
# son taux de réussite RÉEL, et celui du favori du public sur les mêmes
# courses. Une note de 5/5 dont le taux est inférieur au marché reste
# une note de 5/5 — et le dire est le seul usage honnête de l'échelle.
#
# `fiable` n'est vrai que si, à ce niveau, le modèle a fait AU MOINS
# aussi bien que le public sur un effectif suffisant. C'est cette
# valeur, et elle seule, qui autorise l'affichage en vert.

MIN_COURSES_NIVEAU = 40


def echelle_confiance(df: pd.DataFrame, *, cible="y_gagnant", n=5) -> dict:
    """
    Découpe `ecart_top2` en cinq niveaux et mesure ce que vaut chacun.

    `df` = fenêtre de test déjà notée (course_id, proba, ecart_top2).
    """
    if "ecart_top2" not in df.columns or df.empty:
        return {"seuils": [], "niveaux": []}

    favoris = df.loc[df.groupby("course_id")["proba"].idxmax()].copy()
    if len(favoris) < n * MIN_COURSES_NIVEAU:
        return {"seuils": [], "niveaux": []}

    bornes = favoris["ecart_top2"].quantile([i / n for i in range(1, n)]).tolist()
    bornes = sorted(round(float(b), 5) for b in bornes)
    favoris["note"] = 1
    for b in bornes:
        favoris["note"] += (favoris["ecart_top2"] > b).astype(int)

    niveaux = []
    for note, sub in favoris.groupby("note"):
        k = len(sub)
        reussites = int(sub[cible].sum())
        ligne = {
            "note": int(note), "n_courses": k,
            "taux": round(reussites / k, 4) if k else None,
            "ic95": [round(x, 4) for x in _intervalle_binomial(reussites, k)],
            "taux_marche": None, "fiable": False,
        }
        if "mkt_proba_implicite" in df.columns:
            memes = df[df["course_id"].isin(sub["course_id"])]
            memes = memes[memes["mkt_proba_implicite"].notna()]
            if len(memes):
                idx = memes.groupby("course_id")["mkt_proba_implicite"].idxmax()
                tm = float(memes.loc[idx, cible].mean())
                ligne["taux_marche"] = round(tm, 4)
                # Le vert ne s'allume que sur une comparaison tenue.
                ligne["fiable"] = bool(k >= MIN_COURSES_NIVEAU
                                       and ligne["taux"] is not None
                                       and ligne["taux"] >= tm)
        niveaux.append(ligne)
    return {"seuils": bornes, "niveaux": niveaux}


# ---------------------------------------------------------------------
# Le prélèvement : la cote relevée est-elle brute ou nette ?
# ---------------------------------------------------------------------
#
# LA QUESTION, ET POURQUOI ELLE DÉCIDE DU PROJET
#
# La simulation de rentabilité multiplie la cote par (1 − prélèvement).
# Ça n'est correct QUE si `mkt_cote` est un rapport BRUT. Or dans un
# pari mutuel, le rapport affiché est en général déjà NET : le
# prélèvement est retiré de la masse avant répartition. Si c'est le cas
# ici, tous les ROI sont sous-estimés d'environ 18 %, et la conclusion
# « le modèle n'est pas rentable » repose sur une erreur d'unité.
#
# Le juge : ce qui a été RÉELLEMENT payé. Pour un cheval gagnant, le
# rapport définitif du SIMPLE GAGNANT est, par définition, la somme
# perçue pour une mise de 1 €. Le rapport de ce montant à `mkt_cote`
# répond à la question sans modèle et sans hypothèse :
#
#     ratio ≈ 1     → la cote relevée EST le rapport payé.
#                     Elle est donc DÉJÀ nette : appliquer (1 − 15 %)
#                     la ponctionne une seconde fois.
#     ratio ≈ 0,85  → la cote est brute, le prélèvement est à appliquer.
#                     La simulation actuelle est juste.
#     ratio ≈ 100   → le rapport est en centimes. Question d'unité, pas
#                     d'économie : à corriger avant toute lecture.
#
# On mesure la MÉDIANE, pas la moyenne : quelques rapports à 300 € sur
# des outsiders écraseraient toute moyenne.

RATIO_TOLERANCE = 0.04

# La cote de référence est le DERNIER relevé du Simple Gagnant avant le
# départ — la même que celle qui alimente `mkt_cote` dans les features.
# La vue v_cote_finale fait exactement ça ; on ne la réécrit pas.
SQL_RAPPORTS = """
    SELECT c.course_id,
           p.num_pmu,
           cf.rapport   AS cote,
           r.rapport    AS rapport,
           r.mise_base  AS mise_base
      FROM course c
      JOIN partant p  ON p.course_id = c.course_id AND p.ordre_arrivee = 1
      JOIN v_cote_finale cf
        ON cf.course_id = p.course_id AND cf.num_pmu = p.num_pmu
       AND cf.type_pari IN ('SIMPLE_GAGNANT', 'E_SIMPLE_GAGNANT')
      JOIN rapport_definitif r
        ON r.course_id = c.course_id
       AND r.combinaison = p.num_pmu::text
       AND r.type_pari IN ('SIMPLE_GAGNANT', 'E_SIMPLE_GAGNANT')
     WHERE c.date_reunion BETWEEN %s AND %s
       AND r.rapport IS NOT NULL
"""


def verifier_rapports(conn, depuis, jusqua, *, prelevement=PRELEVEMENT_DEFAUT) -> dict:
    """
    Compare la cote relevée au rapport réellement payé, sur les gagnants.

    Ne renvoie JAMAIS de conclusion quand l'échantillon est trop mince :
    c'est précisément le genre de question qu'on n'a le droit de
    trancher qu'une fois.
    """
    try:
        rows = conn.execute(SQL_RAPPORTS, (depuis, jusqua)).fetchall()
    except Exception as exc:  # noqa: BLE001
        conn.rollback()
        return {"n": 0, "verdict": "indisponible", "message": str(exc)}

    d = pd.DataFrame(rows, columns=["course_id", "num_pmu", "cote", "rapport", "mise_base"])
    d = d.apply(pd.to_numeric, errors="coerce").dropna(subset=["cote", "rapport"])
    d = d[(d["cote"] > 1) & (d["rapport"] > 0)]
    if len(d) < 30:
        return {"n": int(len(d)), "verdict": "insuffisant",
                "message": ("il faut au moins 30 gagnants avec un rapport "
                            "connu ; lancer d'abord `python -m pmu.collect "
                            "rapports`")}

    # `parse_rapports_definitifs` rend DÉJÀ des euros perçus pour 1 €
    # misé (le champ `dividendePourUnEuro`, converti de centimes). Ne
    # PAS rediviser par la mise de base : elle ne sert plus qu'à
    # documenter la ligne, et diviser une seconde fois par 2 ferait
    # passer une cote nette pour une cote brute.
    d["ratio"] = d["rapport"] / d["cote"]
    med = float(d["ratio"].median())

    if abs(med - 1.0) <= RATIO_TOLERANCE:
        verdict = "cote_nette"
        message = ("La cote relevée EST le rapport payé : elle est déjà nette "
                   "de prélèvement. La simulation le retire une seconde fois, "
                   "donc elle SOUS-ESTIME le ROI.")
        correction = 1.0 / (1.0 - prelevement)
    elif abs(med - (1.0 - prelevement)) <= RATIO_TOLERANCE:
        verdict = "cote_brute"
        message = ("La cote relevée est brute : le prélèvement doit bien être "
                   "appliqué. La simulation de rentabilité est juste.")
        correction = 1.0
    elif med > 20:
        verdict = "unite"
        message = (f"Le rapport vaut {med:.0f} fois la cote : il est presque "
                   "certainement exprimé en centimes. À corriger avant toute "
                   "lecture économique.")
        correction = None
    else:
        verdict = "inattendu"
        message = (f"Ratio médian {med:.3f} — ni 1 (cote nette), ni "
                   f"{1 - prelevement:.2f} (cote brute). Ne rien conclure : "
                   "regarder d'abord quelques lignes à la main.")
        correction = None

    return {
        "n": int(len(d)),
        "courses": int(d["course_id"].nunique()),
        "ratio_median": round(med, 4),
        "ratio_q1": round(float(d["ratio"].quantile(0.25)), 4),
        "ratio_q3": round(float(d["ratio"].quantile(0.75)), 4),
        "cote_mediane": round(float(d["cote"].median()), 2),
        "rapport_median": round(float(d["rapport"].median()), 2),
        "verdict": verdict,
        "message": message,
        # Facteur à appliquer aux ROI déjà publiés pour les corriger.
        "correction_roi": None if correction is None else round(correction, 4),
    }


# ---------------------------------------------------------------------
# LE test décisif : la surcote
# ---------------------------------------------------------------------
#
# Le ratio « rapport payé / cote » mesuré sur les gagnants s'est révélé
# ambigu : 0,894, entre 1,00 (cote nette) et 0,85 (cote brute), avec un
# étalement énorme (Q1 0,68 — Q3 1,16). Explication : la cote relevée
# avant le départ n'EST PAS le rapport payé — l'argent des dernières
# minutes déplace le prix. Et comme la mesure ne porte que sur les
# gagnants, elle est biaisée : un gagnant est justement un cheval sur
# lequel l'argent est venu.
#
# Ce test-ci n'a aucun de ces défauts, parce qu'il ne regarde pas les
# arrivées du tout. Dans un pari mutuel :
#
#     rapport_i = P (1 − t) / S_i        (P = masse, S_i = mises sur i)
#
# donc en sommant sur tous les partants d'une course :
#
#     Σ 1/rapport_i = (Σ S_i) / (P(1 − t)) = 1 / (1 − t)
#
# La somme des probabilités implicites vaut donc :
#
#     1,176  si la cote est DÉJÀ NETTE de prélèvement (t = 15 %)
#     1,000  si la cote est BRUTE
#
# Aucune arrivée, aucun modèle, aucune sélection. Toutes les courses,
# tous les partants.

SQL_SURCOTE = """
    SELECT cf.course_id, SUM(1.0 / cf.rapport) AS surcote, COUNT(*) AS n
      FROM v_cote_finale cf
      JOIN course c  ON c.course_id = cf.course_id
      JOIN partant p ON p.course_id = cf.course_id AND p.num_pmu = cf.num_pmu
     WHERE c.date_reunion BETWEEN %s AND %s
       AND cf.type_pari IN ('SIMPLE_GAGNANT', 'E_SIMPLE_GAGNANT')
       AND cf.rapport > 1
       AND (p.statut IS NULL OR p.statut <> 'NON_PARTANT')
     GROUP BY cf.course_id
    HAVING COUNT(*) >= 5
"""


def surcote(conn, depuis, jusqua, *, prelevement=PRELEVEMENT_DEFAUT) -> dict:
    """Somme des probabilités implicites par course. Tranche le prélèvement."""
    try:
        rows = conn.execute(SQL_SURCOTE, (depuis, jusqua)).fetchall()
    except Exception as exc:  # noqa: BLE001
        conn.rollback()
        return {"n": 0, "verdict": "indisponible", "message": str(exc)}

    d = pd.DataFrame(rows, columns=["course_id", "surcote", "n"])
    d["surcote"] = pd.to_numeric(d["surcote"], errors="coerce")
    d = d[d["surcote"].between(0.5, 3.0)]
    if len(d) < 50:
        return {"n": int(len(d)), "verdict": "insuffisant",
                "message": "moins de 50 courses avec une cote complète"}

    med = float(d["surcote"].median())
    attendu_net = 1.0 / (1.0 - prelevement)

    if abs(med - attendu_net) <= 0.03:
        verdict = "cote_nette"
        message = (f"Somme des probabilités implicites = {med:.3f}, soit "
                   f"1/(1−{prelevement:.0%}). La cote est DÉJÀ NETTE : la "
                   "simulation retire le prélèvement une seconde fois.")
    elif abs(med - 1.0) <= 0.03:
        verdict = "cote_brute"
        message = ("Somme des probabilités implicites = 1,00. La cote est "
                   "BRUTE : appliquer le prélèvement est correct.")
    else:
        # Le prélèvement déduit, quel qu'il soit. C'est la lecture la
        # plus utile : elle ne suppose pas les 15 % de départ.
        t = 1.0 - 1.0 / med if med > 0 else None
        verdict = "cote_nette" if med > 1.05 else "inattendu"
        if verdict == "cote_nette":
            message = (f"Somme des probabilités implicites = {med:.3f} > 1. La "
                       f"cote est DÉJÀ NETTE, d'un prélèvement mesuré à "
                       f"{t:.1%} — et non des {prelevement:.0%} supposés.")
        else:
            message = (f"Somme des probabilités implicites = {med:.3f}. Ni 1,00 "
                       f"ni {attendu_net:.3f} : ne rien conclure avant d'avoir "
                       "regardé quelques courses à la main.")

    t_mesure = (1.0 - 1.0 / med) if med > 1.0 else None
    return {
        "n": int(len(d)),
        "surcote_mediane": round(med, 4),
        "surcote_q1": round(float(d["surcote"].quantile(0.25)), 4),
        "surcote_q3": round(float(d["surcote"].quantile(0.75)), 4),
        "partants_median": int(d["n"].median()),
        "prelevement_mesure": None if t_mesure is None else round(t_mesure, 4),
        "verdict": verdict,
        "message": message,
    }


def afficher_surcote(v: dict) -> str:
    L = ["── Somme des probabilités implicites " + "─" * 23]
    if v.get("verdict") in (None, "indisponible", "insuffisant"):
        L.append(f"  {v.get('message', 'rien à mesurer')}")
        return "\n".join(L)
    L += [
        f"  courses mesurées         {v['n']}",
        f"  partants (médiane)       {v['partants_median']}",
        f"  somme 1/cote             {v['surcote_mediane']}"
        f"   (Q1 {v['surcote_q1']} — Q3 {v['surcote_q3']})",
    ]
    if v.get("prelevement_mesure") is not None:
        L.append(f"  prélèvement déduit       {v['prelevement_mesure']:.1%}")
    L += ["", "  " + v["message"]]
    return "\n".join(L)


def afficher_rapports(v: dict) -> str:
    L = ["── Cote relevée contre rapport payé " + "─" * 24]
    if v.get("verdict") in (None, "indisponible", "insuffisant"):
        L.append(f"  {v.get('message', 'rien à mesurer')}")
        L.append(f"  gagnants exploitables : {v.get('n', 0)}")
        return "\n".join(L)
    L += [
        f"  gagnants comparés        {v['n']} sur {v['courses']} courses",
        f"  cote médiane             {v['cote_mediane']}",
        f"  rapport payé médian      {v['rapport_median']}",
        f"  ratio rapport / cote     {v['ratio_median']}"
        f"   (Q1 {v['ratio_q1']} — Q3 {v['ratio_q3']})",
        "",
        "  " + v["message"],
    ]
    if v.get("correction_roi") and v["correction_roi"] != 1.0:
        L.append(f"  → multiplier (1 + ROI) par {v['correction_roi']} pour corriger")
    return "\n".join(L)


def note_confiance(ecart: float | None, seuils: list) -> int:
    """Écart 1ᵉʳ/2ᵉ → note de 1 à 5, selon les seuils mesurés."""
    if ecart is None or not seuils:
        return 1
    try:
        e = float(ecart)
    except (TypeError, ValueError):
        return 1
    return 1 + sum(1 for b in seuils if e > b)


def afficher_echelle(ech: dict) -> str:
    niveaux = ech.get("niveaux") or []
    if not niveaux:
        return ("── Échelle de confiance " + "─" * 35
                + "\n  pas assez de courses pour établir les niveaux")
    L = ["── Échelle de confiance " + "─" * 35,
         f"  {'note':>5} {'courses':>8} {'modèle':>9} {'marché':>9}  verdict"]
    for x in sorted(niveaux, key=lambda v: v["note"]):
        tm = x.get("taux_marche")
        L.append(f"  {'★' * x['note']:>5} {x['n_courses']:>8} {x['taux']:>9.1%} "
                 f"{('—' if tm is None else f'{tm:.1%}'):>9}  "
                 + ("au niveau du marché" if x["fiable"] else "en dessous du marché"))
    L.append("  → la note mesure le TRANCHANT du modèle, pas une promesse.")
    L.append("    Seul un niveau « au niveau du marché » est mis en avant.")
    return "\n".join(L)

