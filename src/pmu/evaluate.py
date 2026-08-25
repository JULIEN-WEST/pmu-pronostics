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
               mise: float = 1.0) -> pd.DataFrame:
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

    lignes = []
    for seuil in seuils_valeur:
        paris = d[d["valeur"] >= seuil]
        n = len(paris)
        if n == 0:
            lignes.append({"seuil_valeur": seuil, "n_paris": 0})
            continue
        gagnes = paris[cible].sum()
        # Rapport brut × (1 − prélèvement) : la cote affichée intègre déjà
        # le prélèvement dans un système mutuel, mais on l'applique
        # explicitement pour rendre le paramètre visible et discutable.
        retour = (paris.loc[paris[cible] == 1, col_cote] * mise * (1 - prelevement)).sum()
        engage = n * mise
        lignes.append({
            "seuil_valeur": seuil,
            "n_paris": n,
            "n_gagnants": int(gagnes),
            "taux_reussite": round(float(gagnes / n), 4),
            "cote_moyenne": round(float(paris[col_cote].mean()), 2),
            "engage": round(engage, 2),
            "retour": round(float(retour), 2),
            "roi_pct": round(float((retour - engage) / engage * 100), 2),
            # Écart-type du ROI : sans lui, le ROI n'est pas interprétable.
            "roi_ecart_type_pct": round(
                float(paris[cible].mul(paris[col_cote]).mul(1 - prelevement)
                      .sub(1).std() / np.sqrt(n) * 100), 2),
        })
    return pd.DataFrame(lignes)


# ---------------------------------------------------------------------
# Rapport complet
# ---------------------------------------------------------------------

def rapport(df: pd.DataFrame, cible="y_gagnant", prelevement=PRELEVEMENT_DEFAUT) -> dict:
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
    out["rentabilite"] = sim.to_dict("records") if len(sim) else []

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
       p.ordre_arrivee,
       (SELECT count(*) FROM partant px
         WHERE px.course_id = c.course_id AND px.ordre_arrivee = 1) AS a_un_gagnant
  FROM pronostic pr
  JOIN course  c ON c.course_id = pr.course_id
  JOIN partant p ON p.course_id = pr.course_id AND p.num_pmu = pr.num_pmu
 WHERE pr.modele = %(modele)s
   AND c.date_reunion BETWEEN %(depuis)s AND %(jusqua)s
   AND c.ordre_arrivee IS NOT NULL
"""


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
    return out


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
         f"  Brier              {b['brier']:>16.5f}",
         f"  ECE                {b['ece']:>16.5f}"]
    m = b.get("marche")
    if m and m.get("top1_taux") is not None:
        L.append(f"  favori du public   {m['top1_reussites']:>6} / {m['n_courses']:<7}"
                 f"  {m['top1_taux']:>6.1%}")
        L.append("  → si les deux intervalles se chevauchent, l'écart n'est pas établi")
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

    if rap.get("par_discipline"):
        L.append("\n── Par discipline " + "─" * 42)
        L.append(f"  {'discipline':<14} {'n':>8} {'Brier':>9} {'skill':>8} {'AUC':>7}")
        for disc, s in sorted(rap["par_discipline"].items()):
            L.append(f"  {disc:<14} {s['n']:>8} {s['brier']:>9.5f} "
                     f"{s['brier_skill']:>+8.4f} {str(s['auc']):>7}")
        L.append("  → si les scores divergent fortement, scinder le modèle par discipline")
    return "\n".join(L)
