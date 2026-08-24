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
