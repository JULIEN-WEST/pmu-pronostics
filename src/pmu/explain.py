"""
Pourquoi ce cheval-là — attribution locale par ablation de familles.

===================================================================
LE PROBLÈME

Un pronostic sans justification ne se vérifie pas. « 21,8 % » n'apprend
rien : ni pourquoi, ni sur quoi se fier, ni quand se méfier. Ce qui a
été demandé, ce sont les données QUI JUSTIFIENT le classement —
historique, forme, lignée, chrono.

===================================================================
LA MÉTHODE, ET SES LIMITES

On procède par ABLATION intra-course. Pour une famille de features
(disons la vitesse), on remplace ces colonnes par la MÉDIANE DE LA
COURSE : tous les partants deviennent identiques sur cette famille,
qui cesse donc de les départager. On redemande au modèle de noter la
course, et l'écart avec la note d'origine mesure ce que cette famille
apportait à CE cheval, DANS CE lot.

Trois raisons de préférer cela à SHAP :

  1. Aucune dépendance lourde, et ça marche avec n'importe quel modèle,
     y compris le repli scikit-learn.
  2. La référence est le lot du jour, pas une moyenne d'entraînement.
     « Meilleur chrono DE CETTE COURSE » est ce qui intéresse un
     parieur ; « meilleur chrono en valeur absolue » ne veut rien dire
     entre un trot de 2700 m et un plat de 1200 m.
  3. La renormalisation par course est conservée, donc les écarts
     s'additionnent dans le même espace que les probabilités affichées.

CE QUE ÇA N'EST PAS. Ce n'est pas une décomposition exacte : les
familles interagissent, et la somme des écarts ne fait pas la
probabilité. Ce n'est pas non plus une explication CAUSALE — le modèle
constate des associations, il n'établit pas que le chrono cause la
victoire. Les motifs se lisent comme « voici ce qui, dans ce lot, a
pesé sur la note », jamais comme « voici pourquoi il va gagner ».
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger("pmu.explain")


# ---------------------------------------------------------------------
# Les familles lisibles
# ---------------------------------------------------------------------
# L'ordre compte : il sert de départage quand deux familles pèsent
# autant. On met en tête ce qu'un parieur regarde en premier.

GROUPES: dict[str, dict] = {
    "forme": {
        "titre": "Forme récente",
        "icone": "mdi:chart-line",
        "prefixes": ("mus_",),
        "colonnes": {"p_jours_repos", "r_musique"},
    },
    "vitesse": {
        "titre": "Chrono",
        "icone": "mdi:speedometer",
        "prefixes": ("v_reduction",),
        "colonnes": {"r_vitesse", "r_vitesse_best"},
    },
    "historique": {
        "titre": "Historique du cheval",
        "icone": "mdi:history",
        "prefixes": ("h_cheval",),
        "colonnes": set(),
    },
    "aptitude": {
        "titre": "Aptitude aux conditions",
        "icone": "mdi:target",
        "prefixes": ("a_",),
        "colonnes": set(),
    },
    "lignee": {
        "titre": "Lignée",
        "icone": "mdi:family-tree",
        "prefixes": ("g_",),
        "colonnes": set(),
    },
    "entourage": {
        "titre": "Driver et entourage",
        "icone": "mdi:account-tie",
        "prefixes": ("h_driver", "h_entr", "h_couple", "h_attelage", "h_proprio"),
        "colonnes": set(),
    },
    "palmares": {
        "titre": "Palmarès et gains",
        "icone": "mdi:trophy-variant",
        "prefixes": ("p_taux", "p_gains", "p_nb"),
        "colonnes": {"r_gains", "r_taux_vict", "r_gains_tot"},
    },
    "marge": {
        "titre": "Qualité des arrivées",
        "icone": "mdi:ruler",
        "prefixes": ("v_marge",),
        "colonnes": {"r_marge"},
    },
    "conditions": {
        "titre": "Conditions du jour",
        "icone": "mdi:map-marker-distance",
        "prefixes": ("c_",),
        "colonnes": {"p_age", "p_poids", "p_sexe", "p_deferre", "p_oeilleres"},
    },
    "marche": {
        "titre": "Marché",
        "icone": "mdi:cash-multiple",
        "prefixes": ("mkt_",),
        "colonnes": set(),
    },
}


def colonnes_du_groupe(groupe: str, colonnes: list[str]) -> list[str]:
    g = GROUPES[groupe]
    return [c for c in colonnes
            if c.startswith(g["prefixes"]) or c in g["colonnes"]]


# ---------------------------------------------------------------------
# Mise en forme des chiffres
# ---------------------------------------------------------------------

def reduction_lisible(ms_par_km) -> str | None:
    """
    Réduction kilométrique : millisecondes par km → « 1'12"8 ».
    C'est l'unité que lit un turfiste ; les millisecondes ne parlent
    à personne.
    """
    if ms_par_km is None or (isinstance(ms_par_km, float) and np.isnan(ms_par_km)):
        return None
    total = float(ms_par_km) / 1000.0
    if total <= 0 or total > 600:
        return None
    minutes = int(total // 60)
    secondes = total - minutes * 60
    entier = int(secondes)
    dixiemes = int(round((secondes - entier) * 10))
    if dixiemes == 10:                       # 12,96 s → 13"0, pas 12"10
        entier, dixiemes = entier + 1, 0
    if entier == 60:                         # 1'59,97 → 2'00"0
        minutes, entier = minutes + 1, 0
    return f"{minutes}'{entier:02d}\"{dixiemes}"


def _pct(x, decimales: int = 0) -> str | None:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    return f"{float(x) * 100:.{decimales}f} %"


def _nombre(x, decimales: int = 0) -> str | None:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    return f"{float(x):,.{decimales}f}".replace(",", " ")


def _v(ligne, col, defaut=np.nan):
    return ligne[col] if col in ligne.index else defaut


# ---------------------------------------------------------------------
# Les faits, en français
# ---------------------------------------------------------------------

def faits(ligne: pd.Series) -> dict:
    """
    Les chiffres bruts derrière chaque famille, formulés lisiblement.
    Une famille absente rend une liste vide — jamais une phrase vide,
    jamais « None ».
    """
    out: dict[str, list[str]] = {g: [] for g in GROUPES}

    # -- Forme -------------------------------------------------------
    mus_n, mus_moy = _v(ligne, "mus_n"), _v(ligne, "mus_moy")
    if mus_n and mus_n > 0 and not pd.isna(mus_moy):
        out["forme"].append(f"place moyenne {mus_moy:.1f} sur ses {int(mus_n)} dernières sorties")
    top3 = _v(ligne, "mus_top3")
    if not pd.isna(top3) and mus_n and mus_n > 0:
        out["forme"].append(f"top 3 dans {_pct(top3)} des cas")
    inc = _v(ligne, "mus_incidents")
    if not pd.isna(inc) and inc > 0:
        out["forme"].append(f"{_pct(inc)} d'incidents (disqualifications, chutes)")
    repos = _v(ligne, "p_jours_repos")
    if not pd.isna(repos):
        j = int(round(repos))
        out["forme"].append(
            f"{j} jours depuis sa dernière course"
            + (" — sortie très fraîche" if j <= 7 else
               (" — longue coupure" if j >= 90 else ""))
        )

    # -- Chrono ------------------------------------------------------
    best = reduction_lisible(_v(ligne, "v_reduction_best"))
    moy = reduction_lisible(_v(ligne, "v_reduction_moy"))
    n_chrono = _v(ligne, "v_reduction_n")
    if best:
        out["vitesse"].append(f"meilleur chrono connu {best} au km")
    if moy:
        suffixe = f" sur {int(n_chrono)} courses chronométrées" if n_chrono and n_chrono > 0 else ""
        out["vitesse"].append(f"moyenne {moy} au km{suffixe}")
    progres = _v(ligne, "v_reduction_progres")
    if not pd.isna(progres) and abs(progres) > 300:
        sens = "plus rapide" if progres < 0 else "plus lent"
        out["vitesse"].append(f"dernier chrono {abs(progres) / 1000:.1f} s/km {sens} que sa moyenne")
    rang_v = _v(ligne, "r_vitesse")
    if not pd.isna(rang_v):
        out["vitesse"].append(f"vitesse : {_pct(1 - rang_v)} du lot fait moins bien")

    # -- Historique --------------------------------------------------
    n_hist = _v(ligne, "h_cheval_place_n")
    if n_hist and n_hist > 0:
        out["historique"].append(
            f"{int(n_hist)} courses connues en base"
            + (" — historique mince" if n_hist < 5 else "")
        )
        tp = _v(ligne, "h_cheval_place")
        tg = _v(ligne, "h_cheval_gagne")
        if not pd.isna(tp):
            out["historique"].append(f"placé {_pct(tp)} du temps (taux lissé)")
        if not pd.isna(tg):
            out["historique"].append(f"gagne {_pct(tg)} du temps")
    else:
        out["historique"].append("aucune course antérieure en base — le modèle avance à l'aveugle")

    # -- Aptitude ----------------------------------------------------
    for col, mot in [("a_terrain", "sur ce terrain"),
                     ("a_distance", "sur cette distance"),
                     ("a_hippodrome", "sur cet hippodrome"),
                     ("a_discipline", "dans cette discipline")]:
        delta, n = _v(ligne, f"{col}_delta"), _v(ligne, f"{col}_n")
        if pd.isna(delta) or not n or n < 2:
            continue
        pts = delta * 100
        if abs(pts) < 1.5:
            continue
        signe = "meilleur" if pts > 0 else "moins bon"
        out["aptitude"].append(
            f"{mot} : {abs(pts):.0f} points {signe} que sa moyenne ({int(n)} courses)")

    # -- Lignée ------------------------------------------------------
    pere = _v(ligne, "nom_pere", None)
    g_pere, g_pere_n = _v(ligne, "g_pere"), _v(ligne, "g_pere_n")
    if pere and not pd.isna(g_pere) and g_pere_n and g_pere_n >= 10:
        out["lignee"].append(
            f"produits de {pere} placés {_pct(g_pere)} du temps ({int(g_pere_n)} courses)")
    dt = _v(ligne, "g_pere_terrain_delta")
    if not pd.isna(dt) and abs(dt) > 0.02 and pere:
        sens = "réussit mieux" if dt > 0 else "réussit moins bien"
        out["lignee"].append(f"la lignée {pere} {sens} sur ce type de terrain")
    dd = _v(ligne, "g_pere_distance_delta")
    if not pd.isna(dd) and abs(dd) > 0.02 and pere:
        sens = "convient" if dd > 0 else "ne convient pas"
        out["lignee"].append(f"la distance {sens} à la lignée {pere}")
    croise, croise_n = _v(ligne, "g_croisement"), _v(ligne, "g_croisement_n")
    mere = _v(ligne, "nom_pere_mere", None)
    if croise_n and croise_n >= 10 and not pd.isna(croise) and pere and mere:
        out["lignee"].append(
            f"croisement {pere} × {mere} : {_pct(croise)} de places ({int(croise_n)} courses)")

    # -- Entourage ---------------------------------------------------
    drv = _v(ligne, "driver", None)
    tdp, tdn = _v(ligne, "h_driver_place"), _v(ligne, "h_driver_place_n")
    if not pd.isna(tdp) and tdn and tdn >= 5:
        out["entourage"].append(
            (f"{drv} : " if drv else "driver : ") +
            f"{_pct(tdp)} de places sur {int(tdn)} courses")
    tc, tcn = _v(ligne, "h_couple_gagne"), _v(ligne, "h_couple_gagne_n")
    if not pd.isna(tc) and tcn and tcn >= 3:
        out["entourage"].append(
            f"association cheval/driver déjà vue {int(tcn)} fois, {_pct(tc)} de victoires")
    ent = _v(ligne, "entraineur", None)
    te, ten = _v(ligne, "h_entr_gagne"), _v(ligne, "h_entr_gagne_n")
    if not pd.isna(te) and ten and ten >= 10:
        out["entourage"].append(
            (f"{ent} : " if ent else "entraîneur : ") + f"{_pct(te)} de victoires")
    tpr, tprn = _v(ligne, "h_proprio_place"), _v(ligne, "h_proprio_place_n")
    if not pd.isna(tpr) and tprn and tprn >= 10:
        out["entourage"].append(f"écurie placée {_pct(tpr)} du temps ({int(tprn)} courses)")

    # -- Palmarès ----------------------------------------------------
    nc, nv = _v(ligne, "nombre_courses"), _v(ligne, "nombre_victoires")
    if not pd.isna(nc) and nc:
        v = 0 if pd.isna(nv) else int(nv)
        out["palmares"].append(f"{v} victoires en {int(nc)} courses (palmarès déclaré)")
    gains = _v(ligne, "gains_carriere")
    if not pd.isna(gains) and gains:
        out["palmares"].append(f"{_nombre(gains)} € de gains en carrière")
    rg = _v(ligne, "r_gains")
    if not pd.isna(rg):
        out["palmares"].append(f"gains par course : {_pct(rg)} du lot fait moins bien")

    # -- Marge -------------------------------------------------------
    mm = _v(ligne, "v_marge_moy")
    if not pd.isna(mm):
        out["marge"].append(
            f"battu en moyenne de {mm:.1f} longueur(s) par le cheval qui le précède"
            + (" — arrivées très serrées" if mm < 1.0 else ""))

    # -- Conditions --------------------------------------------------
    corde, nbp = _v(ligne, "c_corde"), _v(ligne, "c_nb_partants")
    if not pd.isna(corde) and not pd.isna(nbp):
        out["conditions"].append(f"corde {int(corde)} sur {int(nbp)}")
    recul = _v(ligne, "c_recul_relatif_lot")
    if not pd.isna(recul) and recul > 0:
        out["conditions"].append(f"reculé de {int(recul)} m par rapport au premier poteau")
    poids = _v(ligne, "p_poids")
    if not pd.isna(poids) and poids:
        out["conditions"].append(f"{poids:.1f} kg")
    age = _v(ligne, "p_age")
    if not pd.isna(age) and age:
        out["conditions"].append(f"{int(age)} ans")
    def_ = _v(ligne, "deferre", None)
    if def_ and str(def_) != "nan":
        out["conditions"].append(f"déferré : {str(def_).lower().replace('_', ' ')}")

    # -- Marché ------------------------------------------------------
    cote = _v(ligne, "mkt_cote")
    if not pd.isna(cote):
        out["marche"].append(f"cote {cote:.1f}")
        rang_c = _v(ligne, "mkt_rang_cote")
        if not pd.isna(rang_c):
            r = int(rang_c)
            out["marche"].append(f"{r}{'ᵉʳ' if r == 1 else 'ᵉ'} choix du public")
    derive = _v(ligne, "mkt_derive")
    if not pd.isna(derive) and abs(derive) > 0.05:
        out["marche"].append(
            "la cote baisse depuis l'ouverture — de l'argent rentre"
            if derive < 0 else "la cote monte depuis l'ouverture — le public s'en détourne")

    return {g: v for g, v in out.items() if v}


# ---------------------------------------------------------------------
# L'ablation
# ---------------------------------------------------------------------

def contributions(modele, df: pd.DataFrame, *, groupes=None) -> pd.DataFrame:
    """
    Écart de probabilité imputable à chaque famille, par partant.

    `df` doit contenir des courses ENTIÈRES : l'ablation remplace chaque
    colonne par la médiane de sa course, ce qui n'a de sens que si tous
    les partants sont là. Passer un cheval isolé donnerait un écart nul
    partout — la médiane d'un seul élément étant lui-même.

    Renvoie un cadre indexé comme `df`, une colonne par famille.
    Positif = la famille pousse ce cheval vers le HAUT du classement.
    """
    colonnes = getattr(modele, "colonnes", None)
    if colonnes is None:                       # ModeleParDiscipline
        colonnes = getattr(getattr(modele, "global_", None), "colonnes", []) or []
    groupes = groupes or list(GROUPES)

    base = modele.predire(df)["proba"].reindex(df.index)
    out = pd.DataFrame(index=df.index)

    for groupe in groupes:
        cols = [c for c in colonnes_du_groupe(groupe, list(colonnes)) if c in df.columns]
        cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
        if not cols:
            continue
        muet = df.copy()
        # Médiane PAR COURSE : la famille cesse de départager les
        # partants sans sortir de la distribution du jour.
        medianes = muet.groupby("course_id", sort=False)[cols].transform("median")
        muet[cols] = medianes
        try:
            ablate = modele.predire(muet)["proba"].reindex(df.index)
        except Exception as exc:               # pragma: no cover
            log.warning("ablation impossible pour %s : %s", groupe, exc)
            continue
        out[groupe] = base - ablate

    return out


def expliquer(modele, df: pd.DataFrame, *, n_motifs: int = 3,
              seuil: float = 0.004) -> dict:
    """
    Assemble tout : contributions + faits → motifs lisibles par partant.

    Renvoie {(course_id, num_pmu): {"motifs": [...], "faits": {...}}}.

    `seuil` écarte le bruit : sous 0,4 point de probabilité, une famille
    n'a rien dit qui mérite d'être écrit.
    """
    if df.empty:
        return {}
    contrib = contributions(modele, df)
    resultat: dict = {}

    for idx, ligne in df.iterrows():
        motifs = []
        if len(contrib.columns):
            valeurs = contrib.loc[idx].dropna()
            for groupe, poids in valeurs.reindex(
                    valeurs.abs().sort_values(ascending=False).index).items():
                if abs(poids) < seuil or len(motifs) >= n_motifs:
                    continue
                motifs.append({
                    "groupe": groupe,
                    "titre": GROUPES[groupe]["titre"],
                    "icone": GROUPES[groupe]["icone"],
                    "sens": "+" if poids > 0 else "−",
                    "poids": round(float(poids), 4),
                })
        f = faits(ligne)
        for m in motifs:
            m["details"] = f.get(m["groupe"], [])
        resultat[(int(ligne["course_id"]), int(ligne["num_pmu"]))] = {
            "motifs": motifs,
            "faits": f,
        }
    return resultat
