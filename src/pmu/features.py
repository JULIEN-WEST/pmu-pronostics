"""
Feature store — construction anti-fuite.

===================================================================
LA RÈGLE, une seule, et tout le fichier en découle :

    aucune feature d'une ligne ne peut dépendre du résultat de cette
    ligne, ni d'aucune course partie après elle.

===================================================================

Elle paraît évidente. Elle est violée dans à peu près tous les projets de
pronostic amateur, et toujours de la même façon : on calcule « le taux de
réussite du cheval » sur l'ensemble des données, cette moyenne contient la
course qu'on cherche à prédire, le modèle affiche 85 % de réussite en
validation et 28 % en réel.

L'implémentation repose sur un seul motif, appliqué partout :

    trier par heure_depart, faire un cumul par groupe, puis DÉCALER d'un
    cran (`shift(1)`) à l'intérieur du groupe.

Après le décalage, la ligne courante voit la somme de tout ce qui la
précède, et rien d'autre. `_taux_glissant()` est la seule fonction qui
fait ça, et toutes les features conditionnelles passent par elle.

Deux garde-fous complètent le dispositif :
  - `tests/test_fuite.py` génère une cible purement aléatoire et vérifie
    que le modèle ne dépasse pas l'aléatoire. S'il y arrive, il y a fuite.
  - Les colonnes de marché sont préfixées `mkt_` et isolables d'un
    argument, parce qu'elles écrasent tout le reste (cf. `COLONNES_MARCHE`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .normalize import parse_musique

# Colonnes issues du marché : légitimes, mais à manier séparément.
# Un modèle qui les voit apprend surtout à recopier la cote — il sera très
# bon en apparence et sans aucune valeur, puisque la cote est déjà connue
# de tous. On entraîne donc DEUX modèles (cf. train.py) :
#   - sans marché  → cherche un écart exploitable
#   - avec marché  → borne haute de ce qui est prévisible
COLONNES_MARCHE = ["mkt_cote", "mkt_proba_implicite", "mkt_rang_cote", "mkt_derive"]

# Lissage bayésien : un cheval 1 victoire / 1 course n'a pas 100 % de
# réussite. On tire le taux vers la moyenne globale avec un poids
# équivalent à PSEUDO_N observations fictives.
PSEUDO_N = 10.0


# ---------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------

def _taux_glissant(
    df: pd.DataFrame, cles: list[str], cible: str, *,
    prior: float, pseudo_n: float = PSEUDO_N, min_n: int = 0,
) -> tuple[pd.Series, pd.Series]:
    """
    Taux de réussite lissé, calculé UNIQUEMENT sur les COURSES ANTÉRIEURES.

    Renvoie (taux, effectif).

    Le point délicat n'est pas d'exclure la ligne courante — ça, tout le
    monde y pense. C'est d'exclure **toute la course courante**.

    Exemple concret : deux demi-frères par le même père courent l'un contre
    l'autre. Si on se contente d'un `shift(1)`, le second voit le résultat
    du premier — alors qu'ils sont partis en même temps. Au moment de
    prédire, cette information n'existe pas. C'est une fuite, discrète mais
    bien réelle, et elle touche toutes les features de lignée, d'entraîneur
    et d'écurie.

    D'où le calcul :

        cumul sur la clé jusqu'ici
      − cumul sur (clé, course) jusqu'ici
      = cumul sur les seules courses strictement antérieures

    Le tri par (heure_depart, course_id) doit être fait EN AMONT : il rend
    les lignes d'une même course contiguës, ce dont dépend ce calcul.
    """
    # Un non-partant, ou une course sans arrivée exploitable, ne doit
    # peser ni au numérateur ni au dénominateur.
    poids = df["est_exploitable"].astype(float)
    succes_ligne = df[cible].fillna(0.0) * poids

    tmp = pd.DataFrame({"_s": succes_ligne, "_w": poids, "_c": df["course_id"]})
    for i, cle in enumerate(cles):
        tmp[f"_k{i}"] = df[cle]
    k_cols = [f"_k{i}" for i in range(len(cles))]

    par_cle = tmp.groupby(k_cols, sort=False, dropna=False)
    par_cle_course = tmp.groupby(k_cols + ["_c"], sort=False, dropna=False)

    succes = par_cle["_s"].cumsum() - par_cle_course["_s"].cumsum()
    effectif = par_cle["_w"].cumsum() - par_cle_course["_w"].cumsum()

    taux = (succes + prior * pseudo_n) / (effectif + pseudo_n)
    if min_n:
        taux = taux.where(effectif >= min_n, np.nan)
    return taux, effectif.astype(float)


def _bande_distance(d: pd.Series) -> pd.Series:
    """Regroupe par tranche de 200 m : 1608 et 1600 sont la même épreuve."""
    return (d / 200).round() * 200


_TERRAIN = {
    "BON": "BON", "BON SOUPLE": "SOUPLE", "SOUPLE": "SOUPLE",
    "TRES SOUPLE": "LOURD", "COLLANT": "LOURD", "LOURD": "LOURD",
    "TRES LOURD": "LOURD", "SEC": "SEC", "BON LEGER": "BON",
    "PSF": "PSF", "PISTE EN SABLE FIBRE": "PSF",
}


def _classe_terrain(s: pd.Series) -> pd.Series:
    up = s.fillna("INCONNU").astype(str).str.upper().str.strip()
    return up.map(_TERRAIN).fillna("AUTRE")


def score_musique(musique: str | None, n: int = 6) -> dict:
    """
    Résume la musique déclarée AVANT la course : elle est donc connue au
    moment de la prédiction, aucune fuite possible.
    """
    perfs = parse_musique(musique)[:n]
    if not perfs:
        return {"mus_n": 0, "mus_moy": np.nan, "mus_top3": np.nan,
                "mus_incidents": np.nan, "mus_derniere": np.nan}
    places = [p["place"] for p in perfs if p["place"] is not None]
    incidents = sum(1 for p in perfs if p["incident"] in
                    ("DISQUALIFIE", "TOMBE", "ARRETE", "RETROGRADE"))
    return {
        "mus_n": len(perfs),
        # Une non-place vaut 11 : pénalisant sans être aberrant.
        "mus_moy": float(np.mean([p["place"] if p["place"] else 11 for p in perfs])),
        "mus_top3": float(np.mean([1.0 if (p["place"] or 99) <= 3 else 0.0 for p in perfs])),
        "mus_incidents": incidents / len(perfs),
        "mus_derniere": float(places[0]) if places else np.nan,
    }


# ---------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------

def construire(df: pd.DataFrame, *, avec_marche: bool = True) -> pd.DataFrame:
    """
    Prend le dataset plat (une ligne = un partant) et renvoie le même
    dataset enrichi des features.

    Colonnes attendues en entrée :
        course_id, heure_depart, discipline, distance, etat_terrain,
        hippodrome_code, nombre_partants, num_pmu, id_cheval, id_driver,
        id_entraineur, nom_pere, nom_pere_mere, age, sexe, place_corde,
        handicap_poids, deferre, oeilleres, musique, nombre_courses,
        nombre_victoires, nombre_places, gains_carriere,
        gains_annee_en_cours, ordre_arrivee, statut
    Optionnelles : cote_finale, cote_ouverture
    """
    df = df.copy()

    # --- Ordre chronologique strict. Tout le fichier en dépend. ---
    df["heure_depart"] = pd.to_datetime(df["heure_depart"], utc=True, errors="coerce")
    df = df.sort_values(["heure_depart", "course_id", "num_pmu"], kind="mergesort")
    df = df.reset_index(drop=True)

    # --- Cibles ---
    place = pd.to_numeric(df["ordre_arrivee"], errors="coerce")
    df["y_gagnant"] = (place == 1).astype(float)
    # Règle PMU : le pari placé paie 3 rangs à partir de 8 partants, 2 en dessous.
    n_part = pd.to_numeric(df["nombre_partants"], errors="coerce")
    seuil_place = np.where(n_part >= 8, 3, 2)
    df["y_place"] = ((place > 0) & (place <= seuil_place)).astype(float)
    # Un non-partant n'a pas de résultat : il sera écarté à l'entraînement.
    df["est_exploitable"] = place.notna() & df["statut"].ne("NON_PARTANT")

    # --- Contexte de course ---
    df["c_distance"] = pd.to_numeric(df["distance"], errors="coerce")
    df["c_bande_distance"] = _bande_distance(df["c_distance"])
    df["c_terrain"] = _classe_terrain(df.get("etat_terrain", pd.Series(index=df.index, dtype=object)))
    df["c_nb_partants"] = n_part
    df["c_allocation"] = pd.to_numeric(df.get("montant_prix"), errors="coerce")
    df["c_corde"] = pd.to_numeric(df["place_corde"], errors="coerce")
    # La corde ne veut rien dire dans l'absolu : 3 sur 8 ≠ 3 sur 18.
    df["c_corde_rel"] = df["c_corde"] / df["c_nb_partants"]

    # --- Identité du partant ---
    df["p_age"] = pd.to_numeric(df["age"], errors="coerce")
    df["p_sexe"] = df["sexe"].astype("category")
    df["p_poids"] = pd.to_numeric(df.get("handicap_poids"), errors="coerce")
    df["p_deferre"] = df.get("deferre", pd.Series(index=df.index, dtype=object)).astype("category")
    df["p_oeilleres"] = df.get("oeilleres", pd.Series(index=df.index, dtype=object)).astype("category")

    # --- Palmarès déclaré (connu avant le départ) ---
    nc = pd.to_numeric(df["nombre_courses"], errors="coerce")
    nv = pd.to_numeric(df["nombre_victoires"], errors="coerce")
    npl = pd.to_numeric(df["nombre_places"], errors="coerce")
    gains = pd.to_numeric(df["gains_carriere"], errors="coerce")
    df["p_nb_courses"] = nc
    df["p_taux_victoire"] = nv / nc.replace(0, np.nan)
    df["p_taux_place"] = npl / nc.replace(0, np.nan)
    df["p_gains_par_course"] = gains / nc.replace(0, np.nan)
    df["p_gains_log"] = np.log1p(gains.clip(lower=0))
    df["p_gains_annee_log"] = np.log1p(
        pd.to_numeric(df.get("gains_annee_en_cours"), errors="coerce").clip(lower=0)
    )

    # --- Musique ---
    mus = pd.DataFrame([score_musique(m) for m in df["musique"]], index=df.index)
    df = pd.concat([df, mus], axis=1)

    # --- Fraîcheur : jours depuis la dernière sortie ---
    derniere = df.groupby("id_cheval", sort=False)["heure_depart"].shift(1)
    df["p_jours_repos"] = (df["heure_depart"] - derniere).dt.total_seconds() / 86400.0

    # --- Position dans le lot : le signal le plus sous-estimé ---
    # Ce qui compte n'est pas le niveau absolu du cheval mais son niveau
    # RELATIF aux autres partants de SA course. Un rang intra-course est
    # calculé sur des colonnes déjà anti-fuite, il n'introduit rien.
    par_course = df.groupby("course_id", sort=False)
    for col, nom in [("p_gains_par_course", "r_gains"),
                     ("p_taux_victoire", "r_taux_vict"),
                     ("mus_moy", "r_musique"),
                     ("p_gains_log", "r_gains_tot")]:
        df[nom] = par_course[col].rank(pct=True, ascending=(nom == "r_musique"))

    # --- Taux glissants : cheval, driver, entraîneur ---
    prior_g = float(df.loc[df["est_exploitable"], "y_gagnant"].mean() or 0.1)
    prior_p = float(df.loc[df["est_exploitable"], "y_place"].mean() or 0.3)

    specs = [
        # (nom, clés, cible, prior, pseudo_n, min_n)
        ("h_cheval_gagne",     ["id_cheval"],                    "y_gagnant", prior_g, 5,  0),
        ("h_cheval_place",     ["id_cheval"],                    "y_place",   prior_p, 5,  0),
        ("h_driver_gagne",     ["id_driver"],                    "y_gagnant", prior_g, 30, 0),
        ("h_driver_place",     ["id_driver"],                    "y_place",   prior_p, 30, 0),
        ("h_entr_gagne",       ["id_entraineur"],                "y_gagnant", prior_g, 30, 0),
        ("h_couple_gagne",     ["id_cheval", "id_driver"],       "y_gagnant", prior_g, 5,  0),
        ("h_attelage_gagne",   ["id_driver", "id_entraineur"],   "y_gagnant", prior_g, 15, 0),
    ]
    for nom, cles, cible, prior, pn, min_n in specs:
        taux, eff = _taux_glissant(df, cles, cible, prior=prior, pseudo_n=pn, min_n=min_n)
        df[nom] = taux
        df[f"{nom}_n"] = eff

    # --- Aptitudes : l'axe « talent sur type de sol / distance / piste » ---
    aptitudes = [
        ("a_terrain",    ["id_cheval", "c_terrain"],        "y_place", prior_p, 4),
        ("a_distance",   ["id_cheval", "c_bande_distance"], "y_place", prior_p, 4),
        ("a_hippodrome", ["id_cheval", "hippodrome_code"],  "y_place", prior_p, 4),
        ("a_discipline", ["id_cheval", "discipline"],       "y_place", prior_p, 4),
    ]
    for nom, cles, cible, prior, pn in aptitudes:
        taux, eff = _taux_glissant(df, cles, cible, prior=prior, pseudo_n=pn)
        df[nom] = taux
        df[f"{nom}_n"] = eff
        # Écart à la forme générale du cheval : « ce cheval est-il MEILLEUR
        # que lui-même sur ce terrain ? » Bien plus informatif que le taux brut,
        # qui ne fait que redire le niveau global du cheval.
        df[f"{nom}_delta"] = taux - df["h_cheval_place"]

    # --- Lignée : père, père de mère, croisement ---
    # C'est l'axe généalogie du projet d'origine. Les produits d'un étalon
    # partagent des aptitudes ; on mesure ça sur les courses ANTÉRIEURES
    # des autres produits, jamais sur la course en cours.
    lignee = [
        ("g_pere",             ["nom_pere"],                    "y_place", prior_p, 40),
        ("g_pere_mere",        ["nom_pere_mere"],               "y_place", prior_p, 40),
        ("g_croisement",       ["nom_pere", "nom_pere_mere"],   "y_place", prior_p, 20),
        ("g_pere_terrain",     ["nom_pere", "c_terrain"],       "y_place", prior_p, 25),
        ("g_pere_mere_terrain", ["nom_pere_mere", "c_terrain"], "y_place", prior_p, 25),
        ("g_pere_distance",    ["nom_pere", "c_bande_distance"], "y_place", prior_p, 25),
        ("g_pere_discipline",  ["nom_pere", "discipline"],      "y_place", prior_p, 25),
    ]
    for nom, cles, cible, prior, pn in lignee:
        taux, eff = _taux_glissant(df, cles, cible, prior=prior, pseudo_n=pn)
        df[nom] = taux
        df[f"{nom}_n"] = eff
    # Spécialisation de la lignée : l'étalon transmet-il une aptitude
    # PARTICULIÈRE au lourd, au-delà de sa qualité moyenne ?
    df["g_pere_terrain_delta"] = df["g_pere_terrain"] - df["g_pere"]
    df["g_pere_mere_terrain_delta"] = df["g_pere_mere_terrain"] - df["g_pere_mere"]
    df["g_pere_distance_delta"] = df["g_pere_distance"] - df["g_pere"]

    # --- Marché ---
    if avec_marche and "cote_finale" in df.columns:
        cote = pd.to_numeric(df["cote_finale"], errors="coerce")
        df["mkt_cote"] = cote
        brute = 1.0 / cote.replace(0, np.nan)
        # Les probabilités implicites somment à ~1,25 (le prélèvement PMU).
        # On renormalise par course pour obtenir une vraie probabilité.
        somme = df.groupby("course_id", sort=False)["mkt_cote"].transform(
            lambda s: (1.0 / pd.to_numeric(s, errors="coerce").replace(0, np.nan)).sum()
        )
        df["mkt_proba_implicite"] = brute / somme
        df["mkt_rang_cote"] = df.groupby("course_id", sort=False)["mkt_cote"].rank(method="min")
        if "cote_ouverture" in df.columns:
            ouv = pd.to_numeric(df["cote_ouverture"], errors="coerce")
            # Cote qui baisse = argent qui rentre = le marché en sait plus.
            df["mkt_derive"] = np.log(cote / ouv.replace(0, np.nan))
        else:
            df["mkt_derive"] = np.nan
    else:
        for col in COLONNES_MARCHE:
            df[col] = np.nan

    return df


def colonnes_features(df: pd.DataFrame, *, avec_marche: bool = False) -> list[str]:
    """Colonnes à donner au modèle. Tout le reste est métadonnée ou cible."""
    prefixes = ("c_", "p_", "mus_", "r_", "h_", "a_", "g_")
    cols = [c for c in df.columns if c.startswith(prefixes)]
    cols += [c for c in ("discipline", "specialite", "c_terrain") if c in df.columns]
    if avec_marche:
        cols += [c for c in COLONNES_MARCHE if c in df.columns]
    # Anti-erreur : aucune colonne de résultat ne doit passer.
    interdites = {
        "ordre_arrivee", "y_gagnant", "y_place", "statut_arrivee", "temps_officiel_ms",
        "reduction_km_ms", "commentaire_apres_course", "distance_cheval_precedent",
        "est_exploitable",
    }
    fuites = interdites.intersection(cols)
    if fuites:
        raise ValueError(f"colonnes de résultat dans les features : {sorted(fuites)}")
    return sorted(set(cols))
