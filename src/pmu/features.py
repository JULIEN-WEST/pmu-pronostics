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

from .normalize import parse_marge, parse_musique

# Colonnes issues du marché : légitimes, mais à manier séparément.
# Un modèle qui les voit apprend surtout à recopier la cote — il sera très
# bon en apparence et sans aucune valeur, puisque la cote est déjà connue
# de tous. On entraîne donc DEUX modèles (cf. train.py) :
#   - sans marché  → cherche un écart exploitable
#   - avec marché  → borne haute de ce qui est prévisible
COLONNES_MARCHE = [
    "mkt_cote", "mkt_proba_implicite", "mkt_rang_cote", "mkt_derive",
    # Trajectoire de la cote. Ce sont des variables de MARCHÉ : elles
    # rendent le modèle `avec_marche` nettement meilleur, mais elles ne
    # peuvent pas créer un écart FACE au marché, puisqu'elles sont le
    # marché. Leur usage réel est ailleurs : savoir si l'argent va vers
    # notre sélection ou s'en détourne, donc quand se taire.
    "mkt_derive_tardive", "mkt_amplitude", "mkt_volatilite",
    "mkt_n_releves", "mkt_rang_derive",
]

# Cibles ordinales. Une course de 14 partants n'apprend qu'un seul bit
# au modèle si la cible est « qui gagne ». Les seuils intermédiaires
# exploitent l'ORDRE D'ARRIVÉE, donc bien plus d'information par course,
# sur exactement les mêmes données.
SEUILS_ORDINAUX = [("y_gagnant", 1), ("y_top2", 2), ("y_top3", 3), ("y_top5", 5)]

# Regroupement des disciplines en FAMILLES.
#
# Pourquoi regrouper plutôt que prendre la discipline telle quelle : les
# trois disciplines d'obstacle réunies pèsent moins qu'un seul jour
# d'attelé. Les scinder donnerait trois modèles faméliques là où un seul,
# nourri des trois, tient debout. À l'inverse attelé et monté, qui sont
# tous deux du trot, ne partagent NI les mêmes drivers, NI la même
# incidence du déferrage, NI la même prime au poids : les fondre serait
# la même erreur en sens inverse.
#
# Ce découpage est une hypothèse, pas une vérité. `ModeleParDiscipline`
# le met à l'épreuve famille par famille et n'en garde que ce qui gagne.
FAMILLES = {
    "ATTELE": "ATTELE",
    "MONTE": "MONTE",
    "PLAT": "PLAT",
    "HAIES": "OBSTACLE",
    "STEEPLECHASE": "OBSTACLE",
    "STEEPLE-CHASE": "OBSTACLE",
    "STEEPLE CHASE": "OBSTACLE",
    "CROSS": "OBSTACLE",
    "CROSS-COUNTRY": "OBSTACLE",
}


def famille(discipline) -> "pd.Series | str":
    """Discipline PMU → famille. Accepte une série ou une chaîne."""
    if isinstance(discipline, pd.Series):
        up = discipline.fillna("").astype(str).str.upper().str.strip()
        return up.map(FAMILLES).fillna("AUTRE")
    return FAMILLES.get(str(discipline or "").upper().strip(), "AUTRE")

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


def _passe_du_cheval(df: pd.DataFrame, valeurs: pd.Series) -> dict:
    """
    Statistiques d'une grandeur NUMÉRIQUE sur les seules courses
    antérieures du cheval : moyenne, meilleur, dernière valeur, effectif.

    Pourquoi c'est plus simple ici que dans `_taux_glissant` : un cheval
    ne court qu'une fois par course. La ligne courante est donc la seule
    de sa course pour cette clé, et un décalage d'un cran suffit — pas
    besoin de retrancher le cumul de la course. Le piège des demi-frères
    qui se voient l'un l'autre n'existe pas quand la clé est le cheval
    lui-même.

    C'est ce qui permet d'exploiter la RÉDUCTION KILOMÉTRIQUE, jusqu'ici
    inutilisée. La place dit qui a gagné ; le chrono dit à quelle vitesse.
    Une 5e place en 1'12 vaut mieux qu'une victoire en 1'16, et aucune
    feature de classement ne peut le savoir.
    """
    v = pd.to_numeric(valeurs, errors="coerce").where(df["est_exploitable"])
    cle = df["id_cheval"]
    presente = v.notna()

    g_somme = v.fillna(0.0).groupby(cle, sort=False).cumsum() - v.fillna(0.0)
    g_n = presente.astype(float).groupby(cle, sort=False).cumsum() - presente.astype(float)

    precedente = v.groupby(cle, sort=False).shift(1)
    meilleur = precedente.groupby(cle, sort=False).cummin()
    # `cummin` laisse un trou tant qu'aucune valeur n'est encore connue ;
    # on propage la dernière valeur atteinte pour boucher ces trous.
    meilleur = meilleur.groupby(cle, sort=False).ffill()

    return {
        "moy": (g_somme / g_n.replace(0, np.nan)),
        "n": g_n,
        "best": meilleur,
        "derniere": precedente.groupby(cle, sort=False).ffill(),
    }


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
    # Décomposition ordinale : « dans les k premiers » pour plusieurs k.
    # Chaque seuil est une cible binaire, et l'ensemble reconstitue
    # l'ordre d'arrivée sans avoir à écrire un modèle de rang complet.
    for nom, k in SEUILS_ORDINAUX:
        df[nom] = ((place > 0) & (place <= k)).astype(float)
    # ── Deux notions distinctes, et les confondre coûte cher ──────────
    #
    # `est_exploitable` : cette ligne COMPTE dans l'historique. Elle a un
    #   résultat connu, donc elle alimente les taux glissants — forme du
    #   cheval, réussite du driver, aptitude de la lignée.
    #
    # `est_cible` : cette ligne peut servir d'EXEMPLE, à l'entraînement
    #   comme à la prédiction. Cela exige des colonnes que les lignes
    #   importées n'ont pas : cote, gains, musique, entraîneur.
    #
    # Les 108 000 performances importées sont exploitables sans être des
    # cibles. Elles donnent au modèle la mémoire de chaque cheval — deux
    # ans et demi de carrière au lieu des deux mois de collecte directe —
    # tout en restant hors du jeu d'entraînement, où leurs colonnes vides
    # brouilleraient l'apprentissage.
    df["est_exploitable"] = place.notna() & df["statut"].ne("NON_PARTANT")
    source = df["source"] if "source" in df.columns else pd.Series("direct", index=df.index)
    df["est_cible"] = df["est_exploitable"] & source.eq("direct")

    # --- Contexte de course ---
    # `famille` sert à router vers un modèle spécialisé (cf. train.py). Elle
    # n'est volontairement PAS préfixée : `colonnes_features()` ne la
    # ramassera donc pas, et elle ne partira jamais dans le modèle — où
    # elle serait de toute façon constante à l'intérieur d'une famille.
    df["famille"] = famille(df["discipline"])
    df["c_distance"] = pd.to_numeric(df["distance"], errors="coerce")
    df["c_bande_distance"] = _bande_distance(df["c_distance"])
    df["c_terrain"] = _classe_terrain(df.get("etat_terrain", pd.Series(index=df.index, dtype=object)))
    df["c_nb_partants"] = n_part
    df["c_allocation"] = pd.to_numeric(df.get("montant_prix"), errors="coerce")
    # --- Conditions d'engagement : collectées depuis le début, lues
    #     seulement maintenant. Le pénétromètre est le seul chiffre du
    #     lot : « bon » est un adjectif, 3,8 est une mesure.
    if "penetrometre" in df.columns:
        df["c_penetrometre"] = pd.to_numeric(df["penetrometre"], errors="coerce")
    for brut, nom in [("categorie_particularite", "c_categorie"),
                      ("categorie_statut", "c_statut_course"),
                      ("condition_age", "c_condition_age"),
                      ("condition_sexe", "c_condition_sexe"),
                      ("corde", "c_sens_corde")]:
        if brut in df.columns:
            df[nom] = df[brut].astype("category")
    if "nombre_declares_partants" in df.columns:
        declares = pd.to_numeric(df["nombre_declares_partants"], errors="coerce")
        df["c_declares"] = declares
        # Beaucoup de non-partants change la physionomie d'une course :
        # moins de monde, moins de trafic, et un lot qui n'est plus celui
        # que le public avait jugé à l'ouverture.
        df["c_taux_non_partants"] = 1.0 - (n_part / declares.replace(0, np.nan))

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

    # --- Vitesse : le chrono, jusqu'ici resté en base sans servir ---
    #
    # `reduction_km_ms` est le temps au kilomètre, en millisecondes. Plus
    # bas = plus rapide. C'est la mesure de valeur la moins bruitée dont
    # on dispose : une place dépend du lot rencontré, un chrono non.
    #
    # ⚠️ C'est une colonne de RÉSULTAT. Elle n'est lisible que sur les
    # courses ANTÉRIEURES du cheval — d'où `_passe_du_cheval`, et jamais
    # `df["reduction_km_ms"]` en direct. La liste `interdites` en fin de
    # fichier verrouille cette règle.
    if "reduction_km_ms" in df.columns:
        red = _passe_du_cheval(df, df["reduction_km_ms"])
        df["v_reduction_moy"] = red["moy"]
        df["v_reduction_best"] = red["best"]
        df["v_reduction_derniere"] = red["derniere"]
        df["v_reduction_n"] = red["n"]
        # Écart entre la moyenne et le record : un cheval régulier et un
        # cheval capable d'un coup d'éclat ne se parient pas pareil.
        df["v_reduction_irregularite"] = df["v_reduction_moy"] - df["v_reduction_best"]
        # Progression : dernier chrono contre moyenne de carrière.
        # Négatif = le cheval va plus vite qu'à son habitude.
        df["v_reduction_progres"] = df["v_reduction_derniere"] - df["v_reduction_moy"]

    # --- Marge d'arrivée : la qualité de la défaite ---
    # Deuxième d'un nez ou deuxième de vingt longueurs, la place est la
    # même et la course n'a rien à voir.
    if "distance_cheval_precedent" in df.columns:
        marge = df["distance_cheval_precedent"].map(parse_marge)
        pas = _passe_du_cheval(df, marge)
        df["v_marge_moy"] = pas["moy"]
        df["v_marge_derniere"] = pas["derniere"]

    # --- Recul au trot : un handicap explicite, connu avant le départ ---
    if "handicap_distance" in df.columns:
        recul = pd.to_numeric(df["handicap_distance"], errors="coerce").fillna(0.0)
        df["c_recul"] = recul
        # 25 m de recul sur 2700 m ne pèsent pas comme sur 1609 m.
        df["c_recul_rel"] = recul / df["c_distance"].replace(0, np.nan)
        # Le recul ne vaut que par comparaison au lot : tout le monde
        # reculé de 25 m, c'est une course sans handicap.
        df["c_recul_relatif_lot"] = recul - df.groupby("course_id", sort=False)[
            "c_recul"].transform("min")

    # --- Position dans le lot : le signal le plus sous-estimé ---
    # Ce qui compte n'est pas le niveau absolu du cheval mais son niveau
    # RELATIF aux autres partants de SA course. Un rang intra-course est
    # calculé sur des colonnes déjà anti-fuite, il n'introduit rien.
    par_course = df.groupby("course_id", sort=False)
    # `ascending=True` pour les grandeurs où PLUS BAS est MEILLEUR
    # (moyenne de musique, chrono au kilomètre, marge encaissée).
    rangs = [("p_gains_par_course", "r_gains", False),
             ("p_taux_victoire", "r_taux_vict", False),
             ("mus_moy", "r_musique", True),
             ("p_gains_log", "r_gains_tot", False),
             ("v_reduction_moy", "r_vitesse", True),
             ("v_reduction_best", "r_vitesse_best", True),
             ("v_marge_moy", "r_marge", True)]
    for col, nom, croissant in rangs:
        if col in df.columns:
            df[nom] = par_course[col].rank(pct=True, ascending=croissant)

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
        # L'écurie : un propriétaire qui aligne trente chevaux n'a pas le
        # même taux qu'un particulier qui en a un. Information disponible
        # avant le départ, et jamais exploitée jusqu'ici.
        ("h_proprio_place",    ["id_proprietaire"],              "y_place",   prior_p, 30, 0),
    ]
    specs = [s for s in specs if all(c in df.columns for c in s[1])]
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

        # --- Trajectoire complète de la cote ---
        # ⚠️ `pd.to_numeric(df.get("absente"))` rend un SCALAIRE nan, pas
        # une série : la ligne suivante casse alors sur `.replace`. Une
        # base d'avant cette version n'a aucune de ces colonnes.
        def _serie(nom):
            if nom not in df.columns:
                return pd.Series(np.nan, index=df.index, dtype="float64")
            return pd.to_numeric(df[nom], errors="coerce")

        t15 = _serie("cote_t15")
        # Le mouvement du DERNIER quart d'heure, isolé du reste. C'est
        # là que se place l'argent tardif, réputé le mieux informé.
        df["mkt_derive_tardive"] = np.log(cote / t15.replace(0, np.nan))
        cmin = _serie("cote_min")
        cmax = _serie("cote_max")
        df["mkt_amplitude"] = np.log(cmax / cmin.replace(0, np.nan))
        df["mkt_volatilite"] = _serie("cote_ecart_type")
        df["mkt_n_releves"] = _serie("cote_n")
        # La dérive n'a de sens que RELATIVE : dans une course, les cotes
        # somment à une constante, donc si tout le monde baisse, personne
        # ne baisse vraiment.
        df["mkt_rang_derive"] = df.groupby("course_id", sort=False)[
            "mkt_derive_tardive"].rank(pct=True, ascending=True)
    else:
        for col in COLONNES_MARCHE:
            df[col] = np.nan

    return df


def colonnes_features(df: pd.DataFrame, *, avec_marche: bool = False) -> list[str]:
    """Colonnes à donner au modèle. Tout le reste est métadonnée ou cible."""
    prefixes = ("c_", "p_", "mus_", "r_", "h_", "a_", "g_", "v_")
    cols = [c for c in df.columns if c.startswith(prefixes)]
    cols += [c for c in ("discipline", "specialite", "c_terrain") if c in df.columns]
    if avec_marche:
        cols += [c for c in COLONNES_MARCHE if c in df.columns]
    # Anti-erreur : aucune colonne de résultat ne doit passer.
    interdites = {
        "ordre_arrivee", "y_place", "statut_arrivee", "temps_officiel_ms",
        "reduction_km_ms", "commentaire_apres_course", "distance_cheval_precedent",
        "est_exploitable", "est_cible", "source",
    }
    # Toutes les cibles ordinales sont des résultats : y_top3 dans les
    # features reviendrait à donner l'arrivée au modèle.
    interdites |= {nom for nom, _ in SEUILS_ORDINAUX}
    fuites = interdites.intersection(cols)
    if fuites:
        raise ValueError(f"colonnes de résultat dans les features : {sorted(fuites)}")
    return sorted(set(cols))
