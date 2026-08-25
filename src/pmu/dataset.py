"""
Extraction du dataset depuis PostgreSQL.

Une ligne = un cheval dans une course. Deux origines cohabitent :

  `direct`   partant collecté sur l'API : riche (cote, driver, gains,
             entraîneur), et c'est le SEUL type de ligne sur lequel on
             entraîne et on prédit.

  `importe`  course passée ramenée par /performances-detaillees : pauvre
             (ni cote, ni gains, ni entraîneur) mais PROFONDE. Elle ne
             sert qu'à nourrir l'historique — forme du cheval, aptitude
             au terrain, lignée — jamais d'exemple d'entraînement.

Pourquoi ce mélange : sur 60 jours de collecte, il y a 35 000 partants
directs et 108 000 performances importées. Ignorer les secondes, c'est
priver le modèle de la mémoire de chaque cheval — précisément ce qu'il
lui faut pour juger une forme.

⚠️ POINT CRITIQUE POUR LA PRÉDICTION
Les features glissantes se calculent sur l'ensemble du cadre trié dans le
temps. Pour prédire les courses de CE SOIR, il faut donc charger
l'historique ET les courses du jour dans le MÊME appel, construire les
features sur le tout, puis ne garder que les lignes du jour.

Charger uniquement les courses du jour donnerait des features vides : le
cheval n'aurait aucune course antérieure dans le cadre. C'est l'erreur qui
fait qu'un modèle « marche à l'entraînement et sort n'importe quoi en
production ».
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd

log = logging.getLogger("pmu.dataset")

# La cote de référence : dernier relevé du Simple Gagnant avant le départ.
_SQL_DIRECT = """
WITH cote_fin AS (
    SELECT DISTINCT ON (co.course_id, co.num_pmu)
           co.course_id, co.num_pmu, co.rapport AS cote_finale
      FROM cote co
      JOIN course c ON c.course_id = co.course_id
     WHERE co.type_pari IN ('SIMPLE_GAGNANT', 'E_SIMPLE_GAGNANT')
       AND (c.heure_depart IS NULL OR co.releve_le <= c.heure_depart)
     ORDER BY co.course_id, co.num_pmu, co.releve_le DESC
),
cote_ouv AS (
    SELECT DISTINCT ON (co.course_id, co.num_pmu)
           co.course_id, co.num_pmu, co.rapport AS cote_ouverture
      FROM cote co
     WHERE co.type_pari IN ('SIMPLE_GAGNANT', 'E_SIMPLE_GAGNANT')
     ORDER BY co.course_id, co.num_pmu, co.releve_le ASC
),
-- ── LA TRAJECTOIRE DE LA COTE ────────────────────────────────────────
-- Jusqu'ici on ne lisait que deux points : l'ouverture et la clôture.
-- La série complète en dit bien plus. Une cote qui s'effondre dans le
-- dernier quart d'heure, c'est de l'argent qui rentre tard, souvent le
-- mieux informé. Une cote qui oscille, c'est un lot que personne
-- n'arrive à départager.
cote_serie AS (
    SELECT co.course_id, co.num_pmu,
           count(*)                              AS cote_n,
           min(co.rapport)                       AS cote_min,
           max(co.rapport)                       AS cote_max,
           stddev_samp(ln(nullif(co.rapport, 0))) AS cote_ecart_type
      FROM cote co
      JOIN course c ON c.course_id = co.course_id
     WHERE co.type_pari IN ('SIMPLE_GAGNANT', 'E_SIMPLE_GAGNANT')
       AND co.rapport > 0
       AND (c.heure_depart IS NULL OR co.releve_le <= c.heure_depart)
     GROUP BY co.course_id, co.num_pmu
),
-- Dernier relevé à T-15 min : le point de comparaison qui isole le
-- mouvement de la toute fin. ⚠️ Aucune fuite : c'est antérieur au
-- départ, donc connu au moment où l'on pronostique.
cote_avant AS (
    SELECT DISTINCT ON (co.course_id, co.num_pmu)
           co.course_id, co.num_pmu, co.rapport AS cote_t15
      FROM cote co
      JOIN course c ON c.course_id = co.course_id
     WHERE co.type_pari IN ('SIMPLE_GAGNANT', 'E_SIMPLE_GAGNANT')
       AND c.heure_depart IS NOT NULL
       AND co.releve_le <= c.heure_depart - interval '15 minutes'
     ORDER BY co.course_id, co.num_pmu, co.releve_le DESC
)
SELECT
    c.course_id, c.heure_depart, c.date_reunion, c.num_reunion, c.num_ordre,
    c.libelle          AS libelle_course,
    c.discipline, c.specialite, c.distance, c.etat_terrain,
    c.montant_prix, c.nombre_partants, c.depart_type,
    r.hippodrome_code,
    h.libelle_long     AS hippodrome,
    p.num_pmu, p.id_cheval,
    ch.nom             AS nom_cheval,
    -- La MÈRE et l'ÉLEVEUR étaient collectés depuis le premier jour et
    -- n'avaient jamais été lus. En trot, la famille maternelle est ce
    -- que suivent les éleveurs — bien avant l'étalon.
    ch.nom_pere, ch.nom_pere_mere, ch.nom_mere, ch.id_eleveur,
    p.age, p.sexe, p.place_corde, p.handicap_poids, p.handicap_distance,
    p.deferre, p.oeilleres,
    p.musique, p.nombre_courses, p.nombre_victoires, p.nombre_places,
    p.gains_carriere, p.gains_annee_en_cours,
    p.id_driver, p.id_entraineur, p.id_proprietaire,
    pd.nom_affiche     AS driver,
    pe.nom_affiche     AS entraineur,
    p.statut, p.ordre_arrivee,
    -- ⚠️ Colonnes de RÉSULTAT. Elles ne peuvent servir que via les
    -- cumuls décalés de features.py, jamais telles quelles sur la
    -- ligne courante. `colonnes_features()` les refuse explicitement.
    p.reduction_km_ms, p.temps_officiel_ms, p.distance_cheval_precedent,
    cf.cote_finale, cv.cote_ouverture,
    -- Conditions d'engagement : déjà collectées, jamais lues jusqu'ici.
    -- Le pénétromètre est une MESURE du terrain, là où `etat_terrain`
    -- n'est qu'un adjectif — « bon » ne veut pas dire la même chose à
    -- Vincennes et à Cagnes.
    c.penetrometre, c.categorie_particularite, c.categorie_statut,
    c.conditions, c.condition_age, c.condition_sexe, c.corde,
    c.nombre_declares_partants,
    cs.cote_n, cs.cote_min, cs.cote_max, cs.cote_ecart_type, ca.cote_t15,
    -- Météo relevée sur place. La pluie des 24 h précédentes fait le
    -- terrain ; l'adjectif « bon » posé par le commissaire ne dit rien
    -- de ce qui est tombé la nuit d'avant.
    mt.temperature AS meteo_temperature, mt.pluie_jour AS meteo_pluie_jour,
    mt.pluie_24h   AS meteo_pluie_24h,  mt.vent_max    AS meteo_vent,
    mt.humidite    AS meteo_humidite,
    -- Avis de l'analyste : classement complet et cote probable, publiés
    -- avant la course. C'est un AVIS, corrélé au marché : il est traité
    -- comme tel dans features.py et exclu du modèle `sans_marche`.
    px.rang_expert, px.cote_probable, px.est_crible,
    'direct'::text     AS source
FROM partant p
JOIN course     c  ON c.course_id = p.course_id
JOIN reunion    r  ON r.date_reunion = c.date_reunion AND r.num_officiel = c.num_reunion
LEFT JOIN hippodrome h ON h.code = r.hippodrome_code
LEFT JOIN cheval    ch ON ch.id_cheval = p.id_cheval
LEFT JOIN personne  pd ON pd.id = p.id_driver
LEFT JOIN personne  pe ON pe.id = p.id_entraineur
LEFT JOIN cote_fin   cf ON cf.course_id = p.course_id AND cf.num_pmu = p.num_pmu
LEFT JOIN cote_ouv   cv ON cv.course_id = p.course_id AND cv.num_pmu = p.num_pmu
LEFT JOIN cote_serie cs ON cs.course_id = p.course_id AND cs.num_pmu = p.num_pmu
LEFT JOIN cote_avant ca ON ca.course_id = p.course_id AND ca.num_pmu = p.num_pmu
LEFT JOIN meteo      mt ON mt.hippodrome_code = r.hippodrome_code
                       AND mt.date_course = c.date_reunion
LEFT JOIN pronostic_expert px ON px.course_id = p.course_id
                             AND px.num_pmu = p.num_pmu
WHERE c.date_reunion BETWEEN %(depuis)s AND %(jusqua)s
"""

# Les lignes importées. Trois précautions :
#
#   1. course_id SYNTHÉTIQUE et NÉGATIF. Le calcul anti-fuite exclut « la
#      course en cours » ; il faut donc un identifiant. On regroupe par
#      (date, hippodrome, distance) pour que deux chevaux d'une même
#      épreuve passée partagent le leur — sinon ils se verraient
#      mutuellement. Négatif pour ne jamais heurter un vrai course_id.
#
#   2. DÉDUPLICATION. Une course peut être à la fois collectée en direct
#      et rapportée par l'endpoint des performances. Sans le NOT EXISTS,
#      elle compterait double dans tous les cumuls.
#
#   3. GÉNÉALOGIE RÉCUPÉRÉE. Ces lignes n'apportent pas le père, mais on
#      connaît le cheval : la jointure sur `cheval` rend leur lignée, ce
#      qui multiplie l'effectif des statistiques d'étalon.
_SQL_IMPORTE = """
SELECT
    -(dense_rank() OVER (ORDER BY pp.date_course, pp.hippodrome_lib, pp.distance))::bigint
                       AS course_id,
    pp.date_course::timestamptz AS heure_depart,
    pp.date_course     AS date_reunion,
    NULL::integer      AS num_reunion,
    NULL::integer      AS num_ordre,
    pp.nom_prix        AS libelle_course,
    pp.discipline, pp.specialite, pp.distance, pp.etat_terrain,
    pp.allocation      AS montant_prix,
    pp.nb_participants AS nombre_partants,
    NULL::text         AS depart_type,
    pp.hippodrome_code,
    pp.hippodrome_lib  AS hippodrome,
    NULL::smallint     AS num_pmu,
    pp.id_cheval,
    ch.nom             AS nom_cheval,
    ch.nom_pere, ch.nom_pere_mere, ch.nom_mere, ch.id_eleveur,
    NULL::smallint     AS age,
    ch.sexe,
    pp.corde           AS place_corde,
    pp.poids_jockey    AS handicap_poids,
    NULL::integer      AS handicap_distance,
    pp.deferre,
    pp.oeillere        AS oeilleres,
    NULL::text         AS musique,
    NULL::smallint     AS nombre_courses,
    NULL::smallint     AS nombre_victoires,
    NULL::smallint     AS nombre_places,
    NULL::numeric      AS gains_carriere,
    NULL::numeric      AS gains_annee_en_cours,
    NULL::bigint       AS id_driver,
    NULL::bigint       AS id_entraineur,
    NULL::bigint       AS id_proprietaire,
    pp.nom_jockey      AS driver,
    NULL::text         AS entraineur,
    'PARTANT'::text    AS statut,
    pp.place           AS ordre_arrivee,
    -- Le vrai trésor de ces lignes : le CHRONO. La place dit qui a
    -- gagné, la réduction kilométrique dit à quelle vitesse — donc si
    -- une victoire valait quelque chose, et si une 5e place cachait
    -- une bonne course.
    pp.reduction_km_ms,
    pp.temps_premier_ms AS temps_officiel_ms,
    pp.distance_avec_precedent AS distance_cheval_precedent,
    NULL::numeric      AS cote_finale,
    NULL::numeric      AS cote_ouverture,
    NULL::numeric      AS penetrometre,
    NULL::text         AS categorie_particularite,
    NULL::text         AS categorie_statut,
    NULL::text         AS conditions,
    NULL::text         AS condition_age,
    NULL::text         AS condition_sexe,
    NULL::text         AS corde,
    NULL::smallint     AS nombre_declares_partants,
    NULL::bigint       AS cote_n,
    NULL::numeric      AS cote_min,
    NULL::numeric      AS cote_max,
    NULL::double precision AS cote_ecart_type,
    NULL::numeric      AS cote_t15,
    NULL::numeric      AS meteo_temperature,
    NULL::numeric      AS meteo_pluie_jour,
    NULL::numeric      AS meteo_pluie_24h,
    NULL::numeric      AS meteo_vent,
    NULL::numeric      AS meteo_humidite,
    NULL::smallint     AS rang_expert,
    NULL::numeric      AS cote_probable,
    NULL::boolean      AS est_crible,
    'importe'::text    AS source
FROM performance_passee pp
LEFT JOIN cheval ch ON ch.id_cheval = pp.id_cheval
WHERE pp.date_course BETWEEN %(depuis)s AND %(jusqua)s
  AND pp.place IS NOT NULL
  AND NOT EXISTS (
        SELECT 1
          FROM partant p2
          JOIN course c2 ON c2.course_id = p2.course_id
         WHERE p2.id_cheval = pp.id_cheval
           AND c2.date_reunion = pp.date_course
  )
"""

_ORDRE = "\nORDER BY heure_depart NULLS LAST, course_id, num_pmu NULLS FIRST"


def charger(conn, depuis: date, jusqua: date, *,
            avec_historique_importe: bool = True) -> pd.DataFrame:
    """
    Dataset sur une plage de dates.

    ⚠️ Surtout PAS `pd.read_sql(sql, conn)` ici. La connexion du projet
    utilise `row_factory=dict_row` ; pandas itère alors les CLÉS de chaque
    dict au lieu de ses valeurs, et rend un cadre où chaque colonne
    contient son propre nom en boucle — `statut` vaut `"statut"` partout.
    Aucune exception n'est levée : le pipeline tourne, `est_exploitable`
    tombe à zéro, et le modèle s'entraîne sur du vide.

    On passe donc par un curseur en tuples et on nomme les colonnes
    depuis `cursor.description`.
    """
    import psycopg.rows

    if avec_historique_importe:
        sql = f"SELECT * FROM (\n{_SQL_DIRECT}\nUNION ALL\n{_SQL_IMPORTE}\n) tout{_ORDRE}"
    else:
        sql = _SQL_DIRECT + _ORDRE

    with conn.cursor(row_factory=psycopg.rows.tuple_row) as cur:
        cur.execute(sql, {"depuis": depuis, "jusqua": jusqua})
        colonnes = [d.name for d in cur.description]
        df = pd.DataFrame(cur.fetchall(), columns=colonnes)

    if len(df):
        par_source = df["source"].value_counts().to_dict()
        log.info("%d lignes (%s) sur %d courses (%s → %s)",
                 len(df),
                 ", ".join(f"{v} {k}" for k, v in sorted(par_source.items())),
                 df["course_id"].nunique(), depuis, jusqua)
    else:
        log.info("aucune ligne entre %s et %s", depuis, jusqua)
    return df


def charger_pour_prediction(conn, jour: date, profondeur_jours: int = 900,
                            **kw) -> pd.DataFrame:
    """
    Historique + courses du jour, dans un seul cadre.

    `profondeur_jours` fixe la mémoire du modèle. 900 jours (~2,5 ans)
    couvre largement la carrière utile d'un cheval de course — et c'est
    précisément la profondeur qu'apportent les performances importées,
    là où la collecte directe ne remonte qu'à son premier jour.
    """
    return charger(conn, jour - timedelta(days=profondeur_jours), jour, **kw)



# ---------------------------------------------------------------------
# Les rapports réellement payés
# ---------------------------------------------------------------------

SQL_RAPPORTS_REELS = """
    SELECT r.course_id,
           r.combinaison::smallint AS num_pmu,
           r.rapport               AS rapport_reel
      FROM rapport_definitif r
     WHERE r.type_pari IN ('SIMPLE_GAGNANT', 'E_SIMPLE_GAGNANT')
       AND r.rapport IS NOT NULL
       AND r.combinaison ~ '^[0-9]+$'
"""


def charger_rapports_reels(conn) -> pd.DataFrame:
    """
    Ce qui a été versé pour 1 € misé, par cheval gagnant.

    Sert à payer la simulation de rentabilité au tarif réel plutôt qu'à
    une cote pré-départ corrigée d'un prélèvement supposé. Absente ou
    vide, la table ne casse rien : la simulation retombe sur l'estimation.

    Le filtre `~ '^[0-9]+$'` écarte les combinaisons de paris multiples
    (« 9-5 ») : seul le Simple Gagnant a un sens ici, et son casting en
    smallint échouerait sur un tiret.
    """
    try:
        rows = conn.execute(SQL_RAPPORTS_REELS).fetchall()
    except Exception as exc:  # noqa: BLE001
        log.warning("rapports réels indisponibles : %s", exc)
        conn.rollback()
        return pd.DataFrame(columns=["course_id", "num_pmu", "rapport_reel"])
    d = pd.DataFrame(rows, columns=["course_id", "num_pmu", "rapport_reel"])
    for c in d.columns:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["course_id", "num_pmu"])
    # SIMPLE_GAGNANT et E_SIMPLE_GAGNANT peuvent coexister pour le même
    # cheval. Sans ce dédoublonnage, la jointure DUPLIQUERAIT le partant
    # dans la fenêtre de test — un pari compté deux fois.
    return d.drop_duplicates(subset=["course_id", "num_pmu"], keep="first")


def stats(conn) -> dict:
    """Volumétrie — sert à la santé de l'API et au capteur HA."""
    q = """
    SELECT
      (SELECT count(*) FROM course)                                   AS courses,
      (SELECT count(*) FROM partant)                                  AS partants,
      (SELECT count(*) FROM cheval)                                   AS chevaux,
      (SELECT count(*) FROM performance_passee)                       AS perfs_importees,
      (SELECT count(*) FROM cote)                                     AS releves_cote,
      (SELECT count(*) FROM course WHERE ordre_arrivee IS NOT NULL)   AS courses_arrivees,
      (SELECT min(date_reunion) FROM course)                          AS depuis,
      (SELECT max(date_reunion) FROM course)                          AS jusqua,
      (SELECT count(*) FROM collecte_journal WHERE statut = 'ERREUR') AS erreurs_collecte
    """
    row = conn.execute(q).fetchone()
    return dict(row) if row else {}
