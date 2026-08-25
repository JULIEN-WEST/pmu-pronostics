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
    ch.nom_pere, ch.nom_pere_mere,
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
    'direct'::text     AS source
FROM partant p
JOIN course     c  ON c.course_id = p.course_id
JOIN reunion    r  ON r.date_reunion = c.date_reunion AND r.num_officiel = c.num_reunion
LEFT JOIN hippodrome h ON h.code = r.hippodrome_code
LEFT JOIN cheval    ch ON ch.id_cheval = p.id_cheval
LEFT JOIN personne  pd ON pd.id = p.id_driver
LEFT JOIN personne  pe ON pe.id = p.id_entraineur
LEFT JOIN cote_fin  cf ON cf.course_id = p.course_id AND cf.num_pmu = p.num_pmu
LEFT JOIN cote_ouv  cv ON cv.course_id = p.course_id AND cv.num_pmu = p.num_pmu
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
    ch.nom_pere, ch.nom_pere_mere,
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
