"""
Extraction du dataset plat depuis PostgreSQL.

Une ligne = un partant, avec son contexte de course et sa cote finale.
C'est exactement ce que `features.construire()` attend en entrée.

⚠️ POINT CRITIQUE POUR LA PRÉDICTION
Les features glissantes (forme, aptitudes, lignée) se calculent sur
l'ensemble du cadre de données trié dans le temps. Pour prédire les courses
de CE SOIR, il faut donc charger l'historique ET les courses du jour dans
le MÊME appel, construire les features sur le tout, puis ne garder que les
lignes du jour.

Charger uniquement les courses du jour donnerait des features vides : le
cheval n'aurait aucune course antérieure dans le cadre, donc aucun
historique. C'est l'erreur qui fait qu'un modèle « marche à l'entraînement
et sort n'importe quoi en production ».
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import pandas as pd

log = logging.getLogger("pmu.dataset")

# La cote de référence : dernier relevé du Simple Gagnant avant le départ.
_SQL = """
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
    p.age, p.sexe, p.place_corde, p.handicap_poids, p.deferre, p.oeilleres,
    p.musique, p.nombre_courses, p.nombre_victoires, p.nombre_places,
    p.gains_carriere, p.gains_annee_en_cours,
    p.id_driver, p.id_entraineur,
    pd.nom_affiche     AS driver,
    pe.nom_affiche     AS entraineur,
    p.statut, p.ordre_arrivee,
    cf.cote_finale, cv.cote_ouverture
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
ORDER BY c.heure_depart NULLS LAST, c.course_id, p.num_pmu
"""


def charger(conn, depuis: date, jusqua: date) -> pd.DataFrame:
    """
    Dataset plat sur une plage de dates.

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

    with conn.cursor(row_factory=psycopg.rows.tuple_row) as cur:
        cur.execute(_SQL, {"depuis": depuis, "jusqua": jusqua})
        colonnes = [d.name for d in cur.description]
        df = pd.DataFrame(cur.fetchall(), columns=colonnes)

    log.info("%d partants sur %d courses (%s → %s)",
             len(df), df["course_id"].nunique() if len(df) else 0, depuis, jusqua)
    return df


def charger_pour_prediction(conn, jour: date, profondeur_jours: int = 900) -> pd.DataFrame:
    """
    Historique + courses du jour, dans un seul cadre.

    `profondeur_jours` fixe la mémoire du modèle. 900 jours (~2,5 ans)
    couvre largement la carrière utile d'un cheval de course tout en
    gardant le calcul des features en quelques secondes.
    """
    return charger(conn, jour - timedelta(days=profondeur_jours), jour)


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
