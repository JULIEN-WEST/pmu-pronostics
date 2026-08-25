"""
Accès base : upserts idempotents.

Toute la collecte doit pouvoir être relancée sans produire de doublon ni
écraser une donnée plus fraîche par une plus ancienne. D'où le parti pris :
`ON CONFLICT DO UPDATE` partout, avec COALESCE pour ne jamais remplacer une
valeur connue par un NULL (une course rejouée avant l'arrivée ne doit pas
effacer l'arrivée déjà collectée).
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import contextmanager
from typing import Any, Iterable

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

log = logging.getLogger(__name__)

DSN = os.environ.get("DATABASE_URL", "postgresql://pmu:pmu@localhost:5432/pmu")


@contextmanager
def connect(dsn: str | None = None):
    with psycopg.connect(dsn or DSN, row_factory=dict_row) as conn:
        conn.execute("SET search_path TO pmu, public")
        yield conn


def apply_schema(conn, path: str = "sql/001_schema.sql") -> None:
    # La vérification passe AVANT l'application : sur une base à l'ancien
    # format, le script échouerait d'abord sur un index, avec un message
    # incompréhensible (« column nom_norme does not exist ») qui n'aide
    # personne à comprendre qu'il faut recréer le volume.
    _verifier_schema(conn)
    with open(path, encoding="utf-8") as fh:
        conn.execute(fh.read())
    conn.commit()


def reinitialiser(conn, jeton: str) -> bool:
    """
    Repart d'une base vide, une seule fois par `jeton`.

    Sert quand une correction rend l'historique déjà collecté inutilisable
    — par exemple le décalage d'un jour des dates de réunion. Supprimer un
    volume Docker depuis Portainer est laborieux et facile à rater ; une
    variable d'environnement l'est beaucoup moins.

    Le jeton (une date, un numéro de version, n'importe quelle chaîne) est
    consigné dans le journal après application. Tant qu'il ne change pas,
    les redémarrages suivants ne réinitialisent plus rien : laisser la
    variable en place ne détruit donc pas la base à chaque relance.
    """
    if not jeton:
        return False
    try:
        if deja_collecte(conn, "reset", jeton):
            return False
    except Exception:  # noqa: BLE001 — schéma absent au tout premier démarrage
        conn.rollback()

    log.warning(
        "PMU_RESET=%s : réinitialisation complète de la base demandée. "
        "Toutes les données collectées sont effacées, la collecte va reprendre "
        "de zéro.", jeton
    )
    conn.execute("DROP SCHEMA IF EXISTS pmu CASCADE")
    conn.execute("CREATE SCHEMA pmu")
    conn.commit()
    return True


def _verifier_schema(conn) -> None:
    """
    Migration automatique quand elle est sans risque.

    Le schéma s'applique avec `CREATE TABLE IF NOT EXISTS` : une base
    créée par une version antérieure n'est donc PAS modifiée. Or
    `cheval.id_cheval` est passé de `bigint` à `text` — le PMU renvoie
    « KHAMEPHIS GAME-AKITA-ZARAK », pas un nombre.

    Plutôt que d'exiger une manipulation de volumes dans Portainer, on
    tranche selon ce que la base contient RÉELLEMENT :

      - aucune donnée exploitable  -> on recrée le schéma, c'est sans
        perte et l'utilisateur n'a rien à faire ;
      - des données               -> on refuse, en expliquant. Effacer
        l'historique des cotes de quelqu'un sans le lui demander serait
        impardonnable : il ne se reconstitue jamais.
    """
    row = conn.execute(
        """
        SELECT data_type FROM information_schema.columns
         WHERE table_schema = 'pmu' AND table_name = 'cheval'
           AND column_name = 'id_cheval'
        """
    ).fetchone()
    if not row or row["data_type"] in ("text", "character varying"):
        return  # base neuve ou déjà au bon format

    ancien = row["data_type"]
    try:
        n = conn.execute("SELECT count(*) AS n FROM pmu.partant").fetchone()["n"]
    except Exception:  # noqa: BLE001 — table absente sur un schéma très ancien
        conn.rollback()
        n = 0

    if n == 0:
        log.warning(
            "schéma obsolète (cheval.id_cheval en %s) et base vide "
            "— recréation automatique, aucune donnée perdue", ancien
        )
        conn.execute("DROP SCHEMA IF EXISTS pmu CASCADE")
        conn.execute("CREATE SCHEMA pmu")
        conn.execute("SET search_path TO pmu, public")
        conn.commit()
        return

    raise RuntimeError(
        "\n"
        "+==============================================================+\n"
        "|  BASE DE DONNEES OBSOLETE - INTERVENTION NECESSAIRE          |\n"
        "+==============================================================+\n"
        f"|  cheval.id_cheval est en '{ancien}' au lieu de 'text'.\n"
        f"|  La base contient {n} partants : je ne l'efface pas tout seul.\n"
        "|\n"
        "|  L'historique des cotes ne se reconstitue jamais apres coup.\n"
        "|  Sauvegarder d'abord si ces donnees comptent :\n"
        "|    docker exec pmu-db pg_dump -U pmu pmu | gzip > pmu.sql.gz\n"
        "|\n"
        "|  Puis, dans Portainer :\n"
        "|    1. Stacks -> pmu -> Stop this stack\n"
        "|    2. Volumes -> cocher pmu_pmu_db et pmu_pmu_data -> Remove\n"
        "|    3. Stacks -> pmu -> Start / Pull and redeploy\n"
        "+==============================================================+"
    )


# ---------------------------------------------------------------------
# Référentiels — renvoient un identifiant, mémoïsés par appelant
# ---------------------------------------------------------------------

def upsert_personne(conn, nom_affiche: str | None, nom_norme: str | None) -> int | None:
    if not nom_norme:
        return None
    row = conn.execute(
        """
        INSERT INTO personne (nom_norme, nom_affiche)
        VALUES (%s, %s)
        ON CONFLICT (nom_norme) DO UPDATE SET nom_affiche = EXCLUDED.nom_affiche
        RETURNING id
        """,
        (nom_norme, nom_affiche or nom_norme),
    ).fetchone()
    return row["id"] if row else None


def upsert_hippodrome(conn, h: dict) -> None:
    if not h or not h.get("code"):
        return
    conn.execute(
        """
        INSERT INTO hippodrome (code, libelle_court, libelle_long, pays_code, pays_libelle)
        VALUES (%(code)s, %(libelle_court)s, %(libelle_long)s, %(pays_code)s, %(pays_libelle)s)
        ON CONFLICT (code) DO UPDATE SET
            libelle_court = COALESCE(EXCLUDED.libelle_court, hippodrome.libelle_court),
            libelle_long  = COALESCE(EXCLUDED.libelle_long,  hippodrome.libelle_long)
        """,
        h,
    )


def upsert_cheval(conn, p: dict, id_eleveur: int | None = None) -> int | None:
    """
    Le cheval vient d'un participant. On ne connaît la généalogie que par
    NOM à ce stade ; la résolution vers des identifiants se fait plus tard
    (link_genealogie), quand on a croisé assez de courses pour identifier
    le père comme un cheval de la base.
    """
    from .normalize import norm_horse

    if p.get("id_cheval") is None:
        return None
    conn.execute(
        """
        INSERT INTO cheval (id_cheval, nom, nom_norme, sexe, race, pays,
                            nom_pere, nom_mere, nom_pere_mere, id_eleveur)
        VALUES (%(id_cheval)s, %(nom)s, %(nom_norme)s, %(sexe)s, %(race)s, %(pays)s,
                %(nom_pere)s, %(nom_mere)s, %(nom_pere_mere)s, %(id_eleveur)s)
        ON CONFLICT (id_cheval) DO UPDATE SET
            nom           = COALESCE(EXCLUDED.nom,           cheval.nom),
            sexe          = COALESCE(EXCLUDED.sexe,          cheval.sexe),
            race          = COALESCE(EXCLUDED.race,          cheval.race),
            pays          = COALESCE(EXCLUDED.pays,          cheval.pays),
            nom_pere      = COALESCE(EXCLUDED.nom_pere,      cheval.nom_pere),
            nom_mere      = COALESCE(EXCLUDED.nom_mere,      cheval.nom_mere),
            nom_pere_mere = COALESCE(EXCLUDED.nom_pere_mere, cheval.nom_pere_mere),
            id_eleveur    = COALESCE(EXCLUDED.id_eleveur,    cheval.id_eleveur),
            maj_le        = now()
        """,
        {
            "id_cheval": p["id_cheval"],
            "nom": p.get("nom_cheval"),
            "nom_norme": norm_horse(p.get("nom_cheval")) or str(p["id_cheval"]),
            "sexe": p.get("sexe"),
            "race": p.get("race"),
            "pays": p.get("pays"),
            "nom_pere": norm_horse(p.get("nom_pere")),
            "nom_mere": norm_horse(p.get("nom_mere")),
            "nom_pere_mere": norm_horse(p.get("nom_pere_mere")),
            "id_eleveur": id_eleveur,
        },
    )
    return p["id_cheval"]


# ---------------------------------------------------------------------
# Programme
# ---------------------------------------------------------------------

def upsert_reunion(conn, r: dict) -> None:
    payload = dict(r)
    payload["meteo"] = Jsonb(payload["meteo"]) if payload.get("meteo") is not None else None
    conn.execute(
        """
        INSERT INTO reunion (date_reunion, num_officiel, hippodrome_code, nature,
                             audience, statut, pays_code, meteo)
        VALUES (%(date_reunion)s, %(num_officiel)s, %(hippodrome_code)s, %(nature)s,
                %(audience)s, %(statut)s, %(pays_code)s, %(meteo)s)
        ON CONFLICT (date_reunion, num_officiel) DO UPDATE SET
            hippodrome_code = COALESCE(EXCLUDED.hippodrome_code, reunion.hippodrome_code),
            statut          = COALESCE(EXCLUDED.statut,          reunion.statut),
            meteo           = COALESCE(EXCLUDED.meteo,           reunion.meteo)
        """,
        payload,
    )


def upsert_course(conn, c: dict) -> int | None:
    payload = dict(c)
    payload["ordre_arrivee"] = (
        Jsonb(payload["ordre_arrivee"]) if payload.get("ordre_arrivee") is not None else None
    )
    row = conn.execute(
        """
        INSERT INTO course (date_reunion, num_reunion, num_ordre, libelle, libelle_court,
            discipline, specialite, categorie_particularite, categorie_statut, conditions,
            condition_age, condition_sexe, distance, distance_unit, corde, depart_type,
            montant_prix, nombre_declares_partants, nombre_partants, etat_terrain,
            penetrometre, heure_depart, statut, ordre_arrivee, rapports_definitifs_disponibles)
        VALUES (%(date_reunion)s, %(num_reunion)s, %(num_ordre)s, %(libelle)s, %(libelle_court)s,
            %(discipline)s, %(specialite)s, %(categorie_particularite)s, %(categorie_statut)s,
            %(conditions)s, %(condition_age)s, %(condition_sexe)s, %(distance)s,
            %(distance_unit)s, %(corde)s, %(depart_type)s, %(montant_prix)s,
            %(nombre_declares_partants)s, %(nombre_partants)s, %(etat_terrain)s,
            %(penetrometre)s, %(heure_depart)s, %(statut)s, %(ordre_arrivee)s,
            %(rapports_definitifs_disponibles)s)
        ON CONFLICT (date_reunion, num_reunion, num_ordre) DO UPDATE SET
            statut          = COALESCE(EXCLUDED.statut,        course.statut),
            nombre_partants = COALESCE(EXCLUDED.nombre_partants, course.nombre_partants),
            etat_terrain    = COALESCE(EXCLUDED.etat_terrain,  course.etat_terrain),
            penetrometre    = COALESCE(EXCLUDED.penetrometre,  course.penetrometre),
            heure_depart    = COALESCE(EXCLUDED.heure_depart,  course.heure_depart),
            -- l'arrivée ne s'efface jamais une fois connue
            ordre_arrivee   = COALESCE(EXCLUDED.ordre_arrivee, course.ordre_arrivee),
            rapports_definitifs_disponibles =
                course.rapports_definitifs_disponibles OR EXCLUDED.rapports_definitifs_disponibles
        RETURNING course_id
        """,
        payload,
    ).fetchone()
    return row["course_id"] if row else None


def upsert_partant(conn, course_id: int, p: dict,
                   id_driver: int | None, id_entraineur: int | None,
                   id_proprietaire: int | None) -> None:
    payload = {
        **{k: p.get(k) for k in (
            "num_pmu", "id_cheval", "age", "sexe", "race", "driver_change",
            "place_corde", "handicap_poids", "handicap_valeur", "handicap_distance",
            "poids_condition_monte", "oeilleres", "deferre", "supplement", "engagement",
            "jument_pleine", "indicateur_inedit", "allure", "musique",
            "nombre_courses", "nombre_victoires", "nombre_places",
            "nombre_places_second", "nombre_places_troisieme",
            "gains_carriere", "gains_victoires", "gains_place",
            "gains_annee_en_cours", "gains_annee_precedente",
            "statut", "ordre_arrivee", "statut_arrivee", "temps_officiel_ms",
            "reduction_km_ms", "distance_cheval_precedent", "commentaire_apres_course",
        )},
        "course_id": course_id,
        "id_driver": id_driver,
        "id_entraineur": id_entraineur,
        "id_proprietaire": id_proprietaire,
    }
    conn.execute(
        """
        INSERT INTO partant (course_id, num_pmu, id_cheval, age, sexe, race,
            id_driver, id_entraineur, id_proprietaire, driver_change,
            place_corde, handicap_poids, handicap_valeur, handicap_distance,
            poids_condition_monte, oeilleres, deferre, supplement, engagement,
            jument_pleine, indicateur_inedit, allure, musique,
            nombre_courses, nombre_victoires, nombre_places,
            nombre_places_second, nombre_places_troisieme,
            gains_carriere, gains_victoires, gains_place,
            gains_annee_en_cours, gains_annee_precedente,
            statut, ordre_arrivee, statut_arrivee, temps_officiel_ms,
            reduction_km_ms, distance_cheval_precedent, commentaire_apres_course)
        VALUES (%(course_id)s, %(num_pmu)s, %(id_cheval)s, %(age)s, %(sexe)s, %(race)s,
            %(id_driver)s, %(id_entraineur)s, %(id_proprietaire)s, %(driver_change)s,
            %(place_corde)s, %(handicap_poids)s, %(handicap_valeur)s, %(handicap_distance)s,
            %(poids_condition_monte)s, %(oeilleres)s, %(deferre)s, %(supplement)s,
            %(engagement)s, %(jument_pleine)s, %(indicateur_inedit)s, %(allure)s, %(musique)s,
            %(nombre_courses)s, %(nombre_victoires)s, %(nombre_places)s,
            %(nombre_places_second)s, %(nombre_places_troisieme)s,
            %(gains_carriere)s, %(gains_victoires)s, %(gains_place)s,
            %(gains_annee_en_cours)s, %(gains_annee_precedente)s,
            %(statut)s, %(ordre_arrivee)s, %(statut_arrivee)s, %(temps_officiel_ms)s,
            %(reduction_km_ms)s, %(distance_cheval_precedent)s, %(commentaire_apres_course)s)
        ON CONFLICT (course_id, num_pmu) DO UPDATE SET
            statut         = COALESCE(EXCLUDED.statut,         partant.statut),
            ordre_arrivee  = COALESCE(EXCLUDED.ordre_arrivee,  partant.ordre_arrivee),
            statut_arrivee = COALESCE(EXCLUDED.statut_arrivee, partant.statut_arrivee),
            temps_officiel_ms = COALESCE(EXCLUDED.temps_officiel_ms, partant.temps_officiel_ms),
            reduction_km_ms   = COALESCE(EXCLUDED.reduction_km_ms,   partant.reduction_km_ms),
            commentaire_apres_course =
                COALESCE(EXCLUDED.commentaire_apres_course, partant.commentaire_apres_course),
            id_driver      = COALESCE(EXCLUDED.id_driver,     partant.id_driver),
            id_entraineur  = COALESCE(EXCLUDED.id_entraineur, partant.id_entraineur)
        """,
        payload,
    )


def insert_cotes(conn, course_id: int, cotes: Iterable[dict]) -> None:
    rows = [(course_id, c["num_pmu"], c["releve_le"], c["type_pari"],
             c["rapport"], c["favoris"], c["grosse_prise"], c["tendance"])
            for c in cotes if c.get("num_pmu") is not None and c.get("releve_le")]
    if not rows:
        return
    conn.cursor().executemany(
        """
        INSERT INTO cote (course_id, num_pmu, releve_le, type_pari,
                          rapport, favoris, grosse_prise, tendance)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (course_id, num_pmu, type_pari, releve_le) DO NOTHING
        """,
        rows,
    )


def insert_performances(conn, lignes: Iterable[dict]) -> int:
    rows = [
        (l["id_cheval"], l["date_course"], l["hippodrome_lib"], l["hippodrome_code"],
         l["nom_prix"], l["discipline"], l["specialite"], l["distance"], l["allocation"],
         l["nb_participants"], l["place"], l["statut_arrivee"], l["corde"], l["poids_jockey"],
         l["nom_jockey"], l["oeillere"], l["deferre"], l["etat_terrain"],
         l["temps_premier_ms"], l["reduction_km_ms"], l["distance_avec_precedent"])
        for l in lignes
    ]
    if not rows:
        return 0
    conn.cursor().executemany(
        """
        INSERT INTO performance_passee (id_cheval, date_course, hippodrome_lib, hippodrome_code,
            nom_prix, discipline, specialite, distance, allocation, nb_participants, place,
            statut_arrivee, corde, poids_jockey, nom_jockey, oeillere, deferre, etat_terrain,
            temps_premier_ms, reduction_km_ms, distance_avec_precedent)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (id_cheval, date_course, hippodrome_lib, distance) DO NOTHING
        """,
        rows,
    )
    return len(rows)


def journal(conn, ressource: str, cle: str, statut: str,
            http_code: int | None = None, message: str | None = None,
            duree_ms: int | None = None) -> None:
    conn.execute(
        """
        INSERT INTO collecte_journal (ressource, cle, statut, http_code, message, duree_ms)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (ressource, cle) DO UPDATE SET
            statut = EXCLUDED.statut, http_code = EXCLUDED.http_code,
            message = EXCLUDED.message, duree_ms = EXCLUDED.duree_ms, fait_le = now()
        """,
        (ressource, cle, statut, http_code, (message or "")[:500], duree_ms),
    )


def deja_collecte(conn, ressource: str, cle: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM collecte_journal WHERE ressource = %s AND cle = %s AND statut = 'OK'",
        (ressource, cle),
    ).fetchone()
    return row is not None


def link_genealogie(conn) -> int:
    """
    Résout nom_pere / nom_mere / nom_pere_mere vers des id_cheval quand le
    parent est lui-même présent dans la base.

    ⚠️ L'homonymie existe en élevage. On ne lie QUE si le nom normalisé est
    unique dans la base — sinon on laisse NULL et on garde le nom, plutôt
    que de rattacher un produit au mauvais étalon.
    """
    # Trois passes ciblées : plus lisible qu'un UPDATE ... FROM à trois
    # jointures optionnelles, et chacune se relit isolément.
    total = 0
    for colonne_nom, colonne_id in (
        ("nom_pere", "id_pere"),
        ("nom_mere", "id_mere"),
        ("nom_pere_mere", "id_pere_mere"),
    ):
        res = conn.execute(
            f"""
            WITH uniques AS (
                SELECT nom_norme, MIN(id_cheval) AS id_cheval
                FROM cheval GROUP BY nom_norme HAVING COUNT(*) = 1
            )
            UPDATE cheval c
               SET {colonne_id} = u.id_cheval
              FROM uniques u
             WHERE c.{colonne_id} IS NULL
               AND c.{colonne_nom} IS NOT NULL
               AND c.{colonne_nom} = u.nom_norme
               AND u.id_cheval <> c.id_cheval
            """
        )
        total += res.rowcount or 0
    return total

SQL_PROPAGER_ARRIVEES = """
WITH rangs AS (
    SELECT c.course_id,
           elem.ord::smallint AS rang,
           (num.valeur #>> '{}')::smallint AS num_pmu
      FROM course c
      CROSS JOIN LATERAL jsonb_array_elements(c.ordre_arrivee)
           WITH ORDINALITY AS elem(groupe, ord)
      -- Le PMU écrit tantôt [[3],[7],[1,9]] — le 3e rang est un ex æquo —
      -- tantôt [3,7,1]. On normalise en enveloppant les scalaires.
      CROSS JOIN LATERAL jsonb_array_elements(
           CASE WHEN jsonb_typeof(elem.groupe) = 'array'
                THEN elem.groupe ELSE jsonb_build_array(elem.groupe) END
      ) AS num(valeur)
     WHERE c.ordre_arrivee IS NOT NULL
       AND (%(jour)s::date IS NULL OR c.date_reunion = %(jour)s::date)
       AND jsonb_typeof(c.ordre_arrivee) = 'array'
       AND (num.valeur #>> '{}') ~ '^[0-9]+$'
)
UPDATE partant p
   SET ordre_arrivee = r.rang
  FROM rangs r
 WHERE p.course_id = r.course_id
   AND p.num_pmu = r.num_pmu
   AND p.ordre_arrivee IS NULL
"""


def propager_arrivees(conn, jour=None) -> int:
    """
    Recopie l'arrivée de la COURSE vers ses PARTANTS, en SQL pur.

    POURQUOI CETTE FONCTION EXISTE

    L'arrivée arrive en base à deux niveaux et à deux moments : au
    niveau de la course dès que le programme est rafraîchi, au niveau
    des partants seulement quand on re-télécharge les participants —
    ce que le planificateur ne faisait qu'à 23 h 30.

    Entre les deux, la base contenait 29 courses arrivées dont AUCUN
    partant n'était classé premier. Conséquence : le tableau de bord
    affichait tous les favoris comme battus, et le bilan de production
    ne pouvait juger aucune course. Le modèle n'y était pour rien.

    La réparation ne demande aucun appel à l'API : l'information est
    déjà là, dans `course.ordre_arrivee`. C'est une simple projection,
    idempotente, qu'on peut lancer aussi souvent qu'on veut.

    `ordre_arrivee IS NULL` dans le WHERE : on ne réécrit jamais une
    place déjà connue, la source directe reste prioritaire.
    """
    cur = conn.execute(SQL_PROPAGER_ARRIVEES, {"jour": jour})
    n = cur.rowcount or 0
    if n:
        log.info("arrivées propagées vers %d partant(s)", n)
    return n



def insert_pronostics_expert(conn, course_id: int, lignes: list[dict]) -> int:
    """Avis de l'analyste pour une course. Idempotent."""
    if not lignes:
        return 0
    conn.cursor().executemany(
        """
        INSERT INTO pronostic_expert (course_id, num_pmu, rang_expert,
                                      cote_probable, est_crible,
                                      commentaire_expert, source_expert)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (course_id, num_pmu) DO UPDATE SET
            rang_expert   = COALESCE(EXCLUDED.rang_expert,   pronostic_expert.rang_expert),
            cote_probable = COALESCE(EXCLUDED.cote_probable, pronostic_expert.cote_probable),
            est_crible    = EXCLUDED.est_crible OR pronostic_expert.est_crible,
            commentaire_expert = COALESCE(EXCLUDED.commentaire_expert,
                                          pronostic_expert.commentaire_expert),
            source_expert = COALESCE(EXCLUDED.source_expert, pronostic_expert.source_expert),
            collecte_le   = now()
        """,
        [(course_id, l["num_pmu"], l.get("rang_expert"), l.get("cote_probable"),
          bool(l.get("est_crible")), l.get("commentaire_expert"), l.get("source_expert"))
         for l in lignes],
    )
    return len(lignes)
