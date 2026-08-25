"""
Propagation de l'arrivée vers les partants.

LE BUG QU'ON RÉPARE ICI

Le bilan de production a rendu ceci en vrai :

    {"n_courses": 0,
     "anomalies": {"courses_sans_gagnant": 29},
     "message": "aucune course exploitable : les arrivées ne sont pas
                 renseignées au niveau des partants"}

Vingt-neuf courses arrivées, aucun partant classé premier. L'arrivée
entrait en base à deux niveaux et à deux moments : sur la COURSE dès le
rafraîchissement du programme, sur les PARTANTS seulement lors du
re-téléchargement des participants — que le planificateur ne faisait
qu'à 23 h 30.

Entre les deux, le tableau de bord affichait TOUS les favoris comme
battus, et le bilan ne pouvait juger aucune course. Le modèle n'y était
pour rien : c'est ce qui rendait le diagnostic si trompeur.

La réparation ne demande aucun appel à l'API — l'information est déjà
en base. Ces tests vérifient qu'elle est exacte, idempotente, et qu'elle
n'écrase jamais une place déjà connue.
"""

from __future__ import annotations

import os
import sys
from datetime import date
from pathlib import Path

import pytest

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE / "src"))

DSN = os.environ.get("PMU_TEST_DSN")
if DSN:
    os.environ["DATABASE_URL"] = DSN

pytestmark = pytest.mark.skipif(not DSN, reason="PMU_TEST_DSN non défini")


@pytest.fixture
def base():
    from pmu import db
    with db.connect(DSN) as conn:
        conn.execute("DROP SCHEMA IF EXISTS pmu CASCADE")
        conn.commit()
        db.apply_schema(conn, RACINE / "sql" / "001_schema.sql")
        conn.execute("INSERT INTO hippodrome (code, libelle_long) "
                     "VALUES ('VIN', 'HIPPODROME DE VINCENNES')")
        conn.execute(
            """INSERT INTO reunion (date_reunion, num_officiel, hippodrome_code)
               VALUES (%s, 1, 'VIN')""", (date(2026, 8, 25),))
        conn.commit()
        yield conn
        conn.execute("DROP SCHEMA IF EXISTS pmu CASCADE")
        conn.commit()


def _course(conn, num_ordre: int, arrivee, partants=8):
    """Une course avec son arrivée au niveau COURSE, et des partants vierges."""
    from psycopg.types.json import Jsonb
    row = conn.execute(
        """INSERT INTO course (date_reunion, num_reunion, num_ordre, ordre_arrivee)
           VALUES (%s, 1, %s, %s) RETURNING course_id""",
        (date(2026, 8, 25), num_ordre,
         Jsonb(arrivee) if arrivee is not None else None)).fetchone()
    cid = row["course_id"]
    for n in range(1, partants + 1):
        conn.execute(
            "INSERT INTO partant (course_id, num_pmu, statut) VALUES (%s, %s, 'PARTANT')",
            (cid, n))
    conn.commit()
    return cid


def _places(conn, cid):
    return {r["num_pmu"]: r["ordre_arrivee"] for r in conn.execute(
        "SELECT num_pmu, ordre_arrivee FROM partant WHERE course_id = %s ORDER BY num_pmu",
        (cid,)).fetchall()}


# ---------------------------------------------------------------------

def test_la_forme_habituelle_liste_de_listes(base):
    """`[[3],[7],[1]]` : le 3 gagne, le 7 est 2e, le 1 est 3e."""
    from pmu import db
    cid = _course(base, 1, [[3], [7], [1]])
    assert db.propager_arrivees(base) == 3
    base.commit()
    p = _places(base, cid)
    assert p[3] == 1 and p[7] == 2 and p[1] == 3
    assert p[2] is None, "un cheval hors de l'arrivée ne doit pas être classé"


def test_la_forme_plate_sans_sous_listes(base):
    """Le PMU écrit parfois `[3, 7, 1]`. Les deux doivent marcher."""
    from pmu import db
    cid = _course(base, 2, [3, 7, 1])
    db.propager_arrivees(base)
    base.commit()
    p = _places(base, cid)
    assert p[3] == 1 and p[7] == 2 and p[1] == 3


def test_les_ex_aequo_partagent_le_rang(base):
    """`[[3],[1,9]]` : le 1 et le 9 sont deuxièmes ensemble."""
    from pmu import db
    cid = _course(base, 3, [[3], [1, 9]], partants=10)
    db.propager_arrivees(base)
    base.commit()
    p = _places(base, cid)
    assert p[3] == 1 and p[1] == 2 and p[9] == 2


def test_les_numeros_en_chaine_sont_acceptes(base):
    """L'API rend tantôt des nombres, tantôt des chaînes."""
    from pmu import db
    cid = _course(base, 4, [["3"], ["7"]])
    db.propager_arrivees(base)
    base.commit()
    p = _places(base, cid)
    assert p[3] == 1 and p[7] == 2


def test_une_place_deja_connue_n_est_jamais_ecrasee(base):
    """
    La source directe reste prioritaire. Écraser une place déjà
    collectée reviendrait à faire confiance à la projection plus qu'aux
    participants eux-mêmes.
    """
    from pmu import db
    cid = _course(base, 5, [[3], [7]])
    base.execute("UPDATE partant SET ordre_arrivee = 4 WHERE course_id = %s AND num_pmu = 3",
                 (cid,))
    base.commit()
    db.propager_arrivees(base)
    base.commit()
    p = _places(base, cid)
    assert p[3] == 4, "une place existante a été écrasée"
    assert p[7] == 2, "les autres partants doivent quand même être remplis"


def test_la_propagation_est_idempotente(base):
    from pmu import db
    _course(base, 6, [[3], [7], [1]])
    premier = db.propager_arrivees(base)
    base.commit()
    second = db.propager_arrivees(base)
    base.commit()
    assert premier == 3
    assert second == 0, "un second passage ne doit plus rien écrire"


def test_une_course_sans_arrivee_est_ignoree(base):
    from pmu import db
    cid = _course(base, 7, None)
    assert db.propager_arrivees(base) == 0
    base.commit()
    assert all(v is None for v in _places(base, cid).values())


@pytest.mark.parametrize("arrivee", [
    [], {}, "3-7-1", [None], [[]], [["abc"]], [{"num": 3}],
])
def test_une_arrivee_deformee_ne_leve_pas(base, arrivee):
    """
    Toute forme inattendue doit être ignorée en silence. Une exception
    ici bloquerait la collecte de la journée entière.
    """
    from pmu import db
    _course(base, 8, arrivee)
    db.propager_arrivees(base)   # ne doit pas lever
    base.commit()


def test_le_filtre_par_jour(base):
    """On doit pouvoir ne réparer qu'une journée."""
    from pmu import db
    cid = _course(base, 9, [[3], [7]])
    assert db.propager_arrivees(base, date(2020, 1, 1)) == 0
    base.commit()
    assert db.propager_arrivees(base, date(2026, 8, 25)) == 2
    base.commit()
    assert _places(base, cid)[3] == 1


def test_le_bilan_redevient_exploitable_apres_reparation(base):
    """
    Le bout-en-bout : c'est exactement la situation rencontrée en
    production — des courses arrivées qu'aucun partant ne renseigne.
    """
    from pmu import db, evaluate as ev, predict

    base.execute(predict.SQL_TABLE)
    cid = _course(base, 10, [[3], [7], [1]])
    base.execute("""INSERT INTO pronostic (course_id, num_pmu, proba, rang, modele)
                    VALUES (%s, 3, 0.4, 1, 'sans_marche'),
                           (%s, 7, 0.3, 2, 'sans_marche')""", (cid, cid))
    base.commit()

    avant = ev.bilan_production(base, modele="sans_marche",
                                depuis=date(2000, 1, 1), jusqua=date(2030, 1, 1))
    assert avant["n_courses"] == 0
    assert avant["anomalies"]["courses_sans_gagnant"] == 1

    db.propager_arrivees(base)
    base.commit()

    apres = ev.bilan_production(base, modele="sans_marche",
                                depuis=date(2000, 1, 1), jusqua=date(2030, 1, 1))
    assert apres["n_courses"] == 1
    assert apres["anomalies"]["courses_sans_gagnant"] == 0
    assert apres["top1_reussites"] == 1, "le favori du modèle avait bien gagné"
