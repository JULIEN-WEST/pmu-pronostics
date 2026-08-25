"""
PMU_RESET — une destruction qui ne doit jamais se répéter.

LE MÉCANISME

`PMU_RESET` porte un jeton (une date, un numéro de version). Au
démarrage, si ce jeton n'a jamais été consigné, la base est vidée et le
jeton est enregistré. Laisser la variable en place ne détruit donc rien
aux redémarrages suivants — c'est toute la promesse faite dans la
documentation de la stack.

LE TROU, ET POURQUOI IL COMPTE

Le jeton était consigné à la FIN de `_preparer_base`, après
`apply_schema`. Tout échec entre le vidage et cette écriture — schéma
invalide, migration en erreur, base momentanément indisponible —
laissait la base vidée SANS trace du jeton. Le redémarrage suivant la
vidait à nouveau. Une variable oubliée dans la stack devenait une
boucle de destruction silencieuse : rien dans l'interface ne la
signale, et le symptôme (« mes données disparaissent ») ne pointe pas
vers sa cause.

Le correctif consigne le jeton dans la MÊME transaction que le vidage,
en créant à la main la seule table dont cette garantie dépend.

Ces tests tournent sur une vraie base PostgreSQL : le comportement à
vérifier est transactionnel, une doublure ne prouverait rien.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE / "src"))

psycopg = pytest.importorskip("psycopg")

from pmu import db  # noqa: E402

DSN = os.environ.get("PMU_TEST_DSN")
pytestmark = pytest.mark.skipif(not DSN, reason="PMU_TEST_DSN non défini")

SCHEMA = RACINE / "sql" / "001_schema.sql"


@pytest.fixture
def conn():
    from psycopg.rows import dict_row
    with psycopg.connect(DSN, row_factory=dict_row) as c:
        c.execute("DROP SCHEMA IF EXISTS pmu CASCADE")
        c.execute("CREATE SCHEMA pmu")
        c.execute("SET search_path TO pmu, public")
        c.commit()
        yield c
        c.execute("DROP SCHEMA IF EXISTS pmu CASCADE")
        c.commit()


def _prepare(conn):
    db.apply_schema(conn, SCHEMA)
    conn.commit()


def test_un_jeton_neuf_vide_la_base(conn):
    _prepare(conn)
    db.journal(conn, "programme", "25082026", "OK")
    conn.commit()
    assert db.deja_collecte(conn, "programme", "25082026")

    assert db.reinitialiser(conn, "2026-08-26") is True
    _prepare(conn)
    assert not db.deja_collecte(conn, "programme", "25082026"), "la base devait être vidée"


def test_le_meme_jeton_ne_vide_pas_deux_fois(conn):
    """La promesse faite à l'utilisateur : laisser la variable en place
    ne détruit pas la base à chaque relance."""
    _prepare(conn)
    assert db.reinitialiser(conn, "2026-08-26") is True
    _prepare(conn)
    db.journal(conn, "programme", "25082026", "OK")
    conn.commit()

    assert db.reinitialiser(conn, "2026-08-26") is False
    assert db.deja_collecte(conn, "programme", "25082026"), (
        "un second appel avec le même jeton a effacé la base"
    )


def test_le_jeton_est_consigne_des_le_vidage(conn):
    """
    LE correctif. Avant même que le schéma complet soit appliqué, la
    trace doit exister — sinon un échec d'application du schéma rouvre
    la porte à un second vidage.
    """
    _prepare(conn)
    db.reinitialiser(conn, "2026-08-26")
    # Rien d'autre n'a tourné : pas d'apply_schema, pas de commit ajouté.
    assert db.deja_collecte(conn, "reset", "2026-08-26"), (
        "le jeton doit être consigné dans la transaction du vidage"
    )


def test_un_echec_apres_le_vidage_ne_provoque_pas_un_second_vidage(conn):
    """
    LE scénario de la boucle, rejoué. `apply_schema` échoue juste après
    la réinitialisation ; au redémarrage suivant, la base ne doit PAS
    être vidée une seconde fois.
    """
    _prepare(conn)
    assert db.reinitialiser(conn, "2026-08-26") is True

    # Le schéma ne s'applique pas — on simule l'échec en ne l'appliquant
    # tout simplement pas, ce qui est l'état exact laissé par une
    # exception à cet endroit.
    conn.rollback()

    # Redémarrage : le jeton est là, donc rien ne doit être détruit.
    assert db.reinitialiser(conn, "2026-08-26") is False


def test_un_jeton_vide_ne_touche_a_rien(conn):
    """Variable absente ou vide : aucune destruction, jamais."""
    _prepare(conn)
    db.journal(conn, "programme", "25082026", "OK")
    conn.commit()
    for jeton in ("", "   "):
        assert db.reinitialiser(conn, jeton.strip()) is False
    assert db.deja_collecte(conn, "programme", "25082026")


def test_changer_le_jeton_vide_de_nouveau(conn):
    """C'est la porte de sortie documentée : changer la valeur relance."""
    _prepare(conn)
    db.reinitialiser(conn, "2026-08-26")
    _prepare(conn)
    db.journal(conn, "programme", "25082026", "OK")
    conn.commit()

    assert db.reinitialiser(conn, "2026-08-27") is True
    _prepare(conn)
    assert not db.deja_collecte(conn, "programme", "25082026")


def test_le_premier_demarrage_sur_base_nue_fonctionne(conn):
    """
    Au tout premier démarrage la table du journal n'existe pas : la
    vérification du jeton doit échouer proprement et laisser le vidage
    se faire, sans lever.
    """
    conn.execute("DROP SCHEMA IF EXISTS pmu CASCADE")
    conn.execute("CREATE SCHEMA pmu")
    conn.commit()
    assert db.reinitialiser(conn, "2026-08-26") is True
    assert db.deja_collecte(conn, "reset", "2026-08-26")
