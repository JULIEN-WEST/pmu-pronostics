"""
Tests de l'amorçage automatique.

Ce qui compte ici n'est pas que le rattrapage marche — c'est qu'il ne se
REFASSE PAS. Un amorçage rejoué à chaque redémarrage du conteneur, c'est
deux heures de requêtes inutiles vers une API non publique à chaque fois
que Portainer redéploie la stack.
"""

from __future__ import annotations

import os
import sys
from datetime import date
from pathlib import Path

import pytest

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE / "scripts"))

DSN = os.environ.get("PMU_TEST_DSN")
if DSN:
    os.environ["DATABASE_URL"] = DSN

from pmu import db  # noqa: E402
from pmu.planificateur import _cadre  # noqa: E402


# ---------------------------------------------------------------------
# Encadrés de journal — ils sont ce que l'utilisateur lira dans Portainer
# ---------------------------------------------------------------------

def test_cadre_est_aligne():
    """Un encadré désaligné donne l'impression que le programme déraille."""
    rendu = _cadre("TITRE", ["ligne courte", "une ligne nettement plus longue"])
    lignes = [l for l in rendu.split("\n") if l]
    largeurs = {len(l) for l in lignes}
    assert len(largeurs) == 1, f"largeurs hétérogènes : {sorted(largeurs)}"


def test_cadre_supporte_un_titre_plus_long_que_le_contenu():
    rendu = _cadre("UN TITRE TRES LONG QUI DEPASSE", ["court"])
    lignes = [l for l in rendu.split("\n") if l]
    assert len({len(l) for l in lignes}) == 1


def test_cadre_supporte_les_lignes_vides():
    rendu = _cadre("T", ["a", "", "b"])
    lignes = [l for l in rendu.split("\n") if l]
    assert len({len(l) for l in lignes}) == 1


# ---------------------------------------------------------------------
# Drapeau d'amorçage
# ---------------------------------------------------------------------

pytestmark_db = pytest.mark.skipif(not DSN, reason="PMU_TEST_DSN non défini")


@pytestmark_db
def test_drapeau_amorcage_empeche_la_reprise():
    """
    `deja_collecte` sur ('amorcage', 'termine') est ce qui distingue un
    premier démarrage d'un redémarrage. Sans lui, chaque redéploiement
    Portainer relancerait deux heures de collecte.
    """
    with db.connect(DSN) as conn:
        conn.execute("DROP SCHEMA IF EXISTS pmu CASCADE")
        conn.commit()
        db.apply_schema(conn, str(RACINE / "sql" / "001_schema.sql"))

        assert db.deja_collecte(conn, "amorcage", "termine") is False

        db.journal(conn, "amorcage", "termine", "OK")
        conn.commit()
        assert db.deja_collecte(conn, "amorcage", "termine") is True

        # Rejouer le journal ne doit pas dupliquer la ligne.
        db.journal(conn, "amorcage", "termine", "OK")
        conn.commit()
        n = conn.execute(
            "SELECT count(*) AS n FROM collecte_journal WHERE ressource = 'amorcage'"
        ).fetchone()["n"]
        assert n == 1


@pytestmark_db
def test_un_echec_ne_pose_pas_le_drapeau():
    """
    Si l'entraînement échoue, l'amorçage doit rester « à faire » pour être
    retenté au prochain tour — pas être marqué terminé.
    """
    with db.connect(DSN) as conn:
        conn.execute("DROP SCHEMA IF EXISTS pmu CASCADE")
        conn.commit()
        db.apply_schema(conn, str(RACINE / "sql" / "001_schema.sql"))

        db.journal(conn, "amorcage", "termine", "ERREUR", message="entraînement KO")
        conn.commit()
        # `deja_collecte` ne considère que le statut OK.
        assert db.deja_collecte(conn, "amorcage", "termine") is False
