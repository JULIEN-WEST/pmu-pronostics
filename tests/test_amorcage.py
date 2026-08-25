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


# ---------------------------------------------------------------------
# Migration de schéma
# ---------------------------------------------------------------------

def _ancien_schema(conn, avec_donnees: bool = False) -> None:
    """Recrée la forme d'avant la correction : id_cheval en bigint."""
    conn.execute("DROP SCHEMA IF EXISTS pmu CASCADE")
    conn.execute("CREATE SCHEMA pmu")
    conn.execute("CREATE TABLE pmu.cheval (id_cheval bigint PRIMARY KEY, nom text)")
    conn.execute("CREATE TABLE pmu.partant (course_id bigint, num_pmu smallint)")
    if avec_donnees:
        conn.execute("INSERT INTO pmu.partant VALUES (1, 1), (1, 2)")
    conn.commit()


@pytestmark_db
def test_migration_automatique_si_la_base_est_vide():
    """
    Le cas réel : une base créée par l'ancienne version, restée vide parce
    que la collecte échouait. La recréer ne perd rien — on le fait donc
    sans rien demander, plutôt que d'imposer une manipulation de volumes.
    """
    from pmu.planificateur import _preparer_base

    with db.connect(DSN) as conn:
        _ancien_schema(conn)

    assert _preparer_base() is True

    with db.connect(DSN) as conn:
        t = conn.execute(
            """SELECT data_type FROM information_schema.columns
                WHERE table_schema='pmu' AND table_name='cheval'
                  AND column_name='id_cheval'"""
        ).fetchone()
        assert t["data_type"] == "text"
        # Le schéma complet a bien été appliqué derrière.
        conn.execute("SELECT count(*) FROM pmu.collecte_journal")


@pytestmark_db
def test_migration_refusee_si_la_base_contient_des_donnees():
    """
    L'historique des cotes ne se reconstitue jamais après coup. Dès qu'il
    y a des partants, on refuse et on explique — jamais d'effacement
    silencieux.
    """
    from pmu.planificateur import _preparer_base

    with db.connect(DSN) as conn:
        _ancien_schema(conn, avec_donnees=True)

    assert _preparer_base() is False

    with db.connect(DSN) as conn:
        n = conn.execute("SELECT count(*) AS n FROM pmu.partant").fetchone()["n"]
        assert n == 2, "les données existantes ne doivent pas être touchées"
        conn.execute("DROP SCHEMA IF EXISTS pmu CASCADE")
        conn.commit()


@pytestmark_db
def test_une_erreur_de_config_n_est_pas_prise_pour_une_base_lente():
    """
    Régression : la boucle de reconnexion attrapait toute exception et
    affichait « base pas encore prête » douze fois. Le vrai message —
    celui qui dit quoi faire — n'apparaissait jamais, et le conteneur
    redémarrait en boucle.

    Seules les erreurs de CONNEXION doivent déclencher un nouvel essai.
    """
    import time as _time
    from pmu.planificateur import _preparer_base

    with db.connect(DSN) as conn:
        _ancien_schema(conn, avec_donnees=True)

    debut = _time.monotonic()
    assert _preparer_base() is False
    # 12 tentatives à 5 s feraient une minute : on doit répondre tout de suite.
    assert _time.monotonic() - debut < 5, "l'erreur de config a été retentée"

    with db.connect(DSN) as conn:
        conn.execute("DROP SCHEMA IF EXISTS pmu CASCADE")
        conn.commit()


@pytestmark_db
def test_reinitialisation_ne_s_applique_qu_une_fois():
    """
    Le point critique : laisser `PMU_RESET` en place dans la stack ne doit
    PAS vider la base à chaque redémarrage du conteneur. Sans cette
    garantie, la variable serait un piège à retardement.
    """
    import os
    from pmu.planificateur import _preparer_base

    os.environ["PMU_RESET"] = "test-2026-08-25"
    try:
        assert _preparer_base() is True
        with db.connect(DSN) as conn:
            conn.execute(
                "INSERT INTO pmu.hippodrome (code, libelle_long) VALUES ('VIN', 'Vincennes')"
            )
            conn.commit()

        # Deuxième démarrage, même jeton : la donnée doit survivre.
        assert _preparer_base() is True
        with db.connect(DSN) as conn:
            n = conn.execute("SELECT count(*) AS n FROM pmu.hippodrome").fetchone()["n"]
            assert n == 1, "la base a été vidée une seconde fois"

        # Jeton différent : là, on réinitialise.
        os.environ["PMU_RESET"] = "test-2026-08-26"
        assert _preparer_base() is True
        with db.connect(DSN) as conn:
            n = conn.execute("SELECT count(*) AS n FROM pmu.hippodrome").fetchone()["n"]
            assert n == 0, "le nouveau jeton n'a pas déclenché la réinitialisation"
            conn.execute("DROP SCHEMA IF EXISTS pmu CASCADE")
            conn.commit()
    finally:
        os.environ.pop("PMU_RESET", None)


@pytestmark_db
def test_sans_jeton_aucune_reinitialisation():
    """Absence de PMU_RESET = on ne touche à rien."""
    import os
    from pmu.planificateur import _preparer_base

    os.environ.pop("PMU_RESET", None)
    assert _preparer_base() is True
    with db.connect(DSN) as conn:
        conn.execute(
            "INSERT INTO pmu.hippodrome (code, libelle_long) VALUES ('ENG', 'Enghien')"
        )
        conn.commit()
    assert _preparer_base() is True
    with db.connect(DSN) as conn:
        n = conn.execute(
            "SELECT count(*) AS n FROM pmu.hippodrome WHERE code = 'ENG'"
        ).fetchone()["n"]
        assert n == 1
        conn.execute("DROP SCHEMA IF EXISTS pmu CASCADE")
        conn.commit()


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
