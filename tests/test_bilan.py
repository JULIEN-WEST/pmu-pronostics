"""
Le bilan de production.

Ce que ce fichier protège : la capacité à répondre honnêtement à
« mes dix favoris sont battus, le modèle est-il cassé ? ».

Trois exigences.

  1. Compter juste. Un favori gagnant est un favori gagnant.
  2. Ne PAS confondre un défaut de collecte avec une contre-performance.
     Une course arrivée dont aucun partant n'est classé 1ᵉʳ compterait
     comme un échec du modèle alors qu'elle ne prouve rien — elle doit
     être écartée et signalée.
  3. Donner l'intervalle de confiance. Sans lui, dix courses ratées
     ressemblent à une preuve, alors qu'elles arrivent une fois sur
     dix-huit avec un modèle parfaitement sain.
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

from pmu import evaluate as ev  # noqa: E402


# ---------------------------------------------------------------------
# L'intervalle de confiance
# ---------------------------------------------------------------------

def test_zero_sur_dix_n_exclut_pas_un_modele_sain():
    """
    LE calcul qui évite de jeter un modèle correct. Zéro réussite sur
    dix courses laisse l'intervalle ouvert bien au-delà de 25 % : on ne
    peut rien conclure, et il faut le dire.
    """
    bas, haut = ev._intervalle_binomial(0, 10)
    assert bas == 0.0
    assert haut > 0.25, (
        f"intervalle [0 ; {haut:.1%}] — il devrait englober un taux de 25 %, "
        "sinon on conclurait à tort que le modèle est cassé"
    )


def test_zero_sur_trois_cents_exclut_un_modele_sain():
    """Le contre-test : avec assez de courses, l'intervalle se referme."""
    bas, haut = ev._intervalle_binomial(0, 300)
    assert haut < 0.05, "sur 300 courses, zéro réussite doit être concluant"


def test_l_intervalle_encadre_le_taux():
    for succes, n in [(25, 100), (1, 3), (140, 500), (0, 1)]:
        bas, haut = ev._intervalle_binomial(succes, n)
        assert 0.0 <= bas <= succes / n <= haut <= 1.0, (succes, n, bas, haut)


def test_l_intervalle_sur_un_effectif_nul():
    assert ev._intervalle_binomial(0, 0) == (0.0, 1.0)


def test_l_intervalle_se_resserre_avec_l_effectif():
    large = ev._intervalle_binomial(5, 20)
    etroit = ev._intervalle_binomial(250, 1000)
    assert (etroit[1] - etroit[0]) < (large[1] - large[0])


# ---------------------------------------------------------------------
# Le bilan, contre une vraie base
# ---------------------------------------------------------------------

pytestmark = pytest.mark.skipif(not DSN, reason="PMU_TEST_DSN non défini")


@pytest.fixture(scope="module")
def base():
    import tests.test_integration as ti  # réutilise l'amorçage existant
    from pmu import db, predict

    with db.connect(DSN) as conn:
        conn.execute("DROP SCHEMA IF EXISTS pmu CASCADE")
        conn.commit()
        ti._semer(conn, n_courses=150)
        jour = conn.execute(
            "SELECT max(date_reunion) AS d FROM course").fetchone()["d"]
        predict.DOSSIER_MODELES = RACINE / ".tmp_modeles_bilan"
        predict.entrainer(conn, avec_marche=False, jusqua=jour)
        predict.pronostiquer(conn, jour)
        yield conn, jour
        conn.execute("DROP SCHEMA IF EXISTS pmu CASCADE")
        conn.commit()


def test_le_bilan_compte_des_courses(base):
    conn, jour = base
    b = ev.bilan_production(conn, modele="sans_marche",
                            depuis=date(2000, 1, 1), jusqua=jour)
    assert b["n_courses"] > 0, "aucune course jugée alors que les arrivées sont là"
    assert 0.0 <= b["top1_taux"] <= 1.0
    assert b["top1_reussites"] <= b["n_courses"]
    bas, haut = b["top1_ic95"]
    assert bas <= b["top1_taux"] <= haut


def test_le_bilan_s_affiche(base):
    conn, jour = base
    texte = ev.afficher_bilan(
        ev.bilan_production(conn, modele="sans_marche",
                            depuis=date(2000, 1, 1), jusqua=jour))
    assert "favori gagnant" in texte
    assert "intervalle à 95 %" in texte
    assert "Dix courses ne prouvent rien" in texte


def test_une_course_sans_gagnant_est_ecartee_et_signalee(base):
    """
    Le cas qui fausse tout : l'arrivée de la course est connue, mais
    aucun partant n'a été mis à jour. Sans ce filtre, elle compte comme
    un échec du modèle alors que c'est la collecte qui a manqué.
    """
    conn, jour = base
    avant = ev.bilan_production(conn, modele="sans_marche",
                               depuis=date(2000, 1, 1), jusqua=jour)

    cible = conn.execute(
        """SELECT c.course_id FROM course c
            WHERE c.ordre_arrivee IS NOT NULL
              AND EXISTS (SELECT 1 FROM pronostic p WHERE p.course_id = c.course_id)
            LIMIT 1""").fetchone()
    assert cible, "la fixture ne contient aucune course exploitable"
    conn.execute("UPDATE partant SET ordre_arrivee = NULL WHERE course_id = %s",
                 (cible["course_id"],))
    conn.commit()

    apres = ev.bilan_production(conn, modele="sans_marche",
                               depuis=date(2000, 1, 1), jusqua=jour)
    assert apres["anomalies"]["courses_sans_gagnant"] >= 1
    assert apres["n_courses"] == avant["n_courses"] - 1, (
        "la course mutilée aurait dû être écartée du décompte"
    )
    assert "défaut de collecte" in ev.afficher_bilan(apres)


def test_le_bilan_sur_une_periode_vide(base):
    conn, _ = base
    b = ev.bilan_production(conn, modele="sans_marche",
                            depuis=date(1990, 1, 1), jusqua=date(1990, 12, 31))
    assert b["n_courses"] == 0
    assert "aucun" in ev.afficher_bilan(b).lower()


def test_le_bilan_sur_un_modele_inexistant(base):
    conn, jour = base
    b = ev.bilan_production(conn, modele="modele_fantome",
                            depuis=date(2000, 1, 1), jusqua=jour)
    assert b["n_courses"] == 0


def test_l_api_sert_le_bilan(base):
    from fastapi.testclient import TestClient
    from pmu.api import app

    r = TestClient(app).get("/bilan")
    assert r.status_code == 200
    corps = r.json()
    assert "n_courses" in corps and "anomalies" in corps
