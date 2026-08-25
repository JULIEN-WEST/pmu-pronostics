"""
Les deux URL que l'utilisateur ouvre pour savoir si son modèle marche,
et le déclencheur de ré-entraînement piloté par variable.

Pourquoi ça mérite des tests : ce sont les seuls chemins par lesquels
quelqu'un qui ne tape pas de commandes peut voir ce que fait sa pile.
S'ils rendent une erreur 500 ou du JSON illisible, tout le travail de
mesure en amont ne sert à personne.
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


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    import tests.test_integration as ti
    from pmu import db, predict

    dossier = tmp_path_factory.mktemp("modeles")
    os.environ["PMU_MODELES"] = str(dossier)
    predict.DOSSIER_MODELES = dossier

    with db.connect(DSN) as conn:
        conn.execute("DROP SCHEMA IF EXISTS pmu CASCADE")
        conn.commit()
        ti._semer(conn, n_courses=150)
        jour = conn.execute(
            "SELECT max(date_reunion) AS d FROM course").fetchone()["d"]
        predict.entrainer(conn, avec_marche=False, jusqua=jour)
        predict.pronostiquer(conn, jour)
        yield conn, jour, dossier
        conn.execute("DROP SCHEMA IF EXISTS pmu CASCADE")
        conn.commit()


def _client():
    from fastapi.testclient import TestClient
    from pmu.api import app
    return TestClient(app)


# ---------------------------------------------------------------------
# /rapport
# ---------------------------------------------------------------------

def test_le_rapport_est_lisible(base):
    conn, jour, dossier = base
    r = _client().get("/rapport")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain")
    texte = r.text
    assert "Rapport d'entraînement" in texte
    # Les trois blocs qui répondent aux questions posées.
    assert "Justesse" in texte
    assert "Calibration" in texte
    assert "Abstention" in texte


def test_le_rapport_dit_ce_qu_il_faut_faire_quand_il_manque(base):
    """
    Une 404 nue n'aide personne. Le message doit dire où regarder.
    """
    r = _client().get("/rapport", params={"modele": "modele_inexistant"})
    assert r.status_code == 404
    detail = r.json()["detail"]
    assert "pmu-collecteur" in detail, "le message n'oriente pas vers les journaux"


# ---------------------------------------------------------------------
# /bilan
# ---------------------------------------------------------------------

def test_le_bilan_en_texte_est_lisible(base):
    r = _client().get("/bilan", params={"format": "texte"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain")
    assert "Bilan de production" in r.text


def test_le_bilan_reste_du_json_par_defaut(base):
    """L'automatisation lit du JSON ; seul l'humain demande du texte."""
    r = _client().get("/bilan")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/json")
    assert "n_courses" in r.json()


def test_les_deux_formats_disent_la_meme_chose(base):
    j = _client().get("/bilan").json()
    t = _client().get("/bilan", params={"format": "texte"}).text
    if j.get("n_courses"):
        assert str(j["n_courses"]) in t


# ---------------------------------------------------------------------
# Le déclencheur de ré-entraînement
# ---------------------------------------------------------------------

def test_le_jeton_ne_declenche_qu_une_fois(base, monkeypatch):
    """
    Le point délicat : laisser la variable en place ne doit PAS relancer
    un entraînement à chaque redémarrage du conteneur — sinon la pile
    passe son temps à s'entraîner au lieu de collecter.
    """
    from pmu import planificateur as pl

    conn, jour, _ = base
    appels = {"n": 0}
    monkeypatch.setattr(pl, "entrainer", lambda *a, **k: appels.__setitem__("n", appels["n"] + 1))
    monkeypatch.setenv("PMU_REENTRAINER", "2026-08-25")

    assert pl.reentrainer_a_la_demande(conn) is True
    assert appels["n"] == 2, "les deux variantes doivent être ré-entraînées"

    # Deuxième démarrage, même jeton : rien ne doit repartir.
    assert pl.reentrainer_a_la_demande(conn) is False
    assert appels["n"] == 2

    # Nouveau jeton : ça repart.
    monkeypatch.setenv("PMU_REENTRAINER", "2026-08-26")
    assert pl.reentrainer_a_la_demande(conn) is True
    assert appels["n"] == 4


def test_sans_jeton_rien_ne_se_passe(base, monkeypatch):
    from pmu import planificateur as pl

    conn, _, _ = base
    monkeypatch.delenv("PMU_REENTRAINER", raising=False)
    appels = {"n": 0}
    monkeypatch.setattr(pl, "entrainer", lambda *a, **k: appels.__setitem__("n", appels["n"] + 1))
    assert pl.reentrainer_a_la_demande(conn) is False
    assert appels["n"] == 0


def test_un_echec_n_est_pas_consigne(base, monkeypatch):
    """
    Si l'entraînement échoue, le jeton ne doit pas être marqué comme
    fait : on doit pouvoir retenter au prochain démarrage sans avoir à
    inventer une nouvelle valeur.
    """
    from pmu import planificateur as pl

    conn, _, _ = base
    monkeypatch.setenv("PMU_REENTRAINER", "jeton-qui-echoue")

    def casse(*a, **k):
        raise RuntimeError("base vide")

    monkeypatch.setattr(pl, "entrainer", casse)
    assert pl.reentrainer_a_la_demande(conn) is False

    appels = {"n": 0}
    monkeypatch.setattr(pl, "entrainer", lambda *a, **k: appels.__setitem__("n", appels["n"] + 1))
    assert pl.reentrainer_a_la_demande(conn) is True, (
        "un échec a été consigné comme un succès — le jeton est brûlé"
    )
