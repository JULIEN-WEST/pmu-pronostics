"""
La page HTML servie à Home Assistant.

Trois choses à garantir, et la troisième est celle qu'on oublie.

  1. Elle est AUTONOME : aucune ressource externe. Le tableau de bord
     doit s'afficher sur un réseau coupé d'Internet.
  2. Elle ne CASSE PAS quand les données manquent — journée vide, cheval
     sans historique, cote absente. Une page blanche ressemble à une
     panne.
  3. Elle ÉCHAPPE ce qui vient de la base. Les noms d'hippodromes et de
     chevaux viennent d'une API tierce ; une apostrophe ou un chevron mal
     placés suffisent à casser le JavaScript de la page.
"""

from __future__ import annotations

import re
import sys
from datetime import date
from pathlib import Path

import pytest

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE / "src"))

from pmu import vue  # noqa: E402


def _course(**kw):
    base = {
        "code": "R1C1", "libelle": "PRIX DES AMATEURS",
        "hippodrome": "HIPPODROME DE LA CAPELLE", "discipline": "ATTELE",
        "distance": 2750, "terrain": "BON", "allocation": 22000.0,
        "partants": 2, "depart": "2026-08-25T11:52:00+00:00",
        "confiance": 0.06, "arrivee_connue": False,
        "selection": [
            {"num": 5, "cheval": "JULIE DU NORD", "driver": "F. NIVARD",
             "entraineur": None, "pere": "GOETMALS WOOD", "pere_mere": None,
             "musique": "3a 1a", "age": 6, "sexe": "FEMELLES", "corde": 1,
             "nb_courses": 24, "nb_victoires": 5, "gains": 48200.0,
             "proba": 0.22, "rang": 1, "cote": 4.2, "valeur": -0.08,
             "arrivee": None,
             "motifs": [{"groupe": "vitesse", "titre": "Chrono", "icone": "mdi:speedometer",
                         "sens": "+", "poids": 0.04, "details": ["meilleur chrono 1'12\"8"]}],
             "faits": {"vitesse": ["meilleur chrono 1'12\"8 au km"]}},
            {"num": 7, "cheval": "JAZZ DE L'ABBAYE", "driver": None,
             "entraineur": None, "pere": None, "pere_mere": None,
             "musique": "", "age": None, "sexe": None, "corde": None,
             "nb_courses": None, "nb_victoires": None, "gains": None,
             "proba": 0.11, "rang": 2, "cote": None, "valeur": None,
             "arrivee": None, "motifs": [], "faits": {}},
        ],
    }
    base.update(kw)
    return base


# ---------------------------------------------------------------------
# Mise en casse
# ---------------------------------------------------------------------

@pytest.mark.parametrize("brut, attendu", [
    ("JULIE DU NORD", "Julie du Nord"),
    ("KORONA DES CHAMPS", "Korona des Champs"),
    ("HAPPY ET FIER", "Happy et Fier"),
    ("", ""), (None, ""),
])
def test_joli(brut, attendu):
    assert vue.joli(brut) == attendu


@pytest.mark.parametrize("brut, attendu", [
    ("HIPPODROME DE LA CAPELLE", "La Capelle"),
    ("HIPPODROME DE TOULOUSE", "Toulouse"),
    ("HIPPODROME D'ENGHIEN", "Enghien"),
    ("HIPPODROME VINCENNES", "Vincennes"),
    ("VINCENNES", "Vincennes"),
    (None, ""),
])
def test_lieu(brut, attendu):
    assert vue.lieu(brut) == attendu


# ---------------------------------------------------------------------
# Autonomie
# ---------------------------------------------------------------------

def test_la_page_est_autonome():
    """
    Aucun appel réseau : ni script, ni feuille de style, ni police, ni
    image distante. Le tableau de bord doit s'afficher sans Internet.
    """
    p = vue.page([_course()], jour=date(2026, 8, 25))
    for interdit in ("<script src", "<link rel=\"stylesheet\"",
                     "https://", "http://", "@import", "url("):
        assert interdit not in p, f"ressource externe détectée : {interdit}"


def test_la_page_contient_le_necessaire():
    p = vue.page([_course()], jour=date(2026, 8, 25),
                 meta={"modele": "sans_marche", "age_heures": 0.4, "frais": True})
    assert p.startswith("<!doctype html>")
    assert "<title>Pronostics PMU — 25/08/2026</title>" in p
    assert 'lang="fr"' in p
    assert "prefers-color-scheme" in p, "pas de thème sombre"
    assert "R1C1" in p and "La Capelle" in p
    assert "Julie du Nord" in p, "les majuscules n'ont pas été adoucies"
    assert "sans_marche" in p


def test_le_html_est_equilibre():
    """Contrôle grossier mais efficace : autant d'ouvertures que de fermetures."""
    p = vue.page([_course()], jour=date(2026, 8, 25))
    for balise in ("html", "head", "body", "style", "script"):
        assert p.count(f"<{balise}") == p.count(f"</{balise}>"), f"balise {balise}"


# ---------------------------------------------------------------------
# Robustesse
# ---------------------------------------------------------------------

def test_journee_vide():
    p = vue.page([], jour=date(2026, 8, 25))
    assert "Aucun pronostic" in p
    assert "<!doctype html>" in p


def test_course_sans_selection():
    p = vue.page([_course(selection=[])], jour=date(2026, 8, 25))
    assert "R1C1" in p


def test_depart_illisible_ne_fait_pas_planter():
    p = vue.page([_course(depart="pas une date")], jour=date(2026, 8, 25))
    assert "R1C1" in p


def test_champs_absents():
    """Une base d'avant l'étape 3 ne fournit ni père, ni chrono, ni gains."""
    minimal = {"code": "R2C4", "selection": [{"num": 1, "proba": 0.3}]}
    p = vue.page([minimal], jour=date(2026, 8, 25))
    assert "R2C4" in p
    assert "None" not in p.split("<script>")[0], "un « None » a été rendu"


# ---------------------------------------------------------------------
# Échappement — la faille la plus facile à laisser passer
# ---------------------------------------------------------------------

def test_les_donnees_de_la_base_sont_echappees():
    """
    Les noms viennent d'une API tierce. Un chevron non échappé et la page
    ne s'affiche plus ; pire, elle exécute ce qu'on y a glissé.
    """
    piege = _course(
        hippodrome="<script>alert(1)</script>",
        libelle='PRIX "DU" <b>GROS</b>',
    )
    piege["selection"][0]["cheval"] = "</script><img src=x onerror=alert(1)>"
    p = vue.page([piege], jour=date(2026, 8, 25))

    # Un seul bloc script : celui de la page. Si la charge JSON n'était
    # pas échappée, le « </script> » du nom de cheval en fermerait un
    # deuxième — et tout ce qui suit deviendrait du HTML exécutable.
    assert p.count("<script>") == 1
    assert p.count("</script>") == 1
    assert "onerror=alert" not in p

    # Dans la charge embarquée, aucun chevron littéral.
    charge = p.split("const DONNEES=")[1].split(";const RAFRAICHIR=")[0]
    assert "<" not in charge and ">" not in charge
    assert "\\u003c" in charge, "l'échappement n'a pas eu lieu"

    # Et malgré l'échappement, la donnée reste intacte après analyse.
    import json
    rendu = json.loads(charge.rstrip(";"))
    assert rendu["courses"][0]["selection"][0]["cheval"].startswith("</Script>".lower()[:2]) \
        or "script" in rendu["courses"][0]["selection"][0]["cheval"].lower()


def test_apostrophe_dans_un_nom():
    """« JAZZ DE L'ABBAYE » ne doit pas casser le JavaScript."""
    p = vue.page([_course()], jour=date(2026, 8, 25))
    assert "Jazz de l'Abbaye" in p or "Jazz de l\\u0027Abbaye" in p


# ---------------------------------------------------------------------
# Fraîcheur
# ---------------------------------------------------------------------

def test_le_pronostic_perime_est_signale():
    frais = vue.page([_course()], jour=date(2026, 8, 25),
                     meta={"age_heures": 0.5, "frais": True})
    vieux = vue.page([_course()], jour=date(2026, 8, 25),
                     meta={"age_heures": 51.0, "frais": False})
    assert "puce alerte" not in frais
    assert "puce alerte" in vieux, "un pronostic de 51 h doit être signalé"


def test_le_favori_gagnant_est_marque():
    gagne = _course(arrivee_connue=True)
    gagne["selection"][0]["arrivee"] = 1
    perdu = _course(arrivee_connue=True)
    perdu["selection"][0]["arrivee"] = 6
    assert "favori gagnant" in vue.page([gagne], jour=date(2026, 8, 25))
    assert "favori battu" in vue.page([perdu], jour=date(2026, 8, 25))
