"""
Tests du parsing.

Les fixtures reproduisent la forme observée sur l'API le 23/08/2026.
Elles ne remplacent PAS une validation contre du vrai JSON : lancer
`python scripts/probe.py` pour capturer des payloads réels, puis
`pytest tests/test_contrat_api.py` pour vérifier que le contrat tient.
"""

from datetime import date, datetime, timezone

import pytest

from pmu import normalize as nz


# ---------------------------------------------------------------------
# Libellés
# ---------------------------------------------------------------------

@pytest.mark.parametrize(
    "brut, attendu",
    [
        ("M. Barzalona", "M BARZALONA"),
        ("M.BARZALONA", "M BARZALONA"),
        ("  Jean-Michel  BAZIRE ", "JEAN MICHEL BAZIRE"),
        ("Éric RAFFIN", "ERIC RAFFIN"),
        (None, None),
        ("", None),
    ],
)
def test_norm_label(brut, attendu):
    assert nz.norm_label(brut) == attendu


def test_norm_label_reconcilie_les_variantes():
    """Deux graphies du même driver doivent produire la même clé."""
    assert nz.norm_person("M. BARZALONA") == nz.norm_person("m.barzalona")


# ---------------------------------------------------------------------
# Types primitifs
# ---------------------------------------------------------------------

def test_ms_to_dt():
    dt = nz.ms_to_dt(1787515200000)
    assert dt is not None and dt.tzinfo == timezone.utc


@pytest.mark.parametrize("valeur", [None, "", 0, "abc", float("inf")])
def test_ms_to_dt_tolere_les_dechets(valeur):
    assert nz.ms_to_dt(valeur) is None


def test_cents_to_eur():
    # 1 753 600 centimes = 17 536 €
    assert nz.cents_to_eur(1753600) == 17536.0
    assert nz.cents_to_eur(None) is None


def test_dig_ne_leve_jamais():
    assert nz.dig({"a": {"b": 1}}, "a", "b") == 1
    assert nz.dig({"a": None}, "a", "b") is None
    assert nz.dig(None, "a") is None
    assert nz.dig({"a": "texte"}, "a", "b") is None


# ---------------------------------------------------------------------
# Musique
# ---------------------------------------------------------------------

def test_parse_musique_nominale():
    res = nz.parse_musique("1a 2a 0a Da 3m")
    assert [x["place"] for x in res] == [1, 2, None, None, 3]
    assert res[2]["incident"] == "NON_PLACE"
    assert res[3]["incident"] == "DISQUALIFIE"
    assert res[0]["discipline"] == "ATTELE"
    assert res[4]["discipline"] == "MONTE"


def test_parse_musique_ignore_le_millesime():
    res = nz.parse_musique("1a 2a (25) 3a")
    assert len(res) == 3


def test_parse_musique_place_zero_nest_pas_une_place():
    """'0' veut dire « au-delà de la 10e », pas « 0e »."""
    assert nz.parse_musique("0p")[0]["place"] is None


@pytest.mark.parametrize("valeur", [None, "", 123, "   ", "???"])
def test_parse_musique_tolere_les_dechets(valeur):
    assert nz.parse_musique(valeur) == []


# ---------------------------------------------------------------------
# Ordre d'arrivée
# ---------------------------------------------------------------------

def test_place_from_ordre_arrivee_avec_ex_aequo():
    ordre = [[3], [7], [1, 9]]
    assert nz._place_from_ordre_arrivee(ordre, 3) == 1
    assert nz._place_from_ordre_arrivee(ordre, 7) == 2
    # Les deux ex æquo partagent le 3e rang.
    assert nz._place_from_ordre_arrivee(ordre, 1) == 3
    assert nz._place_from_ordre_arrivee(ordre, 9) == 3
    assert nz._place_from_ordre_arrivee(ordre, 5) is None


def test_place_from_ordre_arrivee_liste_plate():
    assert nz._place_from_ordre_arrivee([3, 7, 1], 7) == 2


# ---------------------------------------------------------------------
# Participant
# ---------------------------------------------------------------------

PARTICIPANT = {
    "numPmu": 6,
    "idCheval": 8123456,
    "nom": "LE BON TEMPS ROULE",
    "age": 2,
    "sexe": "MALES",
    "race": "PUR SANG",
    "nomPere": "SIYOUNI",
    "nomMere": "WOOT WOOT",
    "nomPereMere": "DUBAWI",
    "driver": "M.BARZALONA",
    "entraineur": "A. FABRE",
    "placeCorde": 6,
    "musique": "2p 1p 4p",
    "nombreCourses": 3,
    "nombreVictoires": 1,
    "gainsParticipant": {"gainsCarriere": 1753600, "gainsAnneeEnCours": 1753600},
    "dernierRapportDirect": {
        "typePari": "SIMPLE_GAGNANT",
        "rapport": 5.5,
        "favoris": True,
        "dateRapport": 1787515200000,
        "nombreIndicateurTendance": -2,
    },
    "statut": "PARTANT",
    "robe": {"libelleCourt": "BAI"},
}


def test_parse_participant_champs_cles():
    row = nz.parse_participant(PARTICIPANT, ordre_arrivee=[[6], [3]])
    assert row["num_pmu"] == 6
    assert row["id_cheval"] == 8123456
    assert row["driver"] == "M.BARZALONA"
    assert row["nom_pere"] == "SIYOUNI"
    assert row["gains_carriere"] == 17536.0     # centimes → euros
    assert row["ordre_arrivee"] == 1            # déduit de ordreArrivee
    assert row["robe"] == "BAI"


def test_parse_participant_sur_dict_vide():
    """Une réponse dégradée ne doit pas casser la collecte."""
    row = nz.parse_participant({})
    assert row["num_pmu"] is None
    assert row["gains_carriere"] is None
    assert row["ordre_arrivee"] is None


def test_parse_participant_jockey_alias_driver():
    """Galop dit 'jockey', trot dit 'driver' — une seule colonne en base."""
    row = nz.parse_participant({"jockey": "C. DEMURO"})
    assert row["driver"] == "C. DEMURO"


def test_parse_cotes():
    releve = datetime(2026, 8, 23, 12, 0, tzinfo=timezone.utc)
    cotes = nz.parse_cotes(PARTICIPANT, releve)
    assert len(cotes) == 1
    c = cotes[0]
    assert c["type_pari"] == "SIMPLE_GAGNANT"
    assert c["rapport"] == 5.5
    assert c["favoris"] is True
    # dateRapport prime sur l'heure de notre appel
    assert c["releve_le"] != releve


def test_parse_cotes_sans_rapport():
    assert nz.parse_cotes({"numPmu": 1}, datetime.now(timezone.utc)) == []


# ---------------------------------------------------------------------
# Course
# ---------------------------------------------------------------------

def test_parse_course_terrain_galop_via_penetrometre():
    course = {
        "numReunion": 1,
        "numOrdre": 3,
        "distance": 2400,
        "discipline": "PLAT",
        "montantPrix": 5200000,
        "penetrometre": {"intitule": "BON SOUPLE", "valeurMesure": 3.4},
        "heureDepart": 1787515200000,
    }
    row = nz.parse_course(course, date(2026, 8, 23))
    assert row["etat_terrain"] == "BON SOUPLE"
    assert row["penetrometre"] == 3.4
    assert row["montant_prix"] == 52000.0
    assert row["heure_depart"] is not None


def test_parse_course_terrain_trot_champ_direct():
    row = nz.parse_course({"etatTerrain": "BON"}, date(2026, 8, 23))
    assert row["etat_terrain"] == "BON"
    assert row["penetrometre"] is None


def test_parse_course_sur_dict_vide():
    row = nz.parse_course({}, None)
    assert row["distance"] is None
    assert row["ordre_arrivee"] is None


# ---------------------------------------------------------------------
# Performances détaillées
# ---------------------------------------------------------------------

PERF = {
    "idCheval": 8123456,
    "nomCheval": "LE BON TEMPS ROULE",
    "coursesCourues": [
        {
            "date": 1785000000000,
            "hippodrome": "DEAUVILLE",
            "nomPrix": "PRIX DE LA MER",
            "discipline": "PLAT",
            "distance": 1600,
            "allocation": 2700000,
            "nbParticipants": 12,
            "place": {"place": 2, "statusArrivee": "PLACE"},
            "etatTerrain": "BON",
            "tempsDuPremier": 96400,
            "participants": [
                {"idCheval": 8123456, "nomJockey": "M.BARZALONA", "corde": 6, "poidsJockey": 56.0},
                {"idCheval": 999, "nomJockey": "AUTRE"},
            ],
        },
        {"date": None, "hippodrome": "SANS DATE"},  # doit être écartée
    ],
}


def test_parse_performances():
    lignes = nz.parse_performances(PERF)
    assert len(lignes) == 1              # la ligne sans date est écartée
    l = lignes[0]
    assert l["id_cheval"] == 8123456
    assert l["place"] == 2
    assert l["nom_jockey"] == "M.BARZALONA"   # pris sur LE bon participant
    assert l["corde"] == 6
    assert l["allocation"] == 27000.0
    assert l["date_course"] == date(2026, 7, 25)


def test_parse_performances_sans_id_cheval():
    assert nz.parse_performances({"coursesCourues": [{"date": 1785000000000}]}) == []
