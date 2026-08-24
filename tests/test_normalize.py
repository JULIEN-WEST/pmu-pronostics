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

# Reproduit fidèlement la réponse observée sur
# /programme/22082026/R1/C1/participants — y compris les champs qui
# ressemblent à du texte mais sont des OBJETS.
PARTICIPANT = {
    "numPmu": 1,
    # ⚠️ Chaîne composée nom-mère-père, pas un entier.
    "idCheval": "KHAMEPHIS GAME-AKITA-ZARAK",
    "nom": "KHAMEPHIS GAME",
    "age": 6,
    "sexe": "HONGRES",
    "race": "PUR-SANG",
    "pays": "France",
    "nomPere": "ZARAK",
    "nomMere": "AKITA",
    "nomPereMere": "GIANT'S CAUSEWAY",
    "eleveur": "ECURIE HARAS DU CHATEAU",
    "proprietaire": "ECURIE HARAS DU CHATEAU",
    "driver": "M. PROTTI",
    "entraineur": "M.BRASME (S)",
    "driverChange": True,
    "placeCorde": 6,
    "handicapPoids": 595,
    "handicapValeur": 32.0,
    "oeilleres": "SANS_OEILLERES",
    "musique": "0p1p0p(25)4p1p1p3p9p0p",
    "nombreCourses": 24,
    "nombreVictoires": 4,
    "nombrePlaces": 12,
    "gainsParticipant": {"gainsCarriere": 1753600, "gainsAnneeEnCours": 1753600},
    "dernierRapportDirect": {
        "typePari": "SIMPLE_GAGNANT",
        "rapport": 12.0,
        "favoris": True,
        "dateRapport": 1787515200000,
        "nombreIndicateurTendance": -2,
    },
    "statut": "PARTANT",
    "ordreArrivee": 2,
    "allure": "GALOP",
    # Les trois pièges : ce sont des objets.
    "robe": {"code": "001", "libelleCourt": "ALEZAN", "libelleLong": "ALEZAN"},
    "commentaireApresCourse": {"texte": "A pris un bon départ.", "source": "PMU"},
    "distanceChevalPrecedent": {"libelleCourt": "1/2 L", "libelleLong": "une demi-longueur",
                                "code": 3, "identifiant": "DEMI_LONGUEUR"},
}


def test_parse_participant_champs_cles():
    row = nz.parse_participant(PARTICIPANT)
    assert row["num_pmu"] == 1
    assert row["driver"] == "M. PROTTI"
    assert row["nom_pere"] == "ZARAK"
    assert row["gains_carriere"] == 17536.0     # centimes → euros
    assert row["ordre_arrivee"] == 2
    assert row["robe"] == "ALEZAN"


def test_id_cheval_reste_une_chaine():
    """
    Le PMU compose l'identifiant à partir du nom, de la mère et du père.
    Le convertir en entier donne None : plus aucun cheval en base, plus
    aucune généalogie, plus aucune performance importée — et pas la
    moindre erreur pour le signaler.
    """
    row = nz.parse_participant(PARTICIPANT)
    assert row["id_cheval"] == "KHAMEPHIS GAME-AKITA-ZARAK"
    assert isinstance(row["id_cheval"], str)


@pytest.mark.parametrize("champ, attendu", [
    ("commentaire_apres_course", "A pris un bon départ."),
    ("distance_cheval_precedent", "une demi-longueur"),
    ("robe", "ALEZAN"),
])
def test_les_champs_objets_sont_aplatis(champ, attendu):
    """
    Passer un dict à psycopg pour une colonne `text` lève « cannot adapt
    type dict ». L'exception remontant au milieu d'une transaction, c'est
    toute la journée de collecte qui est perdue, pas la seule ligne.
    """
    row = nz.parse_participant(PARTICIPANT)
    assert row[champ] == attendu
    assert isinstance(row[champ], str)


def test_aucun_champ_n_est_un_conteneur():
    """Filet général : rien de ce qui part en base ne doit être dict/list."""
    row = nz.parse_participant(PARTICIPANT)
    fautifs = {k: type(v).__name__ for k, v in row.items()
               if isinstance(v, (dict, list, tuple, set))}
    assert not fautifs, f"champs non aplatis : {fautifs}"


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
    assert c["rapport"] == 12.0
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

# Forme réelle de /performances-detaillees/pretty : les blocs sont
# identifiés par numPmu et nomCheval — il n'y a PAS d'idCheval.
PERF = {
    "numPmu": 1,
    "nomCheval": "KHAMEPHIS GAME",
    "coursesCourues": [
        {
            "date": 1785000000000,
            "hippodrome": "DEAUVILLE",
            "nomPrix": "PRIX DE LA MER",
            "discipline": "PLAT",
            "distance": 1600,
            "allocation": 2700000,
            "nbParticipants": 12,
            "place": {"place": 2, "rawValue": "2", "statusArrivee": "PLACE"},
            "etatTerrain": "BON",
            "tempsDuPremier": 96400,
            "participants": [
                {"numPmu": 4, "nomCheval": "AUTRE", "nomJockey": "AUTRE", "itsHim": False},
                {"numPmu": 1, "nomCheval": "KHAMEPHIS GAME", "nomJockey": "M. PROTTI",
                 "corde": 6, "poidsJockey": 56.0, "itsHim": True,
                 "distanceAvecPrecedent": {"libelleLong": "une encolure"}},
            ],
        },
        {"date": None, "hippodrome": "SANS DATE"},  # doit être écartée
    ],
}

ID = "KHAMEPHIS GAME-AKITA-ZARAK"


def test_parse_performances():
    lignes = nz.parse_performances(PERF, ID)
    assert len(lignes) == 1              # la ligne sans date est écartée
    l = lignes[0]
    assert l["id_cheval"] == ID
    assert l["place"] == 2
    assert l["nom_jockey"] == "M. PROTTI"     # pris sur LE bon participant
    assert l["corde"] == 6
    assert l["allocation"] == 27000.0
    assert l["date_course"] == date(2026, 7, 25)
    assert l["distance_avec_precedent"] == "une encolure"


def test_parse_performances_identifie_le_cheval_par_itshim():
    """
    Aucun identifiant dans `coursesCourues[].participants[]` : le seul
    repère fiable est le drapeau `itsHim`. S'en remettre à l'ordre des
    éléments rattacherait les performances au mauvais cheval.
    """
    lignes = nz.parse_performances(PERF, ID)
    assert lignes[0]["nom_jockey"] != "AUTRE"


def test_parse_performances_sans_identifiant_fourni():
    """
    Le bloc ne porte pas d'idCheval : sans identifiant passé par
    l'appelant, on ne produit rien plutôt que des lignes orphelines.
    """
    assert nz.parse_performances(PERF) == []
