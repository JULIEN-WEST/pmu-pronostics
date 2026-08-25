"""
Météo par hippodrome.

CE QUE CES TESTS PROTÈGENT

La forme de la réponse de l'API d'archive n'a PAS pu être confrontée en
direct au moment d'écrire le module : ce service refuse les robots. Elle
vient donc de sa documentation. Or ce projet a déjà perdu trois journées
de collecte sur des schémas devinés — identifiant rendu en chaîne au
lieu d'un entier, champs texte rendus en objets, identifiant absent d'un
bloc — et chaque fois l'erreur était silencieuse.

D'où le parti pris : `resumer()` ne fait AUCUNE confiance à ce qu'elle
reçoit. Ces tests lui envoient des réponses déformées de toutes les
façons imaginables et vérifient qu'elle rend None au lieu de lever. Une
météo manquante coûte quelques features vides ; une exception ferait
tomber la collecte du jour.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pytest

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE / "src"))

from pmu import meteo as mto  # noqa: E402


# ---------------------------------------------------------------------
# Le nom de la commune
# ---------------------------------------------------------------------

@pytest.mark.parametrize("libelle, attendu_premier", [
    ("HIPPODROME DE LA CAPELLE", "CAPELLE"),
    ("HIPPODROME DE VINCENNES", "VINCENNES"),
    ("HIPPODROME D'ENGHIEN", "ENGHIEN"),
    ("HIPPODROME DU LION D'ANGERS", "LION D'ANGERS"),
    ("HIPPODROME DES SABLES", "SABLES"),
    ("VINCENNES", "VINCENNES"),
])
def test_le_prefixe_hippodrome_est_retire(libelle, attendu_premier):
    assert mto.candidats(libelle)[0] == attendu_premier


def test_les_noms_composes_sont_degrades_par_etapes():
    """
    « PARIS-VINCENNES » : le lieu réel est le second terme, mais on
    essaie les deux — le géocodeur tranchera.
    """
    c = mto.candidats("HIPPODROME DE PARIS-VINCENNES")
    assert "VINCENNES" in c and "PARIS" in c
    assert c.index("VINCENNES") < c.index("PARIS"), (
        "le second terme doit être essayé en premier : c'est lui le lieu"
    )


def test_les_candidats_sont_uniques_et_non_vides():
    for libelle in ("HIPPODROME DE CAGNES-SUR-MER MIDI", "HIPPODROME DE VICHY",
                    "HIPPODROME", "  "):
        c = mto.candidats(libelle)
        assert len(c) == len(set(c))
        assert all(x.strip() for x in c)


def test_un_libelle_vide_ne_produit_rien():
    assert mto.candidats("") == []
    assert mto.candidats(None) == []


# ---------------------------------------------------------------------
# L'extraction — le cœur défensif
# ---------------------------------------------------------------------

def _reponse(jour: date, pluie_veille=1.0, pluie_jour=0.5):
    """Réponse conforme à la documentation d'Open-Meteo."""
    veille = (jour - __import__("datetime").timedelta(days=1)).isoformat()
    j = jour.isoformat()
    temps, temp, pluie, vent, humid = [], [], [], [], []
    for d, p in ((veille, pluie_veille), (j, pluie_jour)):
        for h in range(24):
            temps.append(f"{d}T{h:02d}:00")
            temp.append(15.0 + h * 0.1)
            pluie.append(p)
            vent.append(10.0 + h)
            humid.append(70.0)
    return {
        "latitude": 49.98, "longitude": 3.92, "timezone": "Europe/Paris",
        "hourly_units": {"temperature_2m": "°C", "precipitation": "mm"},
        "hourly": {"time": temps, "temperature_2m": temp,
                   "precipitation": pluie, "wind_speed_10m": vent,
                   "relative_humidity_2m": humid},
    }


def test_le_resume_calcule_les_bons_cumuls():
    jour = date(2025, 6, 15)
    r = mto.resumer(_reponse(jour, pluie_veille=1.0, pluie_jour=0.5), jour)
    assert r is not None
    # 24 h de veille à 1 mm, plus les heures du jour avant 8 h (0 à 7) à
    # 0,5 mm : 24 + 8 × 0,5 = 28 mm.
    assert r["pluie_24h"] == pytest.approx(28.0)
    # Journée de course : 8 h à 20 h inclus, soit 13 heures à 0,5 mm.
    assert r["pluie_jour"] == pytest.approx(6.5)
    assert r["vent_max"] == pytest.approx(30.0)   # heure 20 de la journée
    assert 15.0 < r["temperature"] < 17.0
    assert r["humidite"] == pytest.approx(70.0)


def test_le_resume_separe_bien_la_veille_du_jour():
    jour = date(2025, 6, 15)
    sec = mto.resumer(_reponse(jour, pluie_veille=0.0, pluie_jour=0.0), jour)
    deluge = mto.resumer(_reponse(jour, pluie_veille=4.0, pluie_jour=0.0), jour)
    assert sec["pluie_24h"] == 0.0
    assert deluge["pluie_24h"] == pytest.approx(96.0)
    assert deluge["pluie_jour"] == 0.0, "la pluie de la veille a débordé sur le jour"


@pytest.mark.parametrize("charge", [
    None, [], "", 42,
    {},                                        # pas de bloc horaire
    {"hourly": None},
    {"hourly": {}},                            # bloc vide
    {"hourly": {"time": []}},                  # aucune heure
    {"hourly": {"time": "2025-06-15T00:00"}},  # chaîne au lieu d'une liste
    {"error": True, "reason": "quota dépassé"},
])
def test_une_reponse_deformee_rend_none_sans_lever(charge):
    """
    LE test qui compte. Toute forme inattendue doit rendre None — jamais
    une exception, qui ferait tomber la collecte entière.
    """
    assert mto.resumer(charge, date(2025, 6, 15)) is None


def test_une_variable_manquante_ne_casse_rien():
    """L'API peut ne pas rendre toutes les variables demandées."""
    jour = date(2025, 6, 15)
    charge = _reponse(jour)
    del charge["hourly"]["wind_speed_10m"]
    r = mto.resumer(charge, jour)
    assert r is not None
    assert r["vent_max"] is None
    assert r["pluie_24h"] is not None


def test_des_longueurs_incoherentes_sont_ignorees():
    """Une colonne plus courte que `time` est écartée, pas alignée de force."""
    jour = date(2025, 6, 15)
    charge = _reponse(jour)
    charge["hourly"]["precipitation"] = [1.0, 2.0]
    r = mto.resumer(charge, jour)
    assert r is not None
    assert r["pluie_24h"] is None and r["pluie_jour"] is None
    assert r["temperature"] is not None


def test_des_valeurs_nulles_ou_textuelles_sont_ecartees():
    jour = date(2025, 6, 15)
    charge = _reponse(jour)
    charge["hourly"]["precipitation"] = [None] * 48
    charge["hourly"]["temperature_2m"] = ["chaud"] * 48
    r = mto.resumer(charge, jour)
    assert r is not None
    assert r["pluie_24h"] is None and r["temperature"] is None


def test_des_horodatages_illisibles_sont_sautes():
    jour = date(2025, 6, 15)
    charge = _reponse(jour)
    charge["hourly"]["time"][0] = "pas une date"
    charge["hourly"]["time"][1] = "2025-06-14Txx:00"
    r = mto.resumer(charge, jour)
    assert r is not None, "deux horodatages abîmés ne doivent pas tout invalider"


def test_un_jour_absent_de_la_reponse():
    """La prévision peut ne pas couvrir la date demandée."""
    assert mto.resumer(_reponse(date(2025, 6, 15)), date(2030, 1, 1)) is None


# ---------------------------------------------------------------------
# Le client, sans réseau
# ---------------------------------------------------------------------

class _FausseSession:
    """Session HTTP factice : aucun appel réseau dans les tests."""

    def __init__(self, reponses):
        self.reponses = reponses
        self.appels = []

    def get(self, url, params=None, timeout=None):
        self.appels.append((url, params or {}))
        charge = self.reponses.get(url)
        if isinstance(charge, Exception):
            raise charge

        class R:
            @staticmethod
            def raise_for_status():
                pass

            @staticmethod
            def json():
                return charge
        return R()


def test_le_geocodage_prend_le_premier_resultat():
    s = _FausseSession({mto.GEOCODAGE: {"results": [
        {"name": "La Capelle", "latitude": 49.97752, "longitude": 3.91792}]}})
    c = mto.ClientMeteo(session=s)
    assert c.geocoder("HIPPODROME DE LA CAPELLE") == (49.97752, 3.91792)
    assert s.appels[0][1]["country"] == "FR", "la recherche doit être bornée à la France"


def test_le_geocodage_essaie_les_candidats_successifs():
    """Un lieu introuvable sous son nom complet peut l'être en le dégradant."""
    appels = {"n": 0}

    class S(_FausseSession):
        def get(self, url, params=None, timeout=None):
            appels["n"] += 1
            trouve = params.get("name") == "VINCENNES"
            charge = {"results": [{"latitude": 48.8, "longitude": 2.4}]} if trouve else {}
            return type("R", (), {"raise_for_status": staticmethod(lambda: None),
                                  "json": staticmethod(lambda: charge)})()

    c = mto.ClientMeteo(session=S({}))
    assert c.geocoder("HIPPODROME DE PARIS-VINCENNES") == (48.8, 2.4)
    assert appels["n"] >= 2


def test_le_geocodage_rend_none_quand_rien_ne_marche():
    c = mto.ClientMeteo(session=_FausseSession({mto.GEOCODAGE: {"results": []}}))
    assert c.geocoder("HIPPODROME DE NULLE PART") is None


def test_une_panne_reseau_ne_leve_pas():
    import requests
    s = _FausseSession({mto.GEOCODAGE: requests.ConnectionError("réseau coupé")})
    assert mto.ClientMeteo(session=s).geocoder("VINCENNES") is None


def test_le_choix_du_point_d_entree_depend_de_la_date():
    """
    L'archive accuse plusieurs jours de retard, la prévision ne remonte
    que de quelques jours. Se tromper de point d'entrée rend une réponse
    vide, silencieusement.
    """
    jour = date(2025, 6, 15)
    s = _FausseSession({mto.ARCHIVE: _reponse(jour), mto.PREVISION: _reponse(jour)})
    c = mto.ClientMeteo(session=s)

    c.journee(49.9, 3.9, jour, aujourdhui=jour)             # le jour même
    assert s.appels[-1][0] == mto.PREVISION
    c.journee(49.9, 3.9, jour, aujourdhui=date(2025, 7, 30))  # bien plus tard
    assert s.appels[-1][0] == mto.ARCHIVE


def test_la_journee_rend_la_source():
    jour = date(2025, 6, 15)
    c = mto.ClientMeteo(session=_FausseSession({mto.ARCHIVE: _reponse(jour)}))
    r = c.journee(49.9, 3.9, jour, aujourdhui=date(2025, 7, 30))
    assert r["source"] == "archive"


def test_une_erreur_http_rend_none():
    import requests
    jour = date(2025, 6, 15)
    c = mto.ClientMeteo(session=_FausseSession(
        {mto.ARCHIVE: requests.HTTPError("503")}))
    assert c.journee(49.9, 3.9, jour, aujourdhui=date(2025, 7, 30)) is None
