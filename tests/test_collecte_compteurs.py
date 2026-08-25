"""
Les compteurs de collecte — le bug qui a arrêté la production.

CE QUI S'EST PASSÉ

La 1.3 a ajouté l'avis d'expert. `collecte_course` s'est mise à
renvoyer une clé de plus (`expert`) dans son dictionnaire de comptes.
`collecte_jour` additionne ces comptes dans un dictionnaire qu'elle
initialise elle-même, et cette liste-là n'a pas été mise à jour :

    for k, v in s.items():
        total[k] += v        →  KeyError: 'expert'

Résultat : CHAQUE tour de collecte levait, le planificateur attrapait
(« tour en échec, on continue »), et la collecte n'a plus rien
enregistré pendant des heures. Aucun test ne couvrait cette boucle —
les tests portaient sur la lecture de l'avis expert, jamais sur son
intégration dans le comptage.

CE QUE CES TESTS GARANTISSENT

  1. La collecte d'une journée aboutit et compte l'avis expert.
  2. Une clé INCONNUE ne fait plus planter la boucle. C'est le vrai
     correctif : ce bug se reproduira à chaque nouvelle statistique
     tant que deux dictionnaires devront être tenus synchronisés à la
     main. On ne teste pas « expert est dans la liste », on teste
     « n'importe quelle clé passe ».
  3. Le backfill additionne de la même façon — il avait le même défaut,
     latent, sur `ignores`.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pytest

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE / "src"))

from pmu import collect  # noqa: E402


JOUR = date(2026, 8, 25)


# ---------------------------------------------------------------------
# Doublures minimales — ni base ni réseau
# ---------------------------------------------------------------------

class FauxClient:
    """Un programme d'une réunion et deux courses."""

    def fmt_date(self, jour):
        return jour.strftime("%d%m%Y")

    def programme(self, jour, use_cache=False):
        return {"reunions": [{
            "numOfficiel": 1,
            "hippodrome": {"code": "CAP", "libelleLong": "LA CAPELLE"},
            "courses": [
                {"numReunion": 1, "numOrdre": 1, "libelle": "PRIX A",
                 "discipline": "ATTELE", "distance": 2750},
                {"numReunion": 1, "numOrdre": 2, "libelle": "PRIX B",
                 "discipline": "ATTELE", "distance": 2750},
            ],
        }]}


class FausseConn:
    def commit(self):
        pass

    def rollback(self):
        pass


@pytest.fixture
def sans_base(monkeypatch):
    """Neutralise tout ce qui touche la base : seul le comptage nous intéresse."""
    faux = {
        "upsert_hippodrome": None, "upsert_reunion": None,
        "journal": None, "deja_collecte": False,
    }
    for nom, retour in faux.items():
        monkeypatch.setattr(collect.db, nom, lambda *a, _r=retour, **k: _r)
    compteur = {"n": 0}

    def _course(conn, c):
        compteur["n"] += 1
        return compteur["n"]

    monkeypatch.setattr(collect.db, "upsert_course", _course)
    monkeypatch.setattr(collect.db, "link_genealogie", lambda *a, **k: None)
    return compteur


def _stats(**kw):
    base = {"partants": 8, "cotes": 16, "perfs": 40, "ignores": 0, "expert": 8}
    base.update(kw)
    return base


# ---------------------------------------------------------------------
# 1. Le cas qui plantait
# ---------------------------------------------------------------------

def test_la_collecte_d_une_journee_aboutit(monkeypatch, sans_base):
    """
    Le test le plus bête du fichier, et celui qui aurait évité des heures
    de collecte perdues : la journée se collecte sans lever.
    """
    monkeypatch.setattr(collect, "collecte_course",
                        lambda *a, **k: _stats())
    total = collect.collecte_jour(FausseConn(), FauxClient(), JOUR)
    assert total["courses"] == 2
    assert total["partants"] == 16
    assert total["expert"] == 16, "l'avis expert doit être compté, pas ignoré"


def test_une_cle_inconnue_ne_fait_pas_planter(monkeypatch, sans_base):
    """
    LE correctif. Demain une statistique de plus sera ajoutée à
    `collecte_course` ; elle ne doit pas pouvoir arrêter la production.
    """
    monkeypatch.setattr(collect, "collecte_course",
                        lambda *a, **k: _stats(chronos=3, une_nouveaute=1))
    total = collect.collecte_jour(FausseConn(), FauxClient(), JOUR)
    assert total["chronos"] == 6
    assert total["une_nouveaute"] == 2


def test_les_compteurs_annonces_existent_toujours(monkeypatch, sans_base):
    """
    La ligne de journal de fin lit `reunions`, `courses`, `partants`,
    `perfs` et `ignores`. Une journée SANS aucune course doit quand même
    les fournir, sinon c'est le message d'erreur qui plante.
    """
    monkeypatch.setattr(collect, "collecte_course", lambda *a, **k: {})
    total = collect.collecte_jour(FausseConn(), FauxClient(), JOUR)
    for cle in ("reunions", "courses", "partants", "cotes", "perfs", "ignores"):
        assert cle in total, cle


def test_un_programme_absent_rend_des_compteurs_complets(monkeypatch, sans_base):
    class Vide(FauxClient):
        def programme(self, jour, use_cache=False):
            raise collect.PmuNotFound("404")

    total = collect.collecte_jour(FausseConn(), Vide(), JOUR)
    assert total["courses"] == 0
    for cle in ("reunions", "partants", "perfs", "ignores"):
        assert cle in total


# ---------------------------------------------------------------------
# 2. Le même défaut, latent, dans le backfill
# ---------------------------------------------------------------------

def test_le_backfill_additionne_les_memes_cles(monkeypatch, sans_base):
    """
    `backfill` tenait sa PROPRE liste de compteurs, sans `ignores` ni
    `expert`. Le jour où un partant serait ignoré, le rattrapage
    d'historique aurait planté de la même façon.
    """
    monkeypatch.setattr(collect, "collecte_course",
                        lambda *a, **k: _stats(ignores=2))
    cumul = collect.backfill(FausseConn(), FauxClient(),
                             depuis=JOUR, jusqua=JOUR)
    assert cumul["jours"] == 1
    assert cumul["ignores"] == 4
    assert cumul["expert"] == 16


def test_le_backfill_survit_a_une_nouvelle_statistique(monkeypatch, sans_base):
    monkeypatch.setattr(collect, "collecte_course",
                        lambda *a, **k: _stats(inedit=5))
    cumul = collect.backfill(FausseConn(), FauxClient(),
                             depuis=JOUR, jusqua=JOUR)
    assert cumul["inedit"] == 10
