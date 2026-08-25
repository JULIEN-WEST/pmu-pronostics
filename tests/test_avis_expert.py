"""
Avis d'expert (DATAHIPPIQUE, servi par l'API PMU).

CE QUE C'EST

Un classement COMPLET des partants, avec une cote probable, publié
avant la course et consultable rétroactivement. Trouvé en sondant
l'API qu'on interrogeait déjà, après que l'IFCE (commerciale, 500 à
9 000 €/an) et LeTrot (aucune API publique) se soient révélés
inaccessibles.

LES DEUX PIÈGES

  1. LA COTE EST FRACTIONNAIRE. « 3/1 » veut dire trois contre un,
     soit une cote décimale de 4,00 et une probabilité implicite de
     25 %. La lire comme 3,00 donnerait 33 % — une erreur silencieuse,
     systématique, et dans le sens qui gonfle les favoris.

  2. C'EST UN AVIS, PAS UNE MESURE. Un analyste regarde à peu près ce
     que regarde le public, et sa cote probable sert d'ancrage au
     marché lui-même. L'introduire dans le modèle `sans_marche`
     détruirait la seule propriété qui rend ce modèle intéressant :
     être indépendant du consensus.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pmu import features as ft  # noqa: E402
from pmu.normalize import parse_cote_probable, parse_pronostic_expert  # noqa: E402
from test_explain import _cadre  # noqa: E402


# ---------------------------------------------------------------------
# 1. La cote fractionnaire
# ---------------------------------------------------------------------

@pytest.mark.parametrize("brut, attendu", [
    ("3/1", 4.0), ("4/1", 5.0), ("8/1", 9.0), ("43/1", 44.0),
    ("1/2", 1.5), ("6/5", 2.2),
    ("2.5", 2.5), (5, 5.0), ("2,5", 2.5),
])
def test_la_cote_fractionnaire_devient_decimale(brut, attendu):
    assert parse_cote_probable(brut) == pytest.approx(attendu)


def test_trois_contre_un_ne_vaut_pas_trois():
    """
    LE piège. « 3/1 » = 4,00 décimal = 25 % implicite. Le lire 3,00
    donnerait 33 %, et gonflerait tous les favoris de la même façon —
    un biais systématique, jamais une erreur visible.
    """
    assert parse_cote_probable("3/1") == 4.0
    assert 1 / parse_cote_probable("3/1") == pytest.approx(0.25)


@pytest.mark.parametrize("brut", [None, "", "abc", "3/0", "0/0", {}, []])
def test_une_cote_illisible_rend_none(brut):
    assert parse_cote_probable(brut) is None


def test_une_cote_sous_la_barre_est_refusee():
    """Une cote décimale ≤ 1 est impossible : elle rendrait moins que la mise."""
    assert parse_cote_probable(0.8) is None
    assert parse_cote_probable(1) is None


# ---------------------------------------------------------------------
# 2. Le classement
# ---------------------------------------------------------------------

def _charge():
    """Forme relevée en direct sur l'API le 25/08/2026."""
    return {
        "signature": "DATAHIPPIQUE", "source": "DATAHIPPIQUE",
        "numeroReunion": 2, "numeroCourse": 1,
        "selection": [
            {"cote_prob": "3/1", "id_nav_partant": "20260825-CAP-1-3",
             "rang": 1, "num_partant": 3},
            {"cote_prob": "4/1", "id_nav_partant": "20260825-CAP-1-5",
             "rang": 2, "num_partant": 5},
            {"cote_prob": "43/1", "id_nav_partant": "20260825-CAP-1-7",
             "rang": 12, "num_partant": 7},
        ],
    }


def _cribles():
    return {"cribles": [
        {"numPmu": 3, "nom": "JOKAI DES ROIS", "partant": True,
         "commentaire": "S'il reste sage, il peut l'emporter."},
        {"numPmu": 5, "nom": "JULIE DU NORD", "partant": True,
         "commentaire": "Choix prioritaire."},
    ]}


def test_le_classement_est_complet_et_ordonne():
    lignes = parse_pronostic_expert(_charge(), _cribles())
    assert len(lignes) == 3
    assert [l["rang_expert"] for l in lignes] == [1, 2, 12]
    assert lignes[0]["cote_probable"] == 4.0
    assert lignes[2]["cote_probable"] == 44.0
    assert lignes[0]["source_expert"] == "DATAHIPPIQUE"


def test_le_crible_est_distinct_du_classement():
    """
    Être bien classé et être retenu ne sont pas la même chose : le
    crible est la sélection que l'analyste défend explicitement.
    """
    lignes = {l["num_pmu"]: l for l in parse_pronostic_expert(_charge(), _cribles())}
    assert lignes[3]["est_crible"] and lignes[5]["est_crible"]
    assert not lignes[7]["est_crible"]
    assert "sage" in lignes[3]["commentaire_expert"]
    assert lignes[7]["commentaire_expert"] is None


def test_sans_cribles_le_classement_reste_lisible():
    lignes = parse_pronostic_expert(_charge())
    assert len(lignes) == 3
    assert not any(l["est_crible"] for l in lignes)


@pytest.mark.parametrize("charge", [
    None, {}, [], "", 42, {"selection": None}, {"selection": "x"},
    {"selection": [None, 3, "x"]}, {"selection": [{"rang": 1}]},
])
def test_une_charge_deformee_ne_leve_pas(charge):
    """Une forme inattendue rend une liste vide, jamais une exception."""
    assert parse_pronostic_expert(charge) == []


def test_des_cribles_deformes_ne_cassent_rien():
    for cribles in (None, {}, {"cribles": None}, {"cribles": [None, 5]}):
        lignes = parse_pronostic_expert(_charge(), cribles)
        assert len(lignes) == 3


# ---------------------------------------------------------------------
# 3. Les features — et leur cloisonnement
# ---------------------------------------------------------------------

@pytest.fixture(scope="module")
def enrichi():
    base = _cadre(n_courses=60)
    # Un classement plausible : le rang suit le numéro, la cote suit le rang.
    base["rang_expert"] = base.groupby("course_id").cumcount() + 1
    base["cote_probable"] = 2.0 + base["rang_expert"] * 1.5
    base["est_crible"] = base["rang_expert"] <= 3
    return ft.construire(base, avec_marche=True)


def test_les_features_expertes_existent(enrichi):
    for col in ft.COLONNES_EXPERT:
        assert col in enrichi.columns, col
    assert enrichi["x_rang"].notna().all()
    assert enrichi["x_proba"].notna().all()


def test_la_probabilite_experte_somme_a_un_par_course(enrichi):
    sommes = enrichi.groupby("course_id")["x_proba"].sum()
    assert np.allclose(sommes, 1.0, atol=1e-6), (
        f"min {sommes.min():.6f}, max {sommes.max():.6f}"
    )


def test_le_rang_est_rapporte_a_la_taille_du_lot(enrichi):
    """Un 3e sur 8 n'est pas un 3e sur 18."""
    assert enrichi["x_rang_rel"].between(0, 1).all()


def test_l_avis_expert_n_entre_jamais_dans_sans_marche(enrichi):
    """
    LE cloisonnement. Le modèle `sans_marche` n'a d'intérêt que s'il est
    indépendant du consensus ; un avis d'analyste EST un consensus.
    """
    sans = ft.colonnes_features(enrichi, avec_marche=False)
    assert not [c for c in sans if c.startswith("x_")], (
        f"features expertes fuitées dans sans_marche : "
        f"{[c for c in sans if c.startswith('x_')]}"
    )
    avec = ft.colonnes_features(enrichi, avec_marche=True)
    assert len([c for c in avec if c.startswith("x_")]) >= 4


def test_l_ecart_au_marche_est_calcule(enrichi):
    """La seule feature vraiment neuve : là où public et analyste divergent."""
    assert enrichi["x_ecart_marche"].notna().any()
    # C'est une différence de deux probabilités : elle doit rester bornée.
    assert enrichi["x_ecart_marche"].abs().max() <= 1.0


def test_tout_marche_sans_avis_expert():
    """
    Une base d'avant cette version n'a aucune de ces colonnes, et
    l'analyste ne couvre pas toutes les courses. Le pipeline doit
    tourner quand même.
    """
    df = ft.construire(_cadre(n_courses=30), avec_marche=True)
    for col in ft.COLONNES_EXPERT:
        assert col in df.columns and df[col].isna().all()
    assert ft.colonnes_features(df, avec_marche=True)


def test_l_avis_expert_n_est_pas_une_cible(enrichi):
    """Ce sont des entrées, pas des résultats — mais vérifions-le."""
    cols = set(ft.colonnes_features(enrichi, avec_marche=True))
    for interdite in ("rang_expert", "cote_probable", "est_crible"):
        assert interdite not in cols, (
            f"{interdite} est la colonne BRUTE ; seule sa version x_ doit passer"
        )
