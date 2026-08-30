"""
Tests de fiabilisation multi-données.

Ils couvrent les erreurs silencieuses les plus coûteuses :
- contamination entre courses simultanées ;
- fausse lignée commune pour les parents inconnus ;
- confusion entre la carrière du cheval et celle des autres descendants ;
- perte des variables catégorielles avant apprentissage ;
- publication alors qu'aucune bande de confiance ne tient.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE / "src"))

from pmu import features as ft  # noqa: E402
from pmu.predict import _masque_publication  # noqa: E402
from pmu.train import Decoupage, ModelePmu  # noqa: E402


def _glissant(lignes: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(lignes)
    df["heure_depart"] = pd.to_datetime(df["heure_depart"], utc=True)
    df["est_exploitable"] = True
    return df


def test_deux_courses_simultanees_ne_se_voient_pas():
    t0 = pd.Timestamp("2026-01-01T12:00:00Z")
    df = _glissant([
        {"course_id": 1, "heure_depart": t0, "nom_pere": "P", "y_place": 1.0},
        {"course_id": 2, "heure_depart": t0, "nom_pere": "P", "y_place": 0.0},
        {"course_id": 3, "heure_depart": t0 + pd.Timedelta(minutes=5),
         "nom_pere": "P", "y_place": 1.0},
    ])
    _, n = ft._taux_glissant(
        df, ["nom_pere"], "y_place", prior=0.3, pseudo_n=0
    )
    assert list(n) == [0.0, 0.0, 2.0]


def test_les_parents_inconnus_ne_forment_pas_une_fausse_lignee():
    df = _glissant([
        {"course_id": 1, "heure_depart": "2026-01-01T12:00:00Z",
         "nom_pere": None, "y_place": 1.0},
        {"course_id": 2, "heure_depart": "2026-01-02T12:00:00Z",
         "nom_pere": None, "y_place": 0.0},
    ])
    taux, n = ft._taux_glissant(
        df, ["nom_pere"], "y_place", prior=0.3, pseudo_n=10
    )
    assert list(n) == [0.0, 0.0]
    assert taux.isna().all()


def test_le_signal_de_lignee_exclut_les_courses_du_cheval():
    df = _glissant([
        {"course_id": 1, "heure_depart": "2026-01-01T12:00:00Z",
         "id_cheval": "A", "nom_pere": "P", "y_place": 1.0},
        {"course_id": 2, "heure_depart": "2026-01-02T12:00:00Z",
         "id_cheval": "A", "nom_pere": "P", "y_place": 0.0},
        {"course_id": 3, "heure_depart": "2026-01-03T12:00:00Z",
         "id_cheval": "B", "nom_pere": "P", "y_place": 1.0},
    ])
    taux, n = ft._taux_lignee_hors_cheval(
        df, ["nom_pere"], "y_place", prior=0.3, pseudo_n=10, min_n=0
    )
    assert list(n) == [0.0, 0.0, 2.0]
    assert taux.iloc[2] == np.testing.assert_allclose(
        [taux.iloc[2]], [(1.0 + 3.0) / 12.0], rtol=0, atol=1e-12
    )


def test_un_produit_compte_une_fois_meme_s_il_court_souvent():
    df = _glissant([
        {"course_id": 1, "heure_depart": "2026-01-01T12:00:00Z",
         "id_cheval": "A", "nom_pere": "P", "y_gagnant": 1.0},
        {"course_id": 2, "heure_depart": "2026-01-02T12:00:00Z",
         "id_cheval": "A", "nom_pere": "P", "y_gagnant": 0.0},
        {"course_id": 3, "heure_depart": "2026-01-03T12:00:00Z",
         "id_cheval": "B", "nom_pere": "P", "y_gagnant": 0.0},
    ])
    tous = pd.Series(True, index=df.index)
    produits = ft._produits_distincts_avant(df, ["nom_pere"], tous)
    gagnants = ft._produits_distincts_avant(
        df, ["nom_pere"], df["y_gagnant"].gt(0)
    )
    assert list(produits) == [0.0, 0.0, 1.0]
    assert list(gagnants) == [0.0, 0.0, 1.0]


def _cadre_categories(n_courses: int = 180) -> pd.DataFrame:
    lignes = []
    debut = pd.Timestamp("2025-01-01T12:00:00Z")
    for course in range(n_courses):
        heure = debut + pd.Timedelta(hours=course)
        lignes.extend([
            {
                "course_id": course,
                "num_pmu": 1,
                "heure_depart": heure,
                "est_cible": True,
                "y_gagnant": 1.0,
                "c_terrain": "BON",
                "p_sexe": "FEMELLE",
            },
            {
                "course_id": course,
                "num_pmu": 2,
                "heure_depart": heure,
                "est_cible": True,
                "y_gagnant": 0.0,
                "c_terrain": "LOURD",
                "p_sexe": "MALE",
            },
        ])
    return pd.DataFrame(lignes)


def test_les_categories_sont_apprises_et_persistantes(tmp_path):
    df = _cadre_categories()
    decoupage = Decoupage.par_proportions(df["heure_depart"], 0.6, 0.2)
    modele = ModelePmu().entrainer(df, decoupage)

    assert set(modele.categories) == {"c_terrain", "p_sexe"}
    _, _, test = decoupage.masques(df["heure_depart"])
    avant = modele.predire(df[test])
    assert avant["proba"].notna().all()

    modele.sauver(tmp_path / "modele")
    relu = ModelePmu.charger(tmp_path / "modele")
    assert relu.categories == modele.categories
    apres = relu.predire(df[test])
    pd.testing.assert_series_equal(
        avant["proba"].sort_index(), apres["proba"].sort_index(),
        check_names=False,
    )

    inconnu = df[test].head(2).copy()
    inconnu["c_terrain"] = "TERRAIN_NOUVEAU"
    assert relu.predire(inconnu)["proba"].notna().all()


def test_seuil_null_signifie_abstention_et_non_publication():
    pred = pd.DataFrame({
        "ecart_top2": [0.30, 0.30],
        "qualite_course": [0.90, 0.90],
    })
    assert not _masque_publication(pred, {"seuil": None}).any()


def test_publication_exige_signal_et_completude():
    pred = pd.DataFrame({
        "ecart_top2": [0.20, 0.04, 0.20],
        "qualite_course": [0.80, 0.80, 0.30],
    })
    obtenu = _masque_publication(pred, {"seuil": 0.10}, qualite_min=0.55)
    assert list(obtenu) == [True, False, False]
