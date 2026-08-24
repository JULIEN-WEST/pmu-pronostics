"""
Tests anti-fuite.

Ce fichier est le garde-fou le plus important du dépôt. Si l'un de ces
tests casse, aucune métrique produite par le projet n'a de valeur.

Deux niveaux :
  1. `_taux_glissant` calcule-t-il bien sur les COURSES ANTÉRIEURES seules ?
  2. Le canari : cible purement aléatoire → un modèle honnête doit faire
     du 0,50 d'AUC. Au-delà, c'est qu'il lit le résultat quelque part.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from pmu import features as ft


# ---------------------------------------------------------------------
# 1. Le calcul glissant
# ---------------------------------------------------------------------

def _cadre(lignes: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(lignes)
    df["est_exploitable"] = True
    return df


def test_taux_glissant_ignore_la_course_courante():
    """
    Deux demi-frères dans la même course : le second ne doit RIEN savoir
    du résultat du premier. C'est le piège classique des features de lignée.
    """
    df = _cadre([
        # Course 1 — deux produits du même père
        {"course_id": 1, "nom_pere": "GOETMALS", "y_place": 1.0},
        {"course_id": 1, "nom_pere": "GOETMALS", "y_place": 0.0},
        # Course 2 — un troisième produit
        {"course_id": 2, "nom_pere": "GOETMALS", "y_place": 1.0},
    ])
    _, effectif = ft._taux_glissant(df, ["nom_pere"], "y_place", prior=0.3, pseudo_n=0)

    # Les deux lignes de la course 1 ne voient aucune course antérieure.
    assert effectif.iloc[0] == 0
    assert effectif.iloc[1] == 0, "le 2e partant voit son demi-frère de la même course"
    # La course 2 voit les deux partants de la course 1.
    assert effectif.iloc[2] == 2


def test_taux_glissant_valeur_exacte():
    df = _cadre([
        {"course_id": 1, "id_cheval": 7, "y_place": 1.0},
        {"course_id": 2, "id_cheval": 7, "y_place": 0.0},
        {"course_id": 3, "id_cheval": 7, "y_place": 1.0},
    ])
    taux, effectif = ft._taux_glissant(df, ["id_cheval"], "y_place", prior=0.0, pseudo_n=0)
    assert list(effectif) == [0.0, 1.0, 2.0]
    # Sans lissage : 0/0 → nan, 1/1 = 1, 1/2 = 0,5
    assert np.isnan(taux.iloc[0]) or taux.iloc[0] == 0
    assert taux.iloc[1] == pytest.approx(1.0)
    assert taux.iloc[2] == pytest.approx(0.5)


def test_taux_glissant_exclut_les_non_partants():
    df = _cadre([
        {"course_id": 1, "id_cheval": 7, "y_place": 0.0},
        {"course_id": 2, "id_cheval": 7, "y_place": 1.0},
        {"course_id": 3, "id_cheval": 7, "y_place": 1.0},
    ])
    df.loc[0, "est_exploitable"] = False   # non-partant : ne compte pas
    _, effectif = ft._taux_glissant(df, ["id_cheval"], "y_place", prior=0.0, pseudo_n=0)
    assert list(effectif) == [0.0, 0.0, 1.0]


def test_lissage_tire_vers_le_prior():
    """1 victoire sur 1 course ne fait pas un cheval à 100 %."""
    df = _cadre([
        {"course_id": 1, "id_cheval": 7, "y_place": 1.0},
        {"course_id": 2, "id_cheval": 7, "y_place": 0.0},
    ])
    taux, _ = ft._taux_glissant(df, ["id_cheval"], "y_place", prior=0.3, pseudo_n=10)
    # (1 + 0,3×10) / (1 + 10) = 0,3636 — loin de 1,0
    assert taux.iloc[1] == pytest.approx(4.0 / 11.0, abs=1e-6)


# ---------------------------------------------------------------------
# 2. Le canari
# ---------------------------------------------------------------------

def _dataset_synthetique(n_courses: int = 400, graine: int = 7) -> pd.DataFrame:
    """
    Un univers de courses crédible dans sa STRUCTURE (chevaux récurrents,
    drivers récurrents, lignées), mais dont le RÉSULTAT est tiré au hasard.

    Aucune feature ne peut donc être légitimement prédictive. Tout ce qui
    dépasse l'aléatoire est une fuite.
    """
    rng = np.random.default_rng(graine)
    chevaux = np.arange(1, 601)
    drivers = np.arange(1, 81)
    entraineurs = np.arange(1, 41)
    peres = [f"PERE_{i}" for i in range(30)]
    peres_mere = [f"PM_{i}" for i in range(25)]
    hippos = ["VIN", "ENG", "CAG", "DEA", "CHA"]
    terrains = ["BON", "SOUPLE", "COLLANT", "LOURD"]
    disciplines = ["ATTELE", "PLAT", "MONTE"]

    # Chaque cheval garde son père et son père de mère d'une course à l'autre.
    pere_de = {c: rng.choice(peres) for c in chevaux}
    pm_de = {c: rng.choice(peres_mere) for c in chevaux}

    lignes = []
    t0 = pd.Timestamp("2024-01-01", tz="UTC")
    for i in range(n_courses):
        n = int(rng.integers(8, 17))
        partants = rng.choice(chevaux, size=n, replace=False)
        heure = t0 + pd.Timedelta(hours=6 * i)
        gagnant = rng.integers(0, n)                 # ← purement aléatoire
        ordre = rng.permutation(np.arange(1, n + 1))
        ordre[gagnant] = 1
        ordre[np.where(ordre == 1)[0][0] if False else gagnant] = 1

        for j, ch in enumerate(partants):
            lignes.append({
                "course_id": i,
                "heure_depart": heure,
                "num_pmu": j + 1,
                "id_cheval": int(ch),
                "id_driver": int(rng.choice(drivers)),
                "id_entraineur": int(rng.choice(entraineurs)),
                "nom_pere": pere_de[ch],
                "nom_pere_mere": pm_de[ch],
                "discipline": rng.choice(disciplines),
                "specialite": None,
                "distance": int(rng.choice([1600, 2100, 2400, 2850, 3000])),
                "etat_terrain": rng.choice(terrains),
                "hippodrome_code": rng.choice(hippos),
                "nombre_partants": n,
                "montant_prix": float(rng.integers(15000, 90000)),
                "age": int(rng.integers(3, 11)),
                "sexe": rng.choice(["MALES", "FEMELLES", "HONGRES"]),
                "place_corde": j + 1,
                "handicap_poids": float(rng.integers(50, 62)),
                "deferre": rng.choice([None, "DEFERRE_ANTERIEURS", "DEFERRE_QUATRE_PIEDS"]),
                "oeilleres": rng.choice([None, "OEILLERES_CLASSIQUES"]),
                "musique": " ".join(rng.choice(["1a", "2a", "3a", "0a", "Da"], size=5)),
                "nombre_courses": int(rng.integers(1, 60)),
                "nombre_victoires": int(rng.integers(0, 12)),
                "nombre_places": int(rng.integers(0, 25)),
                "gains_carriere": float(rng.integers(0, 250000)),
                "gains_annee_en_cours": float(rng.integers(0, 60000)),
                "statut": "PARTANT",
                # LA cible, tirée au hasard et indépendante de tout le reste
                "ordre_arrivee": int(1 if j == gagnant else rng.integers(2, n + 1)),
            })
    return pd.DataFrame(lignes)


@pytest.fixture(scope="module")
def enrichi() -> pd.DataFrame:
    return ft.construire(_dataset_synthetique(), avec_marche=False)


def test_canari_aucune_feature_ne_predit_le_hasard(enrichi):
    """
    Le test central. Cible aléatoire → AUC attendue 0,50.

    Le seuil est fixé à 0,56 : au-delà, sur 400 courses, ce n'est plus
    de la fluctuation d'échantillonnage mais une vraie fuite.
    """
    df = enrichi[enrichi["est_exploitable"]].copy()
    cols = ft.colonnes_features(df, avec_marche=False)
    X = df[cols].apply(pd.to_numeric, errors="coerce")
    y = df["y_gagnant"]

    # Découpage chronologique, comme en production.
    coupe = int(len(df) * 0.7)
    modele = HistGradientBoostingClassifier(max_iter=120, random_state=0)
    modele.fit(X.iloc[:coupe], y.iloc[:coupe])
    auc = roc_auc_score(y.iloc[coupe:], modele.predict_proba(X.iloc[coupe:])[:, 1])

    assert auc < 0.56, (
        f"AUC = {auc:.3f} sur une cible aléatoire : une feature lit le résultat. "
        "Vérifier _taux_glissant et colonnes_features."
    )


def test_colonnes_features_refuse_les_colonnes_de_resultat(enrichi):
    """Le filet de sécurité doit se déclencher si quelqu'un renomme mal."""
    df = enrichi.copy()
    df["p_ordre_arrivee_deguise"] = df["ordre_arrivee"]
    cols = ft.colonnes_features(df)
    # La colonne déguisée passe (on ne peut pas deviner), mais les vraies
    # colonnes de résultat, elles, ne doivent jamais être là.
    assert "ordre_arrivee" not in cols
    assert "y_gagnant" not in cols
    assert "y_place" not in cols


def test_cible_place_suit_la_regle_pmu(enrichi):
    """3 places payées à partir de 8 partants, 2 en dessous."""
    petit = enrichi[enrichi["c_nb_partants"] < 8]
    if len(petit):
        assert petit.loc[petit["ordre_arrivee"] == 3, "y_place"].eq(0).all()
    grand = enrichi[enrichi["c_nb_partants"] >= 8]
    assert grand.loc[grand["ordre_arrivee"] == 3, "y_place"].eq(1).all()


def test_jours_repos_est_positif(enrichi):
    repos = enrichi["p_jours_repos"].dropna()
    assert (repos >= 0).all(), "un cheval ne court pas avant sa course précédente"
