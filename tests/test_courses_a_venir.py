"""
Pronostiquer une course qui n'a pas encore eu lieu.

LE BUG, ET POURQUOI IL A TENU DES SEMAINES

`pronostiquer()` filtrait les partants du jour sur `est_cible`. Or
`est_cible` est construit sur `est_exploitable`, qui exige
`place.notna()` — un résultat connu. Conséquence :

    AUCUNE course à venir ne passait le filtre.

Le système ne pronostiquait donc que des courses DÉJÀ COURUES, ce qui
est l'exact contraire de sa raison d'être.

Il est resté invisible parce que le symptôme ressemblait à un
fonctionnement normal : la table `pronostic` se remplissait au fil de
la soirée, à mesure que les arrivées tombaient. Le tableau de bord du
soir paraissait juste — dix courses, dix arrivées, trois favoris
gagnants — et la vue vide du matin passait pour « la collecte n'a pas
encore tourné ».

Trace dans les journaux de production, le 26/08 à 08:02, après une
collecte pourtant réussie de 449 partants sur 42 courses :

    WARNING pmu.predict | aucun partant le 2026-08-26

LE CORRECTIF

Deux notions séparées, parce que ce sont deux besoins différents :

  est_cible          direct + résultat connu   → ENTRAÎNEMENT
  est_pronosticable  direct + partant          → PRÉDICTION

Les tests ci-dessous verrouillent la distinction dans les deux sens :
une course à venir doit être pronosticable sans être une cible, et une
performance importée ne doit être ni l'une ni l'autre.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE / "src"))

from pmu import features as ft  # noqa: E402


# Les colonnes que `construire()` exige. Regroupées ici plutôt que
# recopiées quatre fois : ce fichier parle des DRAPEAUX, pas du
# remplissage d'un cadre.
COMMUN = {
    "discipline": "ATTELE", "specialite": None, "distance": 2700,
    "etat_terrain": "BON", "hippodrome_code": "VIN", "montant_prix": 20000.0,
    "id_driver": 1, "id_entraineur": 2, "id_proprietaire": 3,
    "nom_pere": "P1", "nom_pere_mere": "PM1",
    "age": 6, "sexe": "MALES", "handicap_poids": 56.0, "handicap_distance": 0,
    "deferre": None, "oeilleres": None, "musique": "1a 2a 3a",
    "nombre_courses": 20, "nombre_victoires": 3, "nombre_places": 8,
    "gains_carriere": 42000.0, "gains_annee_en_cours": 14000.0,
    "driver": "M. D1", "entraineur": None, "reduction_km_ms": 75000.0,
}


def _cadre():
    """
    Trois courses : une déjà courue, une à venir, une ligne importée.
    """
    lignes = []
    # Course 1 — courue ce matin, arrivée connue.
    for i in range(8):
        lignes.append({
            "course_id": 1, "num_pmu": i + 1, "id_cheval": f"CH{i}",
            "date_reunion": pd.Timestamp("2026-08-26").date(),
            "heure_depart": pd.Timestamp("2026-08-26 11:00", tz="UTC"),
            "ordre_arrivee": i + 1, "statut": "PARTANT", "source": "direct",
            "nombre_partants": 8, "place_corde": i + 1, **COMMUN,
        })
    # Course 2 — à venir cet après-midi : AUCUNE arrivée.
    for i in range(10):
        lignes.append({
            "course_id": 2, "num_pmu": i + 1, "id_cheval": f"CH{i}",
            "date_reunion": pd.Timestamp("2026-08-26").date(),
            "heure_depart": pd.Timestamp("2026-08-26 16:00", tz="UTC"),
            "ordre_arrivee": None, "statut": "PARTANT", "source": "direct",
            "nombre_partants": 10, "place_corde": i + 1, **COMMUN,
        })
    # Un non-partant déclaré dans la course à venir.
    lignes.append({
        "course_id": 2, "num_pmu": 11, "id_cheval": "CH99",
        "date_reunion": pd.Timestamp("2026-08-26").date(),
        "heure_depart": pd.Timestamp("2026-08-26 16:00", tz="UTC"),
        "ordre_arrivee": None, "statut": "NON_PARTANT", "source": "direct",
        "nombre_partants": 10, "place_corde": 11, **COMMUN,
    })
    # Une performance IMPORTÉE datée d'aujourd'hui : une trace du passé,
    # pas une course à prédire.
    lignes.append({
        "course_id": -1, "num_pmu": 1, "id_cheval": "CH0",
        "date_reunion": pd.Timestamp("2026-08-26").date(),
        "heure_depart": pd.Timestamp("2026-08-26 09:00", tz="UTC"),
        "ordre_arrivee": 3, "statut": "PARTANT", "source": "importe",
        "nombre_partants": 12, "place_corde": 1, **COMMUN,
    })
    return pd.DataFrame(lignes)


@pytest.fixture(scope="module")
def df():
    return ft.construire(_cadre(), avec_marche=True)


# ---------------------------------------------------------------------
# 1. LE test qui manquait
# ---------------------------------------------------------------------

def test_une_course_a_venir_est_pronosticable(df):
    """
    Le cœur du sujet. Une course dont l'arrivée est inconnue DOIT
    pouvoir être pronostiquée — c'est même le seul moment où un
    pronostic a une valeur.
    """
    a_venir = df[df["course_id"] == 2]
    partants = a_venir[a_venir["statut"] != "NON_PARTANT"]
    assert partants["est_pronosticable"].all(), (
        "une course à venir n'est pas pronosticable : le système ne "
        "pronostiquerait que des courses déjà courues"
    )


def test_une_course_a_venir_n_est_pas_une_cible_d_entrainement(df):
    """La réciproque : sans résultat, pas d'étiquette, donc pas d'exemple."""
    a_venir = df[df["course_id"] == 2]
    assert not a_venir["est_cible"].any()


def test_une_journee_entiere_a_venir_donne_des_partants():
    """
    Le scénario exact du 26/08 au matin : toutes les courses du jour
    sont devant nous. Filtrer sur `est_cible` rendait zéro partant et
    la journée restait vide jusqu'au soir.
    """
    brut = _cadre()
    brut = brut[brut["course_id"] == 2]          # que la course à venir
    d = ft.construire(brut, avec_marche=True)
    du_jour = d[d["est_pronosticable"] & d["statut"].ne("NON_PARTANT")]
    assert len(du_jour) == 10, f"{len(du_jour)} partants au lieu de 10"
    assert not d["est_cible"].any(), (
        "aucune de ces lignes ne peut servir à l'entraînement, "
        "et c'est normal"
    )


# ---------------------------------------------------------------------
# 2. Ce que le filtre doit continuer d'écarter
# ---------------------------------------------------------------------

def test_une_performance_importee_n_est_jamais_pronosticable(df):
    """
    Une performance importée datée d'aujourd'hui est une trace du
    passé, pas une course à prédire. Elle n'a ni cote, ni musique, ni
    entraîneur — la pronostiquer produirait une ligne creuse.
    """
    importee = df[df["source"] == "importe"]
    assert len(importee) == 1
    assert not importee["est_pronosticable"].any()
    assert not importee["est_cible"].any()


def test_un_non_partant_n_est_pas_pronosticable(df):
    ligne = df[(df["course_id"] == 2) & (df["statut"] == "NON_PARTANT")]
    assert len(ligne) == 1
    assert not ligne["est_pronosticable"].any()


def test_une_course_courue_reste_cible_et_pronosticable(df):
    """
    Une course déjà arrivée sert aux deux : elle alimente
    l'entraînement ET reste affichable avec son verdict.
    """
    courue = df[df["course_id"] == 1]
    assert courue["est_cible"].all()
    assert courue["est_pronosticable"].all()


# ---------------------------------------------------------------------
# 3. Le drapeau ne doit jamais devenir une feature
# ---------------------------------------------------------------------

def test_le_drapeau_ne_part_pas_dans_le_modele(df):
    """
    `est_pronosticable` vaut VRAI partout à la prédiction, et il est
    corrélé à l'arrivée à l'entraînement. Le laisser passer en feature
    serait une fuite — la même famille d'erreur que le chrono.
    """
    for avec in (True, False):
        cols = ft.colonnes_features(df, avec_marche=avec)
        assert "est_pronosticable" not in cols
        assert "est_cible" not in cols
        assert "est_exploitable" not in cols


def test_les_deux_drapeaux_ne_disent_pas_la_meme_chose(df):
    """
    S'ils étaient identiques, la séparation serait décorative et
    quelqu'un les refusionnerait un jour. Ils doivent diverger sur au
    moins une ligne — et ici, sur toute une course.
    """
    assert not df["est_cible"].equals(df["est_pronosticable"])
    divergentes = df[df["est_cible"] != df["est_pronosticable"]]
    assert len(divergentes) >= 10
    # La divergence porte bien sur les courses SANS arrivée.
    assert divergentes["ordre_arrivee"].isna().all()
