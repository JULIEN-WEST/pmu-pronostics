"""
Intégration des performances importées à l'historique.

Ces 108 000 lignes sont la principale réserve de profondeur du projet :
la collecte directe ne remonte qu'à son premier jour, elles remontent
plusieurs saisons. Mais les faire entrer dans le calcul touche au cœur
anti-fuite, d'où cette batterie dédiée.

Trois exigences, par ordre de gravité :

  1. Elles ne doivent JAMAIS devenir des exemples d'entraînement — leurs
     colonnes de cote, gains et musique sont vides.
  2. Elles doivent bien compter dans les cumuls glissants, sinon tout ce
     travail ne sert à rien.
  3. La règle anti-fuite doit continuer de tenir sur le cadre élargi.
"""

from __future__ import annotations

import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE / "scripts"))

DSN = os.environ.get("PMU_TEST_DSN")
if DSN:
    os.environ["DATABASE_URL"] = DSN

from pmu import dataset, db, features as ft  # noqa: E402


# ---------------------------------------------------------------------
# Cadres synthétiques — pas de base requise
# ---------------------------------------------------------------------

def _cadre(n_courses: int = 300, avec_importe: bool = True, graine: int = 3) -> pd.DataFrame:
    """
    Univers minimal mêlant lignes directes et lignes importées, avec un
    résultat TIRÉ AU HASARD : rien ne doit pouvoir le prédire.
    """
    rng = np.random.default_rng(graine)
    lignes = []
    t0 = pd.Timestamp("2025-01-01", tz="UTC")

    # Historique importé : antérieur, une ligne par cheval et par date.
    if avec_importe:
        for j in range(200):
            jour = t0 - pd.Timedelta(days=200 - j)
            for ch in rng.choice(400, size=40, replace=False):
                lignes.append({
                    # Courses partagées entre chevaux d'un même jour, et
                    # STRICTEMENT négatives : le +1 évite le -0, qui vaut 0
                    # et heurterait la première course directe.
                    "course_id": -(j * 1000 + int(ch) % 7 + 1),
                    "heure_depart": jour, "date_reunion": jour.date(),
                    "num_pmu": None, "id_cheval": int(ch),
                    "id_driver": None, "id_entraineur": None,
                    "nom_pere": f"P{int(ch) % 20}", "nom_pere_mere": f"PM{int(ch) % 13}",
                    "discipline": "ATTELE", "specialite": None,
                    "distance": 2700, "etat_terrain": "BON",
                    "hippodrome_code": "VIN", "nombre_partants": 12,
                    "montant_prix": 20000.0, "age": None, "sexe": "MALES",
                    "place_corde": None, "handicap_poids": None,
                    "deferre": None, "oeilleres": None, "musique": None,
                    "nombre_courses": None, "nombre_victoires": None,
                    "nombre_places": None, "gains_carriere": None,
                    "gains_annee_en_cours": None, "statut": "PARTANT",
                    "ordre_arrivee": int(rng.integers(1, 13)),
                    "cote_finale": None, "cote_ouverture": None,
                    "source": "importe",
                })

    # Lignes directes : postérieures, complètes.
    for i in range(n_courses):
        n = 12
        partants = rng.choice(400, size=n, replace=False)
        heure = t0 + pd.Timedelta(hours=6 * i)
        gagnant = rng.integers(0, n)
        for j, ch in enumerate(partants):
            lignes.append({
                "course_id": i, "heure_depart": heure, "date_reunion": heure.date(),
                "num_pmu": j + 1, "id_cheval": int(ch),
                "id_driver": int(rng.integers(1, 60)),
                "id_entraineur": int(rng.integers(1, 30)),
                "nom_pere": f"P{int(ch) % 20}", "nom_pere_mere": f"PM{int(ch) % 13}",
                "discipline": "ATTELE", "specialite": None,
                "distance": 2700, "etat_terrain": "BON",
                "hippodrome_code": "VIN", "nombre_partants": n,
                "montant_prix": 20000.0, "age": 6, "sexe": "MALES",
                "place_corde": j + 1, "handicap_poids": 56.0,
                "deferre": None, "oeilleres": None, "musique": "1a 2a 3a",
                "nombre_courses": 20, "nombre_victoires": 3, "nombre_places": 8,
                "gains_carriere": 30000.0, "gains_annee_en_cours": 9000.0,
                "statut": "PARTANT",
                "ordre_arrivee": int(1 if j == gagnant else rng.integers(2, n + 1)),
                "cote_finale": 8.0, "cote_ouverture": 9.0,
                "source": "direct",
            })
    return pd.DataFrame(lignes)


@pytest.fixture(scope="module")
def enrichi():
    return ft.construire(_cadre(), avec_marche=True)


# ---------------------------------------------------------------------
# 1. Les lignes importées ne sont jamais des exemples
# ---------------------------------------------------------------------

def test_les_lignes_importees_ne_sont_pas_des_cibles(enrichi):
    importe = enrichi[enrichi["source"] == "importe"]
    assert len(importe) > 5000
    assert not importe["est_cible"].any(), (
        "une performance importée est devenue un exemple d'entraînement — "
        "ses colonnes de cote, gains et musique sont vides"
    )


def test_les_lignes_importees_comptent_dans_l_historique(enrichi):
    """Sinon tout l'exercice est inutile."""
    importe = enrichi[enrichi["source"] == "importe"]
    assert importe["est_exploitable"].all()


def test_les_lignes_directes_restent_des_cibles(enrichi):
    direct = enrichi[enrichi["source"] == "direct"]
    assert direct["est_cible"].all()


def test_colonnes_features_refuse_les_marqueurs(enrichi):
    """`source` et `est_cible` ne doivent jamais partir dans le modèle."""
    cols = ft.colonnes_features(enrichi, avec_marche=True)
    for interdite in ("source", "est_cible", "est_exploitable"):
        assert interdite not in cols


# ---------------------------------------------------------------------
# 2. Le gain de profondeur
# ---------------------------------------------------------------------

def test_l_historique_importe_augmente_la_profondeur():
    """
    La mesure qui justifie tout : combien de courses antérieures un cheval
    a-t-il derrière lui au moment où on doit le juger ?
    """
    sans = ft.construire(_cadre(avec_importe=False), avec_marche=True)
    avec = ft.construire(_cadre(avec_importe=True), avec_marche=True)

    def profondeur(df):
        cibles = df[df["est_cible"]]
        return float(cibles["h_cheval_place_n"].mean())

    p_sans, p_avec = profondeur(sans), profondeur(avec)
    assert p_avec > p_sans * 2, (
        f"profondeur moyenne : {p_sans:.1f} sans historique importé, "
        f"{p_avec:.1f} avec — le gain attendu n'est pas là"
    )


# ---------------------------------------------------------------------
# 3. L'anti-fuite tient sur le cadre élargi
# ---------------------------------------------------------------------

def test_le_canari_tient_avec_l_historique_importe(enrichi):
    """
    Même contrôle que `test_fuite.py`, mais sur le cadre mixte. Les lignes
    importées introduisent des courses synthétiques et des colonnes
    vides : deux occasions de casser l'exclusion intra-course.
    """
    df = enrichi[enrichi["est_cible"]].copy()
    cols = ft.colonnes_features(df, avec_marche=False)
    X = df[cols].apply(pd.to_numeric, errors="coerce")
    y = df["y_gagnant"]

    coupe = int(len(df) * 0.7)
    modele = HistGradientBoostingClassifier(max_iter=120, random_state=0)
    modele.fit(X.iloc[:coupe], y.iloc[:coupe])
    auc = roc_auc_score(y.iloc[coupe:], modele.predict_proba(X.iloc[coupe:])[:, 1])

    assert auc < 0.58, (
        f"AUC = {auc:.3f} sur une cible aléatoire, avec historique importé : "
        "l'exclusion intra-course ne tient plus sur les courses synthétiques."
    )


def test_les_courses_synthetiques_ne_heurtent_pas_les_vraies(enrichi):
    """
    Les identifiants de course importés sont négatifs, ceux des courses
    réelles positifs. Une collision ferait fusionner deux épreuves sans
    rapport dans le calcul d'exclusion.
    """
    direct = set(enrichi.loc[enrichi["source"] == "direct", "course_id"])
    importe = set(enrichi.loc[enrichi["source"] == "importe", "course_id"])
    assert not (direct & importe)
    assert all(c < 0 for c in importe)


# ---------------------------------------------------------------------
# 4. La requête SQL, contre une vraie base
# ---------------------------------------------------------------------

pytestmark_db = pytest.mark.skipif(not DSN, reason="PMU_TEST_DSN non défini")


@pytestmark_db
def test_union_sql_et_deduplication():
    """
    Vérifie l'assemblage réel : union, déduplication, identifiants
    négatifs, et récupération de la généalogie depuis la table `cheval`.
    """
    from simulateur import generer
    import tests.test_integration as ti  # réutilise l'amorçage existant

    with db.connect(DSN) as conn:
        conn.execute("DROP SCHEMA IF EXISTS pmu CASCADE")
        conn.commit()
        ti._semer(conn, n_courses=120)

        # Une performance importée ANTÉRIEURE : doit apparaître.
        row = conn.execute(
            "SELECT id_cheval, nom_pere FROM cheval WHERE nom_pere IS NOT NULL LIMIT 1"
        ).fetchone()
        ancienne = date(2022, 5, 4)
        db.insert_performances(conn, [{
            "id_cheval": row["id_cheval"], "date_course": ancienne,
            "hippodrome_lib": "VINCENNES", "hippodrome_code": "VIN",
            "nom_prix": "PRIX ANCIEN", "discipline": "ATTELE", "specialite": None,
            "distance": 2700, "allocation": 30000.0, "nb_participants": 14,
            "place": 3, "statut_arrivee": "PLACE", "corde": 5,
            "poids_jockey": None, "nom_jockey": "M. PROTTI", "oeillere": None,
            "deferre": None, "etat_terrain": "BON", "temps_premier_ms": None,
            "reduction_km_ms": None, "distance_avec_precedent": None,
        }])

        # Une performance qui DOUBLONNE une course déjà collectée en direct.
        doublon = conn.execute(
            """SELECT p.id_cheval, c.date_reunion
                 FROM partant p JOIN course c ON c.course_id = p.course_id
                WHERE p.id_cheval IS NOT NULL LIMIT 1"""
        ).fetchone()
        db.insert_performances(conn, [{
            "id_cheval": doublon["id_cheval"], "date_course": doublon["date_reunion"],
            "hippodrome_lib": "AILLEURS", "hippodrome_code": "AIL",
            "nom_prix": "DOUBLON", "discipline": "ATTELE", "specialite": None,
            "distance": 1234, "allocation": None, "nb_participants": 10,
            "place": 1, "statut_arrivee": "PLACE", "corde": None,
            "poids_jockey": None, "nom_jockey": None, "oeillere": None,
            "deferre": None, "etat_terrain": None, "temps_premier_ms": None,
            "reduction_km_ms": None, "distance_avec_precedent": None,
        }])
        conn.commit()

        df = dataset.charger(conn, date(2020, 1, 1), date(2030, 1, 1))

    sources = df["source"].value_counts().to_dict()
    assert sources.get("direct", 0) > 1000
    assert sources.get("importe", 0) == 1, (
        f"attendu 1 ligne importée après déduplication, obtenu "
        f"{sources.get('importe', 0)} — le NOT EXISTS ne filtre pas"
    )

    ligne = df[df["source"] == "importe"].iloc[0]
    assert ligne["date_reunion"] == ancienne
    assert ligne["course_id"] < 0, "identifiant de course synthétique non négatif"
    assert ligne["nom_pere"] == row["nom_pere"], "généalogie non récupérée"
    assert pd.isna(ligne["cote_finale"])
    assert ligne["ordre_arrivee"] == 3


@pytestmark_db
def test_desactivation_de_l_historique_importe():
    """L'option doit pouvoir être coupée sans rien casser."""
    with db.connect(DSN) as conn:
        df = dataset.charger(conn, date(2020, 1, 1), date(2030, 1, 1),
                             avec_historique_importe=False)
        assert set(df["source"]) == {"direct"}
        conn.execute("DROP SCHEMA IF EXISTS pmu CASCADE")
        conn.commit()
