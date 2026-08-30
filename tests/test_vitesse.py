"""
Vitesse, marge et recul — et surtout : la fuite qu'ils pourraient créer.

`reduction_km_ms` et `distance_cheval_precedent` sont des colonnes de
RÉSULTAT : elles n'existent qu'après l'arrivée. Les faire entrer dans les
features est le geste le plus dangereux du projet — un décalage oublié et
le modèle lit le chrono de la course qu'il doit prédire. Il afficherait
alors des scores superbes et perdrait tout en réel.

D'où le canari de ce fichier : on fabrique un chrono qui DÉSIGNE le
gagnant de sa propre ligne. Si la moindre fuite existe, le modèle
atteindra une AUC proche de 1. S'il reste autour de 0,5, c'est que seules
les courses antérieures ont été lues.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE / "src"))

from pmu import features as ft  # noqa: E402
from pmu.train import ModelePmu  # noqa: E402
from pmu.normalize import MARGE_DISTANCE, parse_marge  # noqa: E402


# ---------------------------------------------------------------------
# 1. Le parseur de marge
# ---------------------------------------------------------------------

@pytest.mark.parametrize("texte, attendu", [
    ("2 L", 2.0), ("1 L", 1.0), ("12 L", 12.0),
    ("1/2 L", 0.5), ("3/4 L", 0.75), ("1/2 LONGUEUR", 0.5),
    ("2 1/2 L", 2.5), ("1 1/4 L", 1.25),
    ("NEZ", 0.05), ("TETE", 0.15), ("COURTE TETE", 0.10), ("ENCOLURE", 0.30),
    ("DIST.", MARGE_DISTANCE), ("DIST", MARGE_DISTANCE),
    ("DEAD-HEAT", 0.0),
    ("", None), (None, None), ("n'importe quoi", None),
])
def test_parse_marge(texte, attendu):
    assert parse_marge(texte) == attendu


def test_la_demi_longueur_ne_devient_pas_une_longueur():
    """
    Le piège exact : « 1/2 » commence par un chiffre. Un motif « nombre »
    testé en premier rendrait 1 au lieu de 0,5 — soit un facteur deux sur
    les arrivées serrées, celles où la marge est justement informative.
    """
    assert parse_marge("1/2 L") == 0.5
    assert parse_marge("1/2 L") != 1.0


def test_marge_absente_ne_vaut_pas_zero():
    """0 signifie « arrivés ensemble ». Une info manquante, c'est None."""
    assert parse_marge(None) is None
    assert parse_marge("") is None


def test_marge_accepte_un_dictionnaire():
    """L'API PMU renvoie parfois un objet au lieu d'une chaîne."""
    assert parse_marge({"libelleCourt": "2 L"}) == 2.0


# ---------------------------------------------------------------------
# 2. Le passé du cheval, à la main
# ---------------------------------------------------------------------

def test_passe_du_cheval_ne_voit_que_l_anterieur():
    """
    Trois courses d'un même cheval, chronos 100, 200, 300. À la première
    il ne sait rien ; à la deuxième il connaît 100 ; à la troisième la
    moyenne de 100 et 200. Jamais sa propre valeur.
    """
    df = pd.DataFrame({
        "id_cheval": ["A", "A", "A"],
        "course_id": [1, 2, 3],
        "est_exploitable": [True, True, True],
        "chrono": [100.0, 200.0, 300.0],
    })
    r = ft._passe_du_cheval(df, df["chrono"])
    assert pd.isna(r["moy"].iloc[0])
    assert r["moy"].iloc[1] == 100.0
    assert r["moy"].iloc[2] == 150.0
    assert list(r["n"]) == [0.0, 1.0, 2.0]
    assert pd.isna(r["derniere"].iloc[0])
    assert r["derniere"].iloc[2] == 200.0
    assert r["best"].iloc[2] == 100.0


def test_passe_du_cheval_ignore_les_lignes_non_exploitables():
    """Un non-partant n'a pas couru : son chrono ne doit rien peser."""
    df = pd.DataFrame({
        "id_cheval": ["A", "A", "A"],
        "course_id": [1, 2, 3],
        "est_exploitable": [True, False, True],
        "chrono": [100.0, 999.0, 300.0],
    })
    r = ft._passe_du_cheval(df, df["chrono"])
    assert r["moy"].iloc[2] == 100.0, "le chrono du non-partant a été compté"
    assert r["n"].iloc[2] == 1.0


def test_passe_du_cheval_supporte_les_trous():
    """Toutes les courses n'ont pas de chrono publié."""
    df = pd.DataFrame({
        "id_cheval": ["A"] * 4,
        "course_id": [1, 2, 3, 4],
        "est_exploitable": [True] * 4,
        "chrono": [np.nan, 200.0, np.nan, 400.0],
    })
    r = ft._passe_du_cheval(df, df["chrono"])
    assert pd.isna(r["moy"].iloc[1])
    assert r["moy"].iloc[3] == 200.0
    assert r["derniere"].iloc[3] == 200.0
    assert r["best"].iloc[3] == 200.0


def test_les_chevaux_ne_se_melangent_pas():
    df = pd.DataFrame({
        "id_cheval": ["A", "B", "A", "B"],
        "course_id": [1, 1, 2, 2],
        "est_exploitable": [True] * 4,
        "chrono": [100.0, 900.0, 110.0, 910.0],
    })
    r = ft._passe_du_cheval(df, df["chrono"])
    assert r["moy"].iloc[2] == 100.0
    assert r["moy"].iloc[3] == 900.0


# ---------------------------------------------------------------------
# 3. LE CANARI — un chrono qui désigne son propre gagnant
# ---------------------------------------------------------------------

def _cadre_piege(n_courses: int = 500, graine: int = 11) -> pd.DataFrame:
    """
    Cible ALÉATOIRE, et chrono construit pour TRAHIR le gagnant de sa
    propre ligne : 1000 pour le gagnant, 2000 pour les autres.

    Si `construire()` laisse passer la moindre lecture non décalée, le
    modèle atteindra une AUC quasi parfaite. C'est un piège volontaire.
    """
    rng = np.random.default_rng(graine)
    lignes = []
    t0 = pd.Timestamp("2025-01-01", tz="UTC")
    for i in range(n_courses):
        n = 10
        chevaux = rng.choice(300, size=n, replace=False)
        gagnant = int(rng.integers(0, n))
        heure = t0 + pd.Timedelta(hours=3 * i)
        places = rng.permutation(np.arange(1, n + 1))
        places[gagnant] = 1
        # on remet un vrai classement cohérent autour du gagnant tiré
        reste = [p for p in range(2, n + 1)]
        rng.shuffle(reste)
        k = 0
        for j in range(n):
            if j == gagnant:
                continue
            places[j] = reste[k]
            k += 1
        for j in range(n):
            lignes.append({
                "course_id": i, "heure_depart": heure, "date_reunion": heure.date(),
                "num_pmu": j + 1, "id_cheval": int(chevaux[j]),
                "id_driver": int(rng.integers(1, 50)),
                "id_entraineur": int(rng.integers(1, 25)),
                "id_proprietaire": int(rng.integers(1, 40)),
                "nom_pere": f"P{int(chevaux[j]) % 15}",
                "nom_pere_mere": f"PM{int(chevaux[j]) % 11}",
                "discipline": "ATTELE", "specialite": None,
                "distance": 2700, "etat_terrain": "BON", "hippodrome_code": "VIN",
                "nombre_partants": n, "montant_prix": 20000.0,
                "age": 6, "sexe": "MALES", "place_corde": j + 1,
                "handicap_poids": 56.0, "handicap_distance": 0,
                "deferre": None, "oeilleres": None, "musique": "1a 2a 3a",
                "nombre_courses": 20, "nombre_victoires": 3, "nombre_places": 8,
                "gains_carriere": 30000.0, "gains_annee_en_cours": 9000.0,
                "statut": "PARTANT", "ordre_arrivee": int(places[j]),
                # ── le piège ──
                "reduction_km_ms": 1000 if places[j] == 1 else 2000,
                "temps_officiel_ms": 100000,
                "distance_cheval_precedent": "NEZ" if places[j] == 1 else "5 L",
                "cote_finale": 8.0, "cote_ouverture": 9.0,
                "source": "direct",
            })
    return pd.DataFrame(lignes)


@pytest.fixture(scope="module")
def piege():
    return ft.construire(_cadre_piege(), avec_marche=True)


def test_le_chrono_de_la_course_en_cours_ne_fuit_pas(piege):
    """
    LE test du fichier. Chrono = 1000 pour le gagnant, 2000 sinon. Une
    AUC élevée signerait une lecture de la ligne courante.
    """
    df = piege[piege["est_cible"]].copy()
    cols = ft.colonnes_features(df, avec_marche=False)
    assert any(c.startswith("v_") for c in cols), \
        "aucune feature de vitesse : le test ne prouverait rien"

    y = df["y_gagnant"]
    coupe = int(len(df) * 0.7)
    preparation = ModelePmu(colonnes=cols)
    preparation._apprendre_categories(df.iloc[:coupe][cols])
    X = preparation._matrice(df)
    masque_categories = [c in preparation._colonnes_categorielles() for c in cols]
    modele = HistGradientBoostingClassifier(
        max_iter=150,
        random_state=0,
        categorical_features=masque_categories if any(masque_categories) else None,
    )
    modele.fit(X.iloc[:coupe], y.iloc[:coupe])
    auc = roc_auc_score(y.iloc[coupe:], modele.predict_proba(X.iloc[coupe:])[:, 1])

    assert auc < 0.58, (
        f"AUC = {auc:.3f} sur une cible aléatoire alors que le chrono "
        "désigne le gagnant de sa propre ligne : la vitesse fuit."
    )


def test_les_colonnes_de_resultat_restent_hors_du_modele(piege):
    cols = set(ft.colonnes_features(piege, avec_marche=True))
    for interdite in ("reduction_km_ms", "temps_officiel_ms",
                      "distance_cheval_precedent", "ordre_arrivee"):
        assert interdite not in cols


def test_la_premiere_course_d_un_cheval_n_a_pas_de_vitesse(piege):
    """Sans passé, pas de chrono. Un 0 laisserait croire à un cheval très rapide."""
    premiere = piege.groupby("id_cheval", sort=False).head(1)
    assert premiere["v_reduction_moy"].isna().all()
    assert (premiere["v_reduction_n"] == 0).all()


def test_la_vitesse_est_bien_calculee_apres_coup(piege):
    """Le passé DOIT être lu — sinon l'anti-fuite serait trivialement vraie."""
    tardives = piege.groupby("id_cheval", sort=False).tail(1)
    renseignees = tardives["v_reduction_moy"].notna().mean()
    assert renseignees > 0.8, (
        f"seulement {renseignees:.0%} des dernières courses ont une vitesse "
        "connue — le calcul ne remonte pas l'historique"
    )


# ---------------------------------------------------------------------
# 4. Recul et écurie
# ---------------------------------------------------------------------

def test_le_recul_est_relatif_au_lot():
    """Tout le monde reculé de 25 m, c'est une course sans handicap."""
    base = _cadre_piege(n_courses=30)
    base["handicap_distance"] = 25
    df = ft.construire(base, avec_marche=False)
    assert (df["c_recul"] == 25).all()
    assert (df["c_recul_relatif_lot"] == 0).all(), \
        "un recul uniforme doit s'annuler une fois rapporté au lot"


def test_le_recul_differencie_quand_il_varie():
    base = _cadre_piege(n_courses=30)
    base["handicap_distance"] = np.where(base["num_pmu"] > 5, 25, 0)
    df = ft.construire(base, avec_marche=False)
    ecarts = df.groupby("course_id")["c_recul_relatif_lot"].max()
    assert (ecarts == 25).all()


def test_l_ecurie_produit_une_feature(piege):
    assert "h_proprio_place" in piege.columns
    assert piege["h_proprio_place"].notna().any()
    assert "h_proprio_place" in ft.colonnes_features(piege)


def test_tout_marche_sans_les_colonnes_optionnelles():
    """
    Une base d'avant l'étape 3 n'a ni chrono ni recul ni propriétaire.
    Le pipeline doit tourner quand même, sinon un redéploiement casse la
    production avant le premier réentraînement.
    """
    base = _cadre_piege(n_courses=40).drop(
        columns=["reduction_km_ms", "temps_officiel_ms",
                 "distance_cheval_precedent", "handicap_distance",
                 "id_proprietaire"])
    df = ft.construire(base, avec_marche=True)
    cols = ft.colonnes_features(df, avec_marche=True)
    assert cols, "aucune feature construite"
    assert not any(c.startswith("v_") for c in cols)
    assert "h_proprio_place" not in cols
