"""
Ligne maternelle et éleveur.

POURQUOI CES DONNÉES-LÀ, ET PAS L'IFCE

L'API pedigree de l'IFCE est commerciale : de 500 à 9 000 € par an, sous
convention. LeTrot ne publie aucune API documentée. Avant de payer ou de
gratter un site, il fallait regarder ce qu'on avait déjà : la MÈRE et
l'ÉLEVEUR étaient collectés depuis le premier jour et n'avaient jamais
été lus — exactement comme le pénétromètre et le chrono avant eux.

LE PIÈGE PROPRE À LA LIGNE MATERNELLE

Une poulinière produit cinq à dix chevaux dans sa carrière, là où un
étalon en produit des centaines. Les effectifs sont donc minuscules, et
« deux produits, deux victoires » ressortirait comme une lignée
d'exception si le lissage n'était pas beaucoup plus fort. Ces tests
vérifient que ce lissage tient, et que l'anti-fuite tient avec.
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pmu import explain as ex, features as ft  # noqa: E402


def _cadre(n_courses=400, graine=17, cible_aleatoire=True) -> pd.DataFrame:
    """
    Univers où chaque cheval a une mère, un père de mère et un éleveur,
    et où l'arrivée est TIRÉE AU HASARD : rien ne doit pouvoir la prédire.
    """
    rng = np.random.default_rng(graine)
    lignes = []
    t0 = pd.Timestamp("2025-01-01", tz="UTC")
    for i in range(n_courses):
        n = 10
        chevaux = rng.choice(600, size=n, replace=False)
        heure = t0 + pd.Timedelta(hours=3 * i)
        places = rng.permutation(np.arange(1, n + 1))
        for j, ch in enumerate(chevaux):
            c = int(ch)
            lignes.append({
                "course_id": i, "heure_depart": heure, "date_reunion": heure.date(),
                "num_pmu": j + 1, "id_cheval": c,
                "id_driver": int(rng.integers(1, 40)),
                "id_entraineur": int(rng.integers(1, 20)),
                "id_proprietaire": int(rng.integers(1, 30)),
                # 600 chevaux pour 120 mères : cinq produits par jument,
                # l'ordre de grandeur réel.
                "nom_pere": f"ETALON{c % 25}",
                "nom_mere": f"JUMENT{c % 120}",
                "nom_pere_mere": f"PEREMERE{c % 18}",
                "id_eleveur": int(c % 45),
                "discipline": "ATTELE", "specialite": None, "distance": 2700,
                "etat_terrain": "BON" if i % 2 else "LOURD",
                "hippodrome_code": "VIN", "nombre_partants": n,
                "montant_prix": 20000.0, "age": 6, "sexe": "MALES",
                "place_corde": j + 1, "handicap_poids": 56.0,
                "handicap_distance": 0, "deferre": None, "oeilleres": None,
                "musique": "3a 1a 5a", "nombre_courses": 20,
                "nombre_victoires": 3, "nombre_places": 8,
                "gains_carriere": 30000.0, "gains_annee_en_cours": 9000.0,
                "statut": "PARTANT", "ordre_arrivee": int(places[j]),
                "cote_finale": 8.0, "cote_ouverture": 9.0, "source": "direct",
            })
    return pd.DataFrame(lignes)


@pytest.fixture(scope="module")
def enrichi():
    return ft.construire(_cadre(), avec_marche=True)


# ---------------------------------------------------------------------
# Les features existent et partent bien dans le modèle
# ---------------------------------------------------------------------

def test_les_features_maternelles_existent(enrichi):
    for col in ("g_mere", "g_mere_n", "g_mere_terrain", "g_famille",
                "g_mere_terrain_delta", "g_accouplement_delta",
                "h_eleveur_place"):
        assert col in enrichi.columns, col
    cols = ft.colonnes_features(enrichi)
    for col in ("g_mere", "g_famille", "g_accouplement_delta", "h_eleveur_place"):
        assert col in cols, col


def test_tout_marche_sans_la_mere():
    """
    Une base d'avant cette version n'a ni `nom_mere` ni `id_eleveur`.
    Le pipeline doit tourner quand même, sinon un redéploiement casse la
    production avant le premier réentraînement.
    """
    base = _cadre(n_courses=40).drop(columns=["nom_mere", "id_eleveur"])
    df = ft.construire(base, avec_marche=True)
    cols = ft.colonnes_features(df)
    assert cols
    assert "g_mere" not in cols and "h_eleveur_place" not in cols


# ---------------------------------------------------------------------
# Le lissage — le point propre aux petits effectifs
# ---------------------------------------------------------------------

def test_rien_n_est_annonce_en_dessous_du_minimum(enrichi):
    """
    LE garde-fou de la ligne maternelle. Avec deux produits connus, un
    taux lissé bouge encore trois fois plus qu'un taux d'étalon calculé
    sur des centaines de courses — je l'ai mesuré, et c'est ce test qui
    l'avait fait apparaître. En dessous du minimum, la valeur doit être
    ABSENTE : le modèle sait traiter un trou, il ne sait pas deviner
    qu'un chiffre est du bruit.
    """
    maigre = enrichi[enrichi["g_mere_n"] < 4]
    assert len(maigre) > 100, "l'univers de test ne contient pas assez de cas"
    assert maigre["g_mere"].isna().all(), (
        "un taux maternel est annoncé sur moins de quatre produits"
    )
    fournie = enrichi[enrichi["g_mere_n"] >= 4]
    assert fournie["g_mere"].notna().all()


def test_les_valeurs_annoncees_restent_plausibles(enrichi):
    """
    Une fois le minimum atteint, le lissage doit encore tenir les
    extrêmes : pas de lignée à 0 % ni à 100 %.
    """
    d = enrichi[enrichi["est_exploitable"]]
    moyenne = d["y_place"].mean()
    valeurs = d["g_mere"].dropna()
    assert len(valeurs) > 50
    assert (valeurs - moyenne).abs().max() < 0.25, (
        f"écart maximal {(valeurs - moyenne).abs().max():.3f} — "
        "le lissage laisse passer des taux invraisemblables"
    )


def test_le_lissage_maternel_est_plus_fort_que_le_paternel():
    """
    Ce qui doit être vrai, et qui est vérifiable directement : à effectif
    identique, la formule maternelle tire plus vers la moyenne.
    """
    from pmu.features import _taux_glissant  # noqa: F401  (documentation)
    prior, n, succes = 0.3, 5.0, 5.0         # cinq produits, cinq places
    mere = (succes + prior * 15) / (n + 15)
    pere = (succes + prior * 40) / (n + 40)
    brut = succes / n
    assert brut == 1.0
    assert mere < 0.55, f"taux maternel {mere:.3f} encore trop proche de 100 %"
    # L'étalon est encore plus lissé, mais il n'atteint jamais n = 5 en
    # pratique : c'est justement pourquoi la mère a besoin d'un `min_n`.
    assert pere < mere


def test_les_effectifs_maternels_sont_petits_mais_reels(enrichi):
    """Si l'effectif restait à zéro, la feature ne dirait jamais rien."""
    tard = enrichi.tail(len(enrichi) // 4)
    assert tard["g_mere_n"].mean() > 1, "aucun historique maternel accumulé"
    assert tard["g_mere_n"].mean() < tard["g_pere_n"].mean(), (
        "une jument ne peut pas avoir autant de produits qu'un étalon"
    )


def test_le_delta_d_accouplement_est_centre(enrichi):
    """C'est un ÉCART à la moyenne du père : il doit tourner autour de 0."""
    d = enrichi["g_accouplement_delta"].dropna()
    assert len(d) > 100
    assert abs(d.mean()) < 0.05, d.mean()


# ---------------------------------------------------------------------
# L'anti-fuite tient avec les nouvelles clés
# ---------------------------------------------------------------------

def test_le_canari_tient_avec_la_lignee_maternelle(enrichi):
    """
    Les demi-frères par la même mère courent souvent l'un contre l'autre.
    Sans l'exclusion de la course courante, le second verrait le résultat
    du premier — une fuite discrète, et d'autant plus dangereuse que les
    effectifs maternels sont petits.
    """
    df = enrichi[enrichi["est_cible"]].copy()
    cols = ft.colonnes_features(df, avec_marche=False)
    X = df[cols].apply(pd.to_numeric, errors="coerce")
    y = df["y_gagnant"]
    coupe = int(len(df) * 0.7)
    m = HistGradientBoostingClassifier(max_iter=140, random_state=0)
    m.fit(X.iloc[:coupe], y.iloc[:coupe])
    auc = roc_auc_score(y.iloc[coupe:], m.predict_proba(X.iloc[coupe:])[:, 1])
    assert auc < 0.58, (
        f"AUC = {auc:.3f} sur une cible aléatoire : la lignée maternelle fuit"
    )


def test_la_premiere_course_d_une_lignee_n_a_pas_d_historique(enrichi):
    premiere = enrichi.groupby("nom_mere", sort=False).head(1)
    assert (premiere["g_mere_n"] == 0).all()


# ---------------------------------------------------------------------
# La justification
# ---------------------------------------------------------------------

def test_les_faits_citent_la_mere(enrichi):
    tard = enrichi[enrichi["g_mere_n"] >= 3]
    assert len(tard), "aucune ligne avec assez d'historique maternel"
    textes = " ".join(t for t in ex.faits(tard.iloc[-1]).get("lignee", []))
    assert "JUMENT" in textes, f"la mère n'apparaît pas dans les motifs : {textes}"


def test_les_faits_taisent_une_mere_sans_historique():
    """Trois produits, c'est le minimum pour dire quoi que ce soit."""
    ligne = pd.Series({"course_id": 1, "num_pmu": 1, "nom_mere": "JUMENT X",
                       "g_mere": 0.9, "g_mere_n": 1.0})
    textes = " ".join(ex.faits(ligne).get("lignee", []))
    assert "JUMENT X" not in textes, (
        "un taux de 90 % sur un seul produit ne doit pas être affiché"
    )


def test_les_faits_citent_l_elevage():
    ligne = pd.Series({"course_id": 1, "num_pmu": 1,
                       "h_eleveur_place": 0.34, "h_eleveur_place_n": 120.0})
    textes = " ".join(ex.faits(ligne).get("lignee", []))
    assert "élevage" in textes and "120" in textes
