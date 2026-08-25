"""
Cible ordinale, trajectoire des cotes, conditions d'engagement, abstention.

Ce que ces tests protègent, dans l'ordre de gravité :

  1. L'HYGIÈNE DU DÉCOUPAGE du modèle ordinal. Il consomme la fenêtre
     de calibration en deux moitiés — empileur puis isotonie. Si les
     deux apprenaient sur les mêmes lignes, la calibration annoncerait
     une justesse qu'elle n'a pas, et « 20 % » cesserait de vouloir
     dire une fois sur cinq.
  2. Les cibles ordinales sont des RÉSULTATS. `y_top3` dans les
     features reviendrait à donner l'arrivée au modèle.
  3. Le seuil d'abstention doit être MESURÉ. Un seuil choisi à la main
     pour faire joli est pire que pas de seuil du tout : il donne une
     fausse assurance sur les courses qu'il laisse passer.
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

from pmu import evaluate as ev, features as ft  # noqa: E402
from pmu.train import (Decoupage, ModeleOrdinal, ModelePmu,  # noqa: E402
                       charger_modele, modele_present)
from test_explain import _cadre  # noqa: E402


@pytest.fixture(scope="module")
def enrichi():
    return ft.construire(_cadre(n_courses=700, decideur="palmares"), avec_marche=True)


@pytest.fixture(scope="module")
def decoupage(enrichi):
    return Decoupage.par_proportions(enrichi["heure_depart"], 0.6, 0.2)


@pytest.fixture(scope="module")
def ordinal(enrichi, decoupage):
    return ModeleOrdinal(cible="y_gagnant").entrainer(enrichi, decoupage)


# ---------------------------------------------------------------------
# 1. Les cibles ordinales
# ---------------------------------------------------------------------

def test_les_cibles_ordinales_sont_emboitees(enrichi):
    """
    Être dans les 2 premiers implique être dans les 3, et ainsi de
    suite. Si l'emboîtement est cassé, la décomposition ne décrit plus
    un ordre et l'empileur apprend du bruit.
    """
    d = enrichi[enrichi["est_exploitable"]]
    assert (d["y_gagnant"] <= d["y_top2"]).all()
    assert (d["y_top2"] <= d["y_top3"]).all()
    assert (d["y_top3"] <= d["y_top5"]).all()


def test_les_seuils_sont_de_plus_en_plus_frequents(enrichi):
    """C'est tout l'intérêt : plus d'exemples positifs par course."""
    d = enrichi[enrichi["est_exploitable"]]
    taux = [d[n].mean() for n, _ in ft.SEUILS_ORDINAUX]
    assert taux == sorted(taux), taux
    assert taux[-1] > taux[0] * 2, (
        f"y_top5 ({taux[-1]:.2%}) devrait être bien plus fréquent que "
        f"y_gagnant ({taux[0]:.2%}) — sinon la décomposition n'apporte rien"
    )


def test_les_cibles_ordinales_ne_partent_pas_dans_le_modele(enrichi):
    cols = set(ft.colonnes_features(enrichi, avec_marche=True))
    for nom, _ in ft.SEUILS_ORDINAUX:
        assert nom not in cols, f"{nom} est une colonne de résultat"


# ---------------------------------------------------------------------
# 2. Le modèle ordinal
# ---------------------------------------------------------------------

def test_l_ordinal_produit_des_probabilites_valides(ordinal, enrichi, decoupage):
    _, _, m_test = decoupage.masques(enrichi["heure_depart"])
    out = ordinal.predire(enrichi[m_test])
    assert out["proba"].between(0, 1).all()
    sommes = out.groupby("course_id")["proba"].sum()
    assert np.allclose(sommes, 1.0, atol=1e-6)
    assert out["rang"].min() == 1


def test_l_ordinal_entraine_bien_plusieurs_seuils(ordinal):
    assert len(ordinal.seuils) >= 3, ordinal.seuils
    assert set(ordinal.modeles) == set(ordinal.seuils)
    assert ordinal.empileur is not None
    assert ordinal.calibrateur is not None


def test_l_empileur_et_l_isotonie_n_apprennent_pas_sur_les_memes_lignes(
        enrichi, decoupage, monkeypatch):
    """
    LE contrôle d'hygiène. On empoisonne la SECONDE moitié de la fenêtre
    de calibration : si l'empileur y avait appris, ses coefficients
    changeraient. Ils ne doivent pas bouger — seule l'isotonie s'y ajuste.
    """
    propre = ModeleOrdinal(cible="y_gagnant").entrainer(enrichi, decoupage)

    _, m_calib, _ = decoupage.masques(enrichi["heure_depart"])
    calib = enrichi[m_calib]
    milieu = calib["heure_depart"].quantile(0.5)
    piege = enrichi.copy()
    seconde_moitie = m_calib & (enrichi["heure_depart"] > milieu)
    piege.loc[seconde_moitie, "y_gagnant"] = 1.0 - piege.loc[seconde_moitie, "y_gagnant"]

    sali = ModeleOrdinal(cible="y_gagnant").entrainer(piege, decoupage)
    np.testing.assert_allclose(
        propre.empileur.coef_, sali.empileur.coef_, rtol=1e-9,
        err_msg="l'empileur a vu la seconde moitié de la calibration — "
                "la calibration annoncerait une justesse qu'elle n'a pas",
    )


def test_l_ordinal_ne_regarde_jamais_le_test(enrichi, decoupage):
    """Même contrôle, sur la fenêtre de test cette fois."""
    _, _, m_test = decoupage.masques(enrichi["heure_depart"])
    piege = enrichi.copy()
    piege.loc[m_test, "y_gagnant"] = 1.0 - piege.loc[m_test, "y_gagnant"]

    a = ModeleOrdinal(cible="y_gagnant").entrainer(enrichi, decoupage)
    b = ModeleOrdinal(cible="y_gagnant").entrainer(piege, decoupage)
    np.testing.assert_allclose(a.empileur.coef_, b.empileur.coef_, rtol=1e-9)


def test_l_ordinal_refuse_une_calibration_trop_courte(enrichi):
    """Mieux vaut une erreur nette qu'un empileur ajusté sur trente lignes."""
    d = Decoupage.par_proportions(enrichi["heure_depart"], 0.98, 0.005)
    with pytest.raises(ValueError):
        ModeleOrdinal(cible="y_gagnant").entrainer(enrichi, d)


def test_aller_retour_sur_disque(ordinal, enrichi, decoupage, tmp_path):
    _, _, m_test = decoupage.masques(enrichi["heure_depart"])
    test = enrichi[m_test]
    avant = ordinal.predire(test)["proba"]
    ordinal.sauver(tmp_path / "m")
    assert modele_present(tmp_path / "m")
    relu = charger_modele(tmp_path / "m")
    assert isinstance(relu, ModeleOrdinal)
    pd.testing.assert_series_equal(avant.sort_index(),
                                   relu.predire(test)["proba"].sort_index(),
                                   check_names=False)


# ---------------------------------------------------------------------
# 3. Trajectoire des cotes et conditions d'engagement
# ---------------------------------------------------------------------

def test_les_features_de_trajectoire_existent(enrichi):
    for col in ("mkt_derive_tardive", "mkt_amplitude", "mkt_volatilite",
                "mkt_n_releves", "mkt_rang_derive"):
        assert col in enrichi.columns, col
    cols = ft.colonnes_features(enrichi, avec_marche=True)
    assert "mkt_derive_tardive" in cols
    # Sans marché, AUCUNE ne doit passer : ce sont des variables de
    # marché, et le modèle `sans_marche` doit rester indépendant de lui.
    assert not any(c.startswith("mkt_")
                   for c in ft.colonnes_features(enrichi, avec_marche=False))


def test_la_trajectoire_se_calcule():
    """La dérive tardive doit refléter le mouvement, pas rester vide."""
    base = _cadre(n_courses=40)
    base["cote_t15"] = base["cote_finale"] * 2.0     # la cote a été divisée par 2
    base["cote_min"] = base["cote_finale"] * 0.9
    base["cote_max"] = base["cote_finale"] * 2.2
    base["cote_ecart_type"] = 0.3
    base["cote_n"] = 12
    df = ft.construire(base, avec_marche=True)
    assert df["mkt_derive_tardive"].notna().all()
    # ln(cote_finale / cote_t15) = ln(0.5) ≈ −0,69 : la cote a baissé.
    assert np.allclose(df["mkt_derive_tardive"], np.log(0.5))
    assert (df["mkt_amplitude"] > 0).all()


def test_pas_de_trajectoire_sans_les_colonnes():
    """
    Une base d'avant cette version n'a aucune de ces colonnes. Le piège :
    `pd.to_numeric(df.get("absente"))` rend un SCALAIRE, et la ligne
    suivante casse sur `.replace`. Le pipeline doit tourner quand même.
    """
    df = ft.construire(_cadre(n_courses=30), avec_marche=True)
    assert df["mkt_derive_tardive"].isna().all()
    assert ft.colonnes_features(df, avec_marche=True)


def test_les_conditions_d_engagement_deviennent_des_features():
    base = _cadre(n_courses=40)
    base["penetrometre"] = 3.8
    base["categorie_particularite"] = "COURSE_A_CONDITIONS"
    base["condition_age"] = "QUATRE_ANS"
    base["condition_sexe"] = "TOUS_CHEVAUX"
    base["corde"] = "CORDE_GAUCHE"
    base["categorie_statut"] = "AMATEUR"
    base["nombre_declares_partants"] = base["nombre_partants"] + 2
    df = ft.construire(base, avec_marche=False)
    cols = ft.colonnes_features(df, avec_marche=False)
    for attendu in ("c_penetrometre", "c_categorie", "c_condition_age",
                    "c_condition_sexe", "c_sens_corde", "c_declares",
                    "c_taux_non_partants"):
        assert attendu in cols, attendu
    # 2 non-partants sur 12 déclarés.
    assert np.allclose(df["c_taux_non_partants"], 2 / 12)


# ---------------------------------------------------------------------
# 4. L'abstention
# ---------------------------------------------------------------------

def _bandes(reussites, marches, n=200):
    return pd.DataFrame({
        "seuil_bas": [0.0, 0.05, 0.10, 0.15],
        "n_courses": [n] * 4,
        "reussite": reussites,
        "reussite_marche": marches,
    })


def test_le_seuil_retient_le_regime_pas_la_bande_isolee():
    """
    Une bande basse qui dépasse le marché par hasard ne doit pas fixer
    le seuil : ce qu'on cherche, c'est un régime qui tient jusqu'en haut.
    """
    b = _bandes([0.30, 0.20, 0.28, 0.35], [0.25, 0.25, 0.25, 0.25])
    assert ev.seuil_abstention(b) == 0.10


def test_aucun_seuil_quand_le_modele_ne_tient_nulle_part():
    b = _bandes([0.20, 0.21, 0.22, 0.23], [0.25, 0.26, 0.27, 0.30])
    assert ev.seuil_abstention(b) is None
    assert "aucun seuil ne tient" in ev.afficher_bandes(b, None)


def test_seuil_zero_quand_le_modele_tient_partout():
    b = _bandes([0.30, 0.31, 0.32, 0.40], [0.25, 0.25, 0.25, 0.25])
    assert ev.seuil_abstention(b) == 0.0


def test_les_bandes_trop_maigres_sont_ignorees():
    """Vingt courses ne suffisent pas à établir un régime."""
    b = _bandes([0.90, 0.10, 0.10, 0.10], [0.25, 0.25, 0.25, 0.25], n=20)
    assert ev.seuil_abstention(b) is None


def test_le_seuil_sur_des_bandes_vides():
    assert ev.seuil_abstention(pd.DataFrame()) is None
    assert "pas assez de courses" in ev.afficher_bandes(pd.DataFrame(), None)


def test_les_bandes_se_calculent_sur_un_vrai_cadre(ordinal, enrichi, decoupage):
    _, _, m_test = decoupage.masques(enrichi["heure_depart"])
    test = enrichi[m_test].copy()
    pred = ordinal.predire(test)
    test["proba"] = pred["proba"].reindex(test.index)
    test["ecart_top2"] = pred["ecart_top2"].reindex(test.index)

    b = ev.bandes_confiance(test)
    assert not b.empty
    assert b["seuil_bas"].is_monotonic_increasing
    assert (b["reussite"] >= 0).all() and (b["reussite"] <= 1).all()
    assert b["n_courses"].sum() == test["course_id"].nunique()
    texte = ev.afficher_bandes(b, ev.seuil_abstention(b))
    assert "Abstention" in texte
