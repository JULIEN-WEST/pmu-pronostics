"""
L'échelle de confiance — de ☆ à ★★★★★.

CE QUE LA NOTE DIT, ET CE QU'ELLE NE DIT PAS

Elle mesure le TRANCHANT : à quel point le modèle sépare son 1ᵉʳ choix
du 2ᵉ. Un tranchant élevé veut dire « le modèle hésite peu ». Il ne veut
pas dire « le modèle a raison », et surtout pas « il faut miser ».

C'est exactement le malentendu qu'un cadre vert peut créer. D'où la
règle qui structure ce fichier : le vert ne s'allume QUE sur un niveau
où le modèle a réellement égalé le favori du public, sur assez de
courses pour que la comparaison tienne. Le reste du temps les étoiles
s'affichent, mais en maigre, et la phrase qui les accompagne dit
explicitement que le niveau n'a pas été tenu.

Trois familles de tests :

  1. Le découpage — cinq niveaux, monotones, refusés si l'échantillon
     est trop mince.
  2. Le verdict — `fiable` est une comparaison mesurée, pas un seuil
     décoratif.
  3. La notation — `note_confiance` doit survivre à tout ce que la base
     peut lui envoyer, y compris rien.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE / "src"))

from pmu import evaluate as ev  # noqa: E402


# ---------------------------------------------------------------------
# Un jeu de test synthétique mais réaliste
# ---------------------------------------------------------------------

def _fenetre(n_courses=400, graine=7, avec_marche=True):
    """
    Une fenêtre de test notée. Le modèle est délibérément MEILLEUR
    quand il est tranchant : c'est la propriété que l'échelle doit
    retrouver toute seule.
    """
    rng = np.random.default_rng(graine)
    lignes = []
    for i in range(n_courses):
        k = int(rng.integers(8, 15))
        # Un écart tiré large, pour peupler les cinq quintiles.
        ecart = float(rng.uniform(0.005, 0.30))
        p1 = 0.12 + ecart
        reste = 1.0 - p1
        autres = rng.dirichlet(np.ones(k - 1)) * reste
        probas = np.concatenate([[p1], autres])
        # Le favori gagne d'autant plus souvent que l'écart est grand.
        gagnant = 0 if rng.random() < (0.18 + 1.2 * ecart) else int(rng.integers(1, k))
        for j in range(k):
            ligne = {
                "course_id": i, "proba": float(probas[j]),
                "ecart_top2": ecart,
                "y_gagnant": int(j == gagnant),
            }
            if avec_marche:
                # Un marché honnête : corrélé au vrai gagnant, mais pas parfait.
                base = 0.55 if j == gagnant else 0.0
                ligne["mkt_proba_implicite"] = float(base + rng.uniform(0, 0.5))
            lignes.append(ligne)
    return pd.DataFrame(lignes)


# ---------------------------------------------------------------------
# 1. Le découpage
# ---------------------------------------------------------------------

def test_cinq_niveaux_et_quatre_seuils():
    ech = ev.echelle_confiance(_fenetre())
    assert len(ech["seuils"]) == 4, "cinq niveaux se délimitent par quatre bornes"
    assert sorted(ech["seuils"]) == ech["seuils"], "les bornes doivent être croissantes"
    assert [x["note"] for x in ech["niveaux"]] == [1, 2, 3, 4, 5]


def test_les_niveaux_sont_a_peu_pres_equilibres():
    """
    Des quintiles : chaque niveau doit peser environ un cinquième. Un
    niveau à trois courses donnerait un taux affiché de 0 % ou 100 %
    sans aucun contenu.
    """
    ech = ev.echelle_confiance(_fenetre(n_courses=500))
    effectifs = [x["n_courses"] for x in ech["niveaux"]]
    assert min(effectifs) >= 0.6 * (500 / 5), f"niveau trop mince : {effectifs}"


def test_un_echantillon_trop_mince_est_refuse():
    """
    Cinq niveaux réclament au minimum 5 × 40 courses. En dessous, on ne
    rend pas une échelle bancale : on n'en rend aucune. Une note fausse
    est pire qu'une note absente, parce qu'elle a l'air d'une mesure.
    """
    ech = ev.echelle_confiance(_fenetre(n_courses=120))
    assert ech == {"seuils": [], "niveaux": []}


@pytest.mark.parametrize("df", [
    pd.DataFrame(),
    pd.DataFrame({"course_id": [1], "proba": [0.3]}),  # pas d'ecart_top2
])
def test_une_fenetre_inutilisable_ne_leve_pas(df):
    assert ev.echelle_confiance(df) == {"seuils": [], "niveaux": []}


def test_le_tranchant_suit_la_reussite():
    """
    LA propriété attendue. Sur ce jeu construit exprès, le taux de
    réussite du favori doit croître avec la note — sinon l'échelle
    classe du bruit et le cadre vert ne veut rien dire.
    """
    ech = ev.echelle_confiance(_fenetre(n_courses=600))
    taux = [x["taux"] for x in sorted(ech["niveaux"], key=lambda v: v["note"])]
    assert taux[-1] > taux[0], f"pas de progression : {taux}"


# ---------------------------------------------------------------------
# 2. Le verdict — la partie qui protège de la surinterprétation
# ---------------------------------------------------------------------

def test_l_intervalle_encadre_le_taux():
    for x in ev.echelle_confiance(_fenetre())["niveaux"]:
        bas, haut = x["ic95"]
        assert bas <= x["taux"] <= haut, x


def test_sans_marche_aucun_niveau_n_est_declare_fiable():
    """
    Sans cote, il n'y a rien à comparer. Le vert doit rester éteint :
    « je n'ai pas pu vérifier » ne devient jamais « c'est bon ».
    """
    ech = ev.echelle_confiance(_fenetre(avec_marche=False))
    assert ech["niveaux"], "l'échelle doit exister quand même"
    assert not any(x["fiable"] for x in ech["niveaux"])
    assert all(x["taux_marche"] is None for x in ech["niveaux"])


def test_un_modele_battu_par_le_marche_n_allume_rien():
    """
    Le cas qui compte vraiment, parce que c'est celui de la production :
    le modèle ne bat pas le marché. Aucun niveau ne doit alors être
    encadré en vert, quel que soit son tranchant.
    """
    df = _fenetre(n_courses=500)
    # Un marché parfait : il désigne toujours le gagnant.
    df["mkt_proba_implicite"] = np.where(df["y_gagnant"] == 1, 0.99, 0.01)
    ech = ev.echelle_confiance(df)
    assert not any(x["fiable"] for x in ech["niveaux"]), (
        "un marché parfait ne peut être égalé : rien ne doit passer au vert"
    )
    assert all(x["taux_marche"] == 1.0 for x in ech["niveaux"])


def test_un_modele_qui_tient_le_niveau_est_reconnu():
    """La réciproque : si le modèle égale un marché nul, le vert s'allume."""
    df = _fenetre(n_courses=500)
    df["mkt_proba_implicite"] = np.where(df["y_gagnant"] == 1, 0.01, 0.99)
    ech = ev.echelle_confiance(df)
    assert all(x["fiable"] for x in ech["niveaux"])


def test_le_taux_marche_porte_sur_les_memes_courses():
    """
    Comparer le modèle d'un niveau au marché mesuré sur TOUTES les
    courses serait truqué : chaque niveau doit être confronté à son
    propre lot.

    Pour le prouver sans ambiguïté, on fabrique un marché qui réussit
    partout SAUF sur les courses les plus tranchées. Si la comparaison
    était globale, les cinq niveaux afficheraient le même taux marché.
    """
    df = _fenetre(n_courses=500)
    ech = ev.echelle_confiance(df)
    haut = ech["seuils"][-1]

    tranchee = df["ecart_top2"] > haut
    # Marché parfait ailleurs, marché aveugle sur les courses tranchées.
    df["mkt_proba_implicite"] = np.where(
        df["y_gagnant"] == 1,
        np.where(tranchee, 0.01, 0.99),
        np.where(tranchee, 0.99, 0.01),
    )
    niveaux = {x["note"]: x["taux_marche"] for x in ev.echelle_confiance(df)["niveaux"]}
    assert niveaux[5] == 0.0, f"le niveau 5 doit isoler le marché aveugle : {niveaux}"
    assert all(niveaux[n] == 1.0 for n in (1, 2, 3, 4)), niveaux


# ---------------------------------------------------------------------
# 3. La notation d'une course
# ---------------------------------------------------------------------

@pytest.mark.parametrize("ecart, attendu", [
    (0.001, 1), (0.02, 1),
    (0.03, 2), (0.06, 3), (0.10, 4), (0.20, 5),
    (100.0, 5),
])
def test_la_note_suit_les_seuils(ecart, attendu):
    seuils = [0.02, 0.05, 0.09, 0.15]
    assert ev.note_confiance(ecart, seuils) == attendu


def test_la_note_est_toujours_dans_la_plage():
    seuils = [0.02, 0.05, 0.09, 0.15]
    for e in np.linspace(-1, 2, 200):
        assert 1 <= ev.note_confiance(float(e), seuils) <= 5


@pytest.mark.parametrize("ecart", [None, float("nan"), "abc", {}, []])
def test_un_ecart_illisible_retombe_sur_la_note_la_plus_basse(ecart):
    """
    Une donnée manquante ne doit JAMAIS produire une note élevée : le
    doute se lit comme peu de certitude, jamais comme beaucoup.
    """
    assert ev.note_confiance(ecart, [0.02, 0.05, 0.09, 0.15]) == 1


def test_sans_echelle_toutes_les_courses_sont_a_une_etoile():
    """
    Avant le premier réentraînement porteur de l'échelle, il n'y a pas
    de seuils. La vue doit rester lisible, sans course mise en avant.
    """
    assert ev.note_confiance(0.42, []) == 1
    assert ev.note_confiance(0.42, None) == 1


def test_la_note_du_niveau_correspond_a_l_echelle():
    """
    Cohérence bout en bout : renoter la fenêtre avec `note_confiance`
    doit redonner exactement le découpage de `echelle_confiance`.
    """
    df = _fenetre(n_courses=500)
    ech = ev.echelle_confiance(df)
    favoris = df.loc[df.groupby("course_id")["proba"].idxmax()]
    refaites = [ev.note_confiance(e, ech["seuils"]) for e in favoris["ecart_top2"]]
    compte = pd.Series(refaites).value_counts().sort_index().tolist()
    attendu = [x["n_courses"] for x in sorted(ech["niveaux"], key=lambda v: v["note"])]
    assert compte == attendu


# ---------------------------------------------------------------------
# 4. L'affichage
# ---------------------------------------------------------------------

def test_le_rapport_dit_le_verdict_en_toutes_lettres():
    texte = ev.afficher_echelle(ev.echelle_confiance(_fenetre(n_courses=500)))
    assert "Échelle de confiance" in texte
    assert "★" in texte
    assert "marché" in texte
    assert "pas une promesse" in texte, (
        "le rapport doit refuser explicitement la lecture « miser ici »"
    )


def test_une_echelle_vide_s_affiche_sans_planter():
    texte = ev.afficher_echelle({"seuils": [], "niveaux": []})
    assert "pas assez de courses" in texte
    assert ev.afficher_echelle({}).strip()
