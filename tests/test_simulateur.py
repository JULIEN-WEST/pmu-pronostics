"""
Étalonnage du simulateur.

Un simulateur trop généreux fait croire que la méthode marche. Ces tests
le maintiennent aligné sur ce qu'on observe en courses réelles :

  - le favori du public gagne à peu près une fois sur trois ;
  - miser aveuglément sur lui est PERDANT (le prélèvement) ;
  - le marché ne connaît pas l'arrivée à l'avance.

Le troisième point est le vrai garde-fou : la première version du
simulateur construisait la cote à partir du score BRUITÉ, donc d'un
marché qui avait déjà vu la course. Le favori gagnait alors 82 % du temps
et le pari sur lui rapportait +72 %. Ce test empêche la rechute.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from simulateur import generer  # noqa: E402


@pytest.fixture(scope="module")
def courses():
    return generer(n_courses=1500, n_chevaux=2000)


@pytest.fixture(scope="module")
def favoris(courses):
    df = courses.copy()
    df["y"] = (df["ordre_arrivee"] == 1).astype(int)
    return df.loc[df.groupby("course_id")["cote_finale"].idxmin()]


def test_le_favori_gagne_environ_une_fois_sur_trois(favoris):
    taux = favoris["y"].mean()
    assert 0.25 <= taux <= 0.42, (
        f"le favori gagne {taux:.1%} — invraisemblable. "
        "Régler `echelle` : plus haut = courses trop prévisibles."
    )


def test_miser_sur_le_favori_est_perdant(favoris):
    """
    Dans un système mutuel, le prélèvement rend la stratégie naïve perdante.
    Un simulateur où elle gagne fabrique de l'argent qui n'existe pas.
    """
    roi = (favoris["y"] * favoris["cote_finale"]).sum() / len(favoris) - 1
    assert -0.40 < roi < -0.05, f"ROI du favori = {roi:+.1%}, hors de tout réalisme"


def test_le_marche_ne_connait_pas_larrivee(courses):
    """
    Le test qui a rattrapé le bug d'origine. Si la cote était construite à
    partir du résultat, la corrélation entre probabilité implicite et
    victoire serait énorme. Elle doit rester modeste.
    """
    df = courses.copy()
    df["y"] = (df["ordre_arrivee"] == 1).astype(int)
    df["implicite"] = 1.0 / df["cote_finale"]
    corr = df["implicite"].corr(df["y"])
    assert corr < 0.55, (
        f"corrélation cote/victoire = {corr:.2f} : le marché voit l'arrivée. "
        "Vérifier que la cote dérive des scores LATENTS, pas des scores bruités."
    )


def test_les_cotes_ressemblent_a_de_vraies_cotes(favoris, courses):
    assert 1.8 <= favoris["cote_finale"].median() <= 4.5
    assert courses["cote_finale"].min() >= 1.1


def test_le_palmares_declare_ne_contient_pas_la_course_en_cours(courses):
    """
    Un cheval qui prend son départ pour la première fois doit afficher
    0 course et 0 victoire — pas 1.
    """
    premiers = courses.sort_values("heure_depart").groupby("id_cheval").head(1)
    assert (premiers["nombre_courses"] == 0).all()
    assert (premiers["nombre_victoires"] == 0).all()
    assert premiers["musique"].isna().all()


def test_gagnant_unique_par_course(courses):
    gagnants = courses[courses["ordre_arrivee"] == 1].groupby("course_id").size()
    assert (gagnants == 1).all()
