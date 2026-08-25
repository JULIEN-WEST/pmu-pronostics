"""
La couche d'explication.

Deux exigences, et la seconde est la plus facile à rater.

  1. Le motif doit être JUSTE : si on fabrique un univers où une seule
     famille décide de l'arrivée, l'ablation doit désigner CETTE
     famille, pas une autre.
  2. Le motif doit être HONNÊTE : quand une donnée manque, il faut le
     dire, pas afficher un chiffre par défaut. Un « 0 % de réussite »
     inventé pour un cheval inconnu est pire que pas de motif du tout —
     il donne une fausse assurance.
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

from pmu import explain as ex, features as ft  # noqa: E402
from pmu.train import Decoupage, ModeleParDiscipline, ModelePmu  # noqa: E402


# ---------------------------------------------------------------------
# 1. Mise en forme
# ---------------------------------------------------------------------

@pytest.mark.parametrize("ms, attendu", [
    (72800, "1'12\"8"),
    (60000, "1'00\"0"),
    (119970, "2'00\"0"),      # arrondi qui franchit la minute
    (72960, "1'13\"0"),       # arrondi qui franchit la seconde
    (81500, "1'21\"5"),
])
def test_reduction_lisible(ms, attendu):
    assert ex.reduction_lisible(ms) == attendu


def test_reduction_lisible_refuse_l_absurde():
    """Mieux vaut ne rien afficher qu'un chrono impossible."""
    assert ex.reduction_lisible(None) is None
    assert ex.reduction_lisible(np.nan) is None
    assert ex.reduction_lisible(0) is None
    assert ex.reduction_lisible(-5) is None
    assert ex.reduction_lisible(10**9) is None


# ---------------------------------------------------------------------
# 2. L'ablation désigne-t-elle la bonne famille ?
# ---------------------------------------------------------------------

def _cadre(n_courses=600, decideur="palmares", graine=5) -> pd.DataFrame:
    """
    Univers où UNE SEULE famille décide de l'arrivée.

    `decideur='palmares'` → les gains font le gagnant.
    `decideur='vitesse'`  → le chrono passé fait le gagnant.

    C'est le seul montage qui permette de vérifier une attribution :
    on connaît la bonne réponse d'avance.
    """
    rng = np.random.default_rng(graine)
    lignes, chrono_par_cheval = [], {}
    t0 = pd.Timestamp("2025-01-01", tz="UTC")
    for i in range(n_courses):
        n = 10
        chevaux = rng.choice(200, size=n, replace=False)
        heure = t0 + pd.Timedelta(hours=3 * i)
        gains = rng.gamma(2.0, 4000.0, size=n)
        for c in chevaux:
            chrono_par_cheval.setdefault(int(c), float(rng.uniform(70000, 85000)))
        chronos = np.array([chrono_par_cheval[int(c)] for c in chevaux])

        score = gains / 4000.0 if decideur == "palmares" else (85000 - chronos) / 5000.0
        bruit = rng.gumbel(0, 1.0, size=n)
        ordre = np.argsort(-(score / (score.std() or 1) + bruit))
        place = np.empty(n, dtype=int)
        place[ordre] = np.arange(1, n + 1)

        for j in range(n):
            lignes.append({
                "course_id": i, "heure_depart": heure, "date_reunion": heure.date(),
                "num_pmu": j + 1, "id_cheval": int(chevaux[j]),
                "id_driver": int(rng.integers(1, 40)),
                "id_entraineur": int(rng.integers(1, 20)),
                "id_proprietaire": int(rng.integers(1, 30)),
                "nom_pere": f"P{int(chevaux[j]) % 12}",
                "nom_pere_mere": f"PM{int(chevaux[j]) % 9}",
                "discipline": "ATTELE", "specialite": None, "distance": 2700,
                "etat_terrain": "BON", "hippodrome_code": "VIN",
                "nombre_partants": n, "montant_prix": 20000.0,
                "age": 6, "sexe": "MALES", "place_corde": j + 1,
                "handicap_poids": 56.0, "handicap_distance": 0,
                "deferre": None, "oeilleres": None, "musique": "1a 2a 3a",
                "nombre_courses": 20, "nombre_victoires": 3, "nombre_places": 8,
                "gains_carriere": float(gains[j]),
                "gains_annee_en_cours": float(gains[j]) / 3,
                "driver": f"M. D{int(rng.integers(1, 40))}",
                "entraineur": None,
                "statut": "PARTANT", "ordre_arrivee": int(place[j]),
                # Chrono propre au cheval : le passé prédit donc le présent.
                "reduction_km_ms": float(chronos[j] + rng.normal(0, 400)),
                "temps_officiel_ms": 200000,
                "distance_cheval_precedent": "2 L",
                "cote_finale": 8.0, "cote_ouverture": 8.0,
                "source": "direct",
            })
    return pd.DataFrame(lignes)


def _modele_et_course(decideur: str):
    df = ft.construire(_cadre(decideur=decideur), avec_marche=True)
    d = Decoupage.par_proportions(df["heure_depart"], 0.6, 0.2)
    m = ModelePmu(cible="y_gagnant").entrainer(df, d)
    _, _, m_test = d.masques(df["heure_depart"])
    test = df[m_test]
    return m, test


@pytest.fixture(scope="module")
def palmares():
    return _modele_et_course("palmares")


@pytest.fixture(scope="module")
def vitesse():
    return _modele_et_course("vitesse")


def test_l_ablation_designe_la_famille_qui_decide(palmares):
    """
    Les gains font le gagnant : « palmarès » doit dominer l'attribution
    sur le favori de chaque course.
    """
    modele, test = palmares
    c = ex.contributions(modele, test)
    proba = modele.predire(test)["proba"].reindex(test.index)
    favoris = proba.groupby(test["course_id"]).idxmax()
    gagnantes = c.loc[favoris].abs().idxmax(axis=1).value_counts(normalize=True)
    assert gagnantes.index[0] == "palmares", (
        f"famille dominante attendue « palmares », obtenue « {gagnantes.index[0] }» :\n"
        f"{gagnantes.head()}"
    )
    assert gagnantes.iloc[0] > 0.5


def test_l_ablation_suit_le_chrono_quand_c_est_lui_qui_decide(vitesse):
    """Le contre-test : changez la loi, l'attribution doit suivre."""
    modele, test = vitesse
    c = ex.contributions(modele, test)
    proba = modele.predire(test)["proba"].reindex(test.index)
    favoris = proba.groupby(test["course_id"]).idxmax()
    gagnantes = c.loc[favoris].abs().idxmax(axis=1).value_counts(normalize=True)
    assert gagnantes.index[0] == "vitesse", (
        f"famille dominante attendue « vitesse », obtenue « {gagnantes.index[0]} » :\n"
        f"{gagnantes.head()}"
    )


def test_le_signe_est_coherent_avec_le_classement(palmares):
    """
    Sur la famille décisive, le mieux noté doit recevoir une
    contribution positive et le moins bien noté une négative.
    """
    modele, test = palmares
    proba = modele.predire(test)["proba"].reindex(test.index)
    c = ex.contributions(modele, test, groupes=["palmares"])["palmares"]
    d = pd.DataFrame({"p": proba, "c": c, "course": test["course_id"]})
    premiers = d.loc[d.groupby("course")["p"].idxmax()]
    derniers = d.loc[d.groupby("course")["p"].idxmin()]
    assert (premiers["c"] > 0).mean() > 0.75
    assert (derniers["c"] < 0).mean() > 0.75


def test_les_contributions_sont_alignees_sur_l_entree(palmares):
    modele, test = palmares
    c = ex.contributions(modele, test)
    assert list(c.index) == list(test.index)
    assert len(c.columns) >= 4


# ---------------------------------------------------------------------
# 3. L'honnêteté des faits
# ---------------------------------------------------------------------

def test_les_faits_ne_fabriquent_rien_quand_tout_manque():
    """
    Un cheval dont on ne sait rien ne doit produire AUCUN chiffre — ni
    « 0 % », ni « nan », ni « None ». Le seul message acceptable est
    qu'on ne sait pas.
    """
    vide = pd.Series({"course_id": 1, "num_pmu": 1, "h_cheval_place_n": 0.0})
    f = ex.faits(vide)
    assert "historique" in f
    assert any("aucune course antérieure" in t for t in f["historique"])
    plat = " ".join(t for v in f.values() for t in v)
    for interdit in ("nan", "None", "%.1f", "0 %"):
        assert interdit not in plat, f"« {interdit} » apparaît dans : {plat}"


def test_les_faits_citent_les_effectifs(palmares):
    """
    Un taux sans son effectif est un piège : 100 % sur une course et
    100 % sur deux cents ne se lisent pas pareil.
    """
    modele, test = palmares
    une = test[test["course_id"] == test["course_id"].iloc[0]]
    f = ex.faits(une.iloc[0])
    hist = " ".join(f.get("historique", []))
    if "placé" in hist:
        assert "courses connues" in " ".join(f["historique"])


def test_faits_supporte_une_ligne_incomplete():
    """Une base d'avant l'étape 3 n'a ni chrono ni écurie."""
    ligne = pd.Series({
        "course_id": 1, "num_pmu": 3, "mus_n": 5, "mus_moy": 3.2,
        "h_cheval_place_n": 12.0, "h_cheval_place": 0.34,
    })
    f = ex.faits(ligne)
    assert "forme" in f and "historique" in f
    assert "vitesse" not in f, "un chrono a été inventé"


def test_expliquer_produit_des_motifs_lisibles(palmares):
    modele, test = palmares
    une = test[test["course_id"] == test["course_id"].iloc[-1]].copy()
    res = ex.expliquer(modele, une)
    assert len(res) == len(une)
    for (cid, num), e in res.items():
        assert isinstance(cid, int) and isinstance(num, int)
        for m in e["motifs"]:
            assert m["sens"] in ("+", "−")
            assert m["titre"] and m["icone"].startswith("mdi:")
            assert abs(m["poids"]) >= 0.004, "un motif sous le seuil de bruit"
        assert len(e["motifs"]) <= 3


def test_expliquer_sur_un_cadre_vide():
    assert ex.expliquer(None, pd.DataFrame()) == {}


def test_expliquer_marche_avec_le_modele_par_discipline():
    """
    `ModeleParDiscipline` n'expose pas `.colonnes` : l'explication doit
    aller les chercher sur le modèle global, sinon aucune famille n'est
    reconnue et tous les motifs disparaissent silencieusement.
    """
    df = ft.construire(_cadre(n_courses=400), avec_marche=True)
    d = Decoupage.par_proportions(df["heure_depart"], 0.6, 0.2)
    m = ModeleParDiscipline().entrainer(df, d)
    _, _, m_test = d.masques(df["heure_depart"])
    test = df[m_test]
    une = test[test["course_id"] == test["course_id"].iloc[-1]].copy()
    c = ex.contributions(m, une)
    assert len(c.columns) >= 4, "aucune famille reconnue sur le modèle scindé"
    assert c.abs().to_numpy().sum() > 0
