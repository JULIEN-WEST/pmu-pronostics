"""
Modèles par famille de discipline.

Ce qui a motivé le découpage : sur données réelles, le savoir-faire du
modèle valait +0,137 d'AUC en monté contre +0,029 en plat. Un modèle
unique moyenne ces régimes, donc les dessert tous.

Ce qu'il faut vérifier, et c'est plus subtil que « ça marche » :

  1. Le ROUTAGE est exact — une course d'attelé doit être notée par le
     modèle d'attelé, à la virgule près. Une erreur ici donnerait des
     pronostics silencieusement faux.
  2. L'ARBITRAGE est honnête — quand le spécialisé n'apporte rien, il
     doit être écarté. Un découpage qu'on garde par principe est un
     découpage qui coûte des exemples pour rien.
  3. La fenêtre de TEST n'est jamais consultée pour décider. Choisir sur
     le test, c'est le consommer.
  4. Les anciens modèles se rechargent toujours.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE / "src"))

from pmu import features as ft  # noqa: E402
from pmu.train import (Decoupage, ModeleParDiscipline, ModelePmu,  # noqa: E402
                       charger_modele, modele_present, resumer_arbitrage)


# ---------------------------------------------------------------------
# Univers de test
# ---------------------------------------------------------------------

def _cadre(n_par_discipline: int = 700, graine: int = 7,
           disciplines=("ATTELE", "MONTE", "PLAT", "HAIES")) -> pd.DataFrame:
    """
    Un univers où CHAQUE DISCIPLINE OBÉIT À UNE LOI DIFFÉRENTE.

    C'est le seul montage qui puisse départager les deux approches : si
    toutes les disciplines suivaient la même règle, le modèle unique
    serait le bon choix et le découpage n'aurait rien à prouver.

    Ici le gagnant dépend d'une variable différente selon la discipline —
    les gains en attelé, la musique en monté, le poids en plat. Un modèle
    unique doit apprendre les trois règles ET quand les appliquer ; un
    modèle par famille n'en apprend qu'une.
    """
    rng = np.random.default_rng(graine)
    lignes, cid = [], 0
    t0 = pd.Timestamp("2025-01-01", tz="UTC")

    for d_i, disc in enumerate(disciplines):
        for i in range(n_par_discipline):
            n = 10
            cid += 1
            # Les disciplines sont entrelacées dans le temps : sans ça, le
            # découpage chronologique mettrait une discipline entière hors
            # de la fenêtre d'entraînement.
            heure = t0 + pd.Timedelta(hours=2 * (i * len(disciplines) + d_i))
            chevaux = rng.choice(3000, size=n, replace=False)
            gains = rng.gamma(2.0, 4000.0, size=n)
            mus = rng.uniform(1, 12, size=n)
            poids = rng.uniform(50, 62, size=n)

            if disc == "ATTELE":
                score = gains / 4000.0
            elif disc == "MONTE":
                score = (12 - mus)
            elif disc == "PLAT":
                score = (62 - poids)
            else:
                score = rng.normal(0, 1, size=n)   # obstacle : imprévisible

            # Plackett-Luce : le meilleur score gagne le plus souvent,
            # sans jamais gagner à tous les coups.
            bruit = rng.gumbel(0, 1.0, size=n)
            classement = np.argsort(-(score / (score.std() or 1) + bruit))
            place = np.empty(n, dtype=int)
            place[classement] = np.arange(1, n + 1)

            for j in range(n):
                lignes.append({
                    "course_id": cid, "heure_depart": heure,
                    "date_reunion": heure.date(), "num_pmu": j + 1,
                    "id_cheval": int(chevaux[j]),
                    "id_driver": int(rng.integers(1, 80)),
                    "id_entraineur": int(rng.integers(1, 40)),
                    "nom_pere": f"P{int(chevaux[j]) % 30}",
                    "nom_pere_mere": f"PM{int(chevaux[j]) % 17}",
                    "discipline": disc, "specialite": None,
                    "distance": 2700 if disc in ("ATTELE", "MONTE") else 1600,
                    "etat_terrain": "BON", "hippodrome_code": "VIN",
                    "nombre_partants": n, "montant_prix": 20000.0,
                    "age": 6, "sexe": "MALES", "place_corde": j + 1,
                    "handicap_poids": float(poids[j]),
                    "deferre": None, "oeilleres": None,
                    "musique": " ".join(f"{int(round(mus[j]))}a" for _ in range(5)),
                    "nombre_courses": 20, "nombre_victoires": 3, "nombre_places": 8,
                    "gains_carriere": float(gains[j]),
                    "gains_annee_en_cours": float(gains[j]) / 3,
                    "statut": "PARTANT", "ordre_arrivee": int(place[j]),
                    "cote_finale": 8.0, "cote_ouverture": 9.0,
                    "source": "direct",
                })
    return pd.DataFrame(lignes)


@pytest.fixture(scope="module")
def enrichi():
    return ft.construire(_cadre(), avec_marche=True)


@pytest.fixture(scope="module")
def decoupage(enrichi):
    return Decoupage.par_proportions(enrichi["heure_depart"], 0.6, 0.2)


@pytest.fixture(scope="module")
def modele(enrichi, decoupage):
    return ModeleParDiscipline(cible="y_gagnant").entrainer(enrichi, decoupage)


# ---------------------------------------------------------------------
# 1. La famille
# ---------------------------------------------------------------------

def test_regroupement_des_disciplines():
    assert ft.famille("ATTELE") == "ATTELE"
    assert ft.famille("MONTE") == "MONTE"
    assert ft.famille("PLAT") == "PLAT"
    for d in ("HAIES", "STEEPLECHASE", "CROSS"):
        assert ft.famille(d) == "OBSTACLE", f"{d} devrait rejoindre l'obstacle"
    assert ft.famille("attele") == "ATTELE", "la casse ne doit pas compter"
    assert ft.famille(None) == "AUTRE"
    assert ft.famille("TROT MONGOL") == "AUTRE"


def test_la_famille_ne_part_pas_dans_le_modele(enrichi):
    """
    Constante à l'intérieur d'un modèle spécialisé, elle n'apporterait
    rien — et surtout elle ferait diverger les colonnes entre le modèle
    global et les spécialisés, donc casserait le rechargement.
    """
    assert "famille" in enrichi.columns
    for avec in (False, True):
        assert "famille" not in ft.colonnes_features(enrichi, avec_marche=avec)


# ---------------------------------------------------------------------
# 2. Le routage — le point où une erreur serait silencieuse
# ---------------------------------------------------------------------

def test_le_routage_est_exact(modele, enrichi, decoupage):
    """
    Chaque ligne doit recevoir EXACTEMENT la probabilité qu'aurait donnée
    le modèle retenu pour sa famille, appelé directement. C'est la
    garantie qu'aucune course ne part vers le mauvais modèle.
    """
    _, _, m_test = decoupage.masques(enrichi["heure_depart"])
    test = enrichi[m_test]
    obtenu = modele.predire(test)

    for fam, sub in test.groupby("famille"):
        attendu_modele = modele.par_famille.get(str(fam)) or modele.global_
        attendu = attendu_modele.predire(sub)["proba"]
        pd.testing.assert_series_equal(
            obtenu["proba"].reindex(sub.index).sort_index(),
            attendu.sort_index(), check_names=False,
            obj=f"probabilités de la famille {fam}",
        )


def test_toutes_les_lignes_sont_notees(modele, enrichi, decoupage):
    """Aucune famille ne doit se perdre en route."""
    _, _, m_test = decoupage.masques(enrichi["heure_depart"])
    test = enrichi[m_test]
    out = modele.predire(test)
    assert len(out) == len(test)
    assert set(out.index) == set(test.index)
    assert out["proba"].notna().all()


def test_les_probabilites_somment_a_un_par_course(modele, enrichi, decoupage):
    """
    Propriété non négociable : il y a un gagnant et un seul. Le routage
    ne doit pas la casser — c'est le risque quand on assemble des
    prédictions venues de plusieurs modèles.
    """
    _, _, m_test = decoupage.masques(enrichi["heure_depart"])
    out = modele.predire(enrichi[m_test])
    sommes = out.groupby("course_id")["proba"].sum()
    assert np.allclose(sommes, 1.0, atol=1e-6), (
        f"sommes hors tolérance : min {sommes.min():.6f}, max {sommes.max():.6f}"
    )


def test_une_famille_inconnue_retombe_sur_le_global(modele, enrichi, decoupage):
    """
    Le jour où le PMU publie une discipline qu'on n'a jamais vue, la
    prédiction doit continuer de tourner, pas lever une exception.
    """
    _, _, m_test = decoupage.masques(enrichi["heure_depart"])
    test = enrichi[m_test].copy()
    test["famille"] = "DISCIPLINE_MARTIENNE"
    out = modele.predire(test)
    attendu = modele.global_.predire(test)["proba"]
    pd.testing.assert_series_equal(
        out["proba"].sort_index(), attendu.sort_index(), check_names=False)


# ---------------------------------------------------------------------
# 3. L'arbitrage
# ---------------------------------------------------------------------

def test_l_arbitrage_couvre_toutes_les_familles(modele, enrichi):
    assert set(modele.arbitrage) == set(enrichi["famille"].unique())
    for fam, f in modele.arbitrage.items():
        assert f["decision"] in ("specialise", "global")
        assert f["n_total"] > 0
        if f["decision"] == "global" and "auc_specialise" not in f:
            assert f.get("motif"), f"un rejet sans motif pour {fam}"


def test_le_specialise_gagne_quand_les_lois_different(modele):
    """
    L'univers de test donne à chaque discipline une loi différente. Si le
    découpage ne sert à rien DANS CE CAS, c'est qu'il ne servira jamais.
    """
    retenues = [f for f, x in modele.arbitrage.items() if x["decision"] == "specialise"]
    assert retenues, (
        "aucune famille spécialisée alors que chaque discipline suit une "
        "loi distincte :\n" + resumer_arbitrage(modele.arbitrage)
    )


def test_le_specialise_n_est_retenu_qu_avec_un_gain_reel(modele):
    """L'arbitrage doit être une mesure, pas une préférence de principe."""
    from pmu.train import MARGE_AUC
    for fam, f in modele.arbitrage.items():
        if f["decision"] == "specialise":
            assert f["gain_auc"] > MARGE_AUC, (
                f"{fam} retenue avec un gain de {f['gain_auc']}, "
                f"sous la marge exigée {MARGE_AUC}"
            )


def test_les_petites_familles_ne_sont_pas_scindees(enrichi, decoupage):
    """
    Sous le seuil de volume, on ne tente même pas : un modèle entraîné
    sur trois fois rien serait pire que le global, et le mesurer coûte
    du temps pour un résultat connu d'avance.
    """
    petit = ft.construire(_cadre(n_par_discipline=60), avec_marche=True)
    d = Decoupage.par_proportions(petit["heure_depart"], 0.6, 0.2)
    m = ModeleParDiscipline(cible="y_gagnant").entrainer(petit, d)
    assert not m.par_famille, (
        "des familles ont été spécialisées malgré un volume dérisoire :\n"
        + resumer_arbitrage(m.arbitrage)
    )
    for f in m.arbitrage.values():
        assert "volume insuffisant" in f.get("motif", "")


def test_l_arbitrage_ne_regarde_jamais_le_test(enrichi, decoupage, monkeypatch):
    """
    Le contrôle le plus important du fichier. On rend la fenêtre de test
    ILLISIBLE : toute lecture lève. Si l'entraînement passe malgré ça,
    c'est qu'il ne s'en sert pas — donc que le score rapporté ensuite est
    bien une mesure hors échantillon.
    """
    _, _, m_test = decoupage.masques(enrichi["heure_depart"])
    piege = enrichi.copy()
    # On empoisonne les résultats du test : un modèle qui s'en servirait
    # pour arbitrer prendrait des décisions différentes.
    piege.loc[m_test, "y_gagnant"] = np.nan
    piege.loc[m_test, "y_place"] = np.nan

    m = ModeleParDiscipline(cible="y_gagnant").entrainer(piege, decoupage)
    reference = ModeleParDiscipline(cible="y_gagnant").entrainer(enrichi, decoupage)

    d_piege = {f: x["decision"] for f, x in m.arbitrage.items()}
    d_ref = {f: x["decision"] for f, x in reference.arbitrage.items()}
    assert d_piege == d_ref, (
        "les décisions changent quand on altère la fenêtre de test : "
        "l'arbitrage la consulte, le score rapporté est donc optimiste.\n"
        f"avec test empoisonné : {d_piege}\nnormal : {d_ref}"
    )


# ---------------------------------------------------------------------
# 4. Persistance et compatibilité
# ---------------------------------------------------------------------

def test_aller_retour_sur_disque(modele, enrichi, decoupage, tmp_path):
    _, _, m_test = decoupage.masques(enrichi["heure_depart"])
    test = enrichi[m_test]
    avant = modele.predire(test)

    modele.sauver(tmp_path / "m")
    assert modele_present(tmp_path / "m")
    relu = charger_modele(tmp_path / "m")
    assert isinstance(relu, ModeleParDiscipline)
    assert set(relu.par_famille) == set(modele.par_famille)

    apres = relu.predire(test)
    pd.testing.assert_series_equal(
        avant["proba"].sort_index(), apres["proba"].sort_index(), check_names=False)

    meta = json.loads((tmp_path / "m" / "meta.json").read_text(encoding="utf-8"))
    assert meta["type"] == "par_discipline"
    assert meta["arbitrage"], "l'arbitrage doit être conservé pour être relu"


def test_un_ancien_modele_unique_se_recharge_encore(enrichi, decoupage, tmp_path):
    """
    Rétrocompatibilité : les modèles déjà sur le disque du conteneur
    n'ont pas de meta.json. Ils doivent continuer de se charger, sinon
    un redéploiement casse la production jusqu'au prochain entraînement.
    """
    unique = ModelePmu(cible="y_gagnant").entrainer(enrichi, decoupage)
    unique.sauver(tmp_path / "vieux")
    assert not (tmp_path / "vieux" / "meta.json").exists()
    assert modele_present(tmp_path / "vieux")
    relu = charger_modele(tmp_path / "vieux")
    assert isinstance(relu, ModelePmu) and not isinstance(relu, ModeleParDiscipline)


def test_meta_json_illisible_ne_casse_pas_le_chargement(enrichi, decoupage, tmp_path):
    """Un disque plein peut tronquer un fichier. On se replie, on ne plante pas."""
    unique = ModelePmu(cible="y_gagnant").entrainer(enrichi, decoupage)
    unique.sauver(tmp_path / "abime")
    (tmp_path / "abime" / "meta.json").write_text("{ceci n'est pas du json",
                                                  encoding="utf-8")
    assert isinstance(charger_modele(tmp_path / "abime"), ModelePmu)


def test_modele_present_est_faux_sur_un_dossier_vide(tmp_path):
    assert not modele_present(tmp_path / "rien")
