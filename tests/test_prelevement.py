"""
Le prélèvement : de l'ambigu au décisif.

CE QUI A ÉCHOUÉ

Première tentative : comparer, sur les chevaux gagnants, le rapport
payé à la cote relevée avant le départ. Résultat en production sur
284 gagnants : ratio médian 0,894, avec Q1 0,68 et Q3 1,16.

Ni 1,00 (cote nette) ni 0,85 (cote brute). Et surtout un étalement de
±30 %, qui a livré l'explication : la cote pré-départ N'EST PAS le
rapport payé — l'argent des dernières minutes déplace le prix. Pire,
la mesure ne portait que sur les gagnants, et un gagnant est
précisément un cheval sur lequel l'argent est venu. Biais de sélection
garanti, dans le sens observé.

CE QUI TRANCHE

L'identité du pari mutuel. Avec P la masse, t le prélèvement et S_i les
mises sur le partant i :

    rapport_i = P (1 − t) / S_i    donc    Σ 1/rapport_i = 1 / (1 − t)

La somme des probabilités implicites d'une course vaut donc 1/(1−t) si
la cote est déjà nette, et 1,00 si elle est brute. Aucune arrivée,
aucun modèle, aucune sélection : toutes les courses, tous les partants.

ET ENSUITE

Une fois la question tranchée, elle n'a plus à l'être : la simulation
de rentabilité paie au rapport RÉELLEMENT versé. Plus de prélèvement
supposé, plus de cote pré-départ prise pour un rapport. La deuxième
moitié de ce fichier vérifie que ce basculement est correct — et
notamment qu'une course dont le rapport est inconnu est ÉCARTÉE, et
non comptée perdante.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE / "src"))

from pmu import evaluate as ev  # noqa: E402

DEBUT, FIN = date(2026, 1, 1), date(2026, 12, 31)


# ---------------------------------------------------------------------
# 1. La surcote
# ---------------------------------------------------------------------

class Conn:
    def __init__(self, lignes):
        self.lignes = lignes

    def execute(self, sql, params=None):
        return self

    def fetchall(self):
        return self.lignes

    def rollback(self):
        pass


def _courses(surcote, n=200, partants=12, bruit=0.0, graine=3):
    """n courses dont la somme des probabilités implicites vaut `surcote`."""
    rng = np.random.default_rng(graine)
    ecarts = rng.normal(0, bruit, n) if bruit else np.zeros(n)
    return [(i, float(surcote + ecarts[i]), partants) for i in range(n)]


def test_une_cote_nette_est_reconnue():
    """
    Σ 1/cote = 1,176 = 1/(1−15 %). Le prélèvement est DÉJÀ retiré : la
    simulation qui le retire une seconde fois sous-estime le retour.
    """
    v = ev.surcote(Conn(_courses(1.1765)), DEBUT, FIN)
    assert v["verdict"] == "cote_nette"
    assert v["prelevement_mesure"] == pytest.approx(0.15, abs=0.005)
    assert "DÉJÀ NETTE" in v["message"]


def test_une_cote_brute_est_reconnue():
    """Σ 1/cote = 1,00 : appliquer le prélèvement est correct."""
    v = ev.surcote(Conn(_courses(1.0)), DEBUT, FIN)
    assert v["verdict"] == "cote_brute"
    assert v["prelevement_mesure"] is None


def test_le_prelevement_reel_est_deduit_et_non_suppose():
    """
    Si le prélèvement réel est de 12 % et non des 15 % supposés, la
    mesure doit le DIRE — pas le forcer dans la case la plus proche.
    C'est tout l'intérêt de mesurer plutôt que de paramétrer.
    """
    v = ev.surcote(Conn(_courses(1 / 0.88)), DEBUT, FIN)
    assert v["verdict"] == "cote_nette"
    assert v["prelevement_mesure"] == pytest.approx(0.12, abs=0.005)
    assert "12" in v["message"]


def test_une_somme_sous_un_ne_conclut_rien():
    """
    0,90 est impossible dans un mutuel : la somme ne peut pas descendre
    sous 1. Le bon comportement est de refuser de trancher.
    """
    v = ev.surcote(Conn(_courses(0.90)), DEBUT, FIN)
    assert v["verdict"] == "inattendu"


def test_le_bruit_ne_fait_pas_derailler_la_mediane():
    """
    Les cotes sont arrondies et quelques courses ont des non-partants.
    Un écart-type de 4 points ne doit pas changer le verdict.
    """
    v = ev.surcote(Conn(_courses(1.1765, n=400, bruit=0.04)), DEBUT, FIN)
    assert v["verdict"] == "cote_nette"


def test_les_courses_absurdes_sont_ecartees():
    lignes = _courses(1.1765, n=200) + [
        (900, 12.0, 8),    # une cote à 0,1 quelque part
        (901, 0.02, 8),    # une seule cote relevée sur la course
    ]
    v = ev.surcote(Conn(lignes), DEBUT, FIN)
    assert v["n"] == 200


def test_trop_peu_de_courses_ne_tranche_pas():
    v = ev.surcote(Conn(_courses(1.1765, n=30)), DEBUT, FIN)
    assert v["verdict"] == "insuffisant"


def test_une_table_absente_ne_leve_pas():
    class Cassee(Conn):
        def execute(self, sql, params=None):
            raise RuntimeError("relation v_cote_finale does not exist")

    assert ev.surcote(Cassee([]), DEBUT, FIN)["verdict"] == "indisponible"


def test_l_affichage_dit_le_prelevement_mesure():
    texte = ev.afficher_surcote(ev.surcote(Conn(_courses(1.1765)), DEBUT, FIN))
    assert "somme 1/cote" in texte
    assert "prélèvement déduit" in texte
    assert ev.afficher_surcote({}).strip()


# ---------------------------------------------------------------------
# 2. La rentabilité payée au tarif réel
# ---------------------------------------------------------------------

def _paris(n_courses=200, cote=5.0, rapport=None, taux=0.20, graine=11):
    """
    Un partant misable par course. `rapport` None = aucun rapport connu.
    """
    rng = np.random.default_rng(graine)
    gagne = rng.random(n_courses) < taux
    d = pd.DataFrame({
        "course_id": np.arange(n_courses),
        "num_pmu": 1,
        "proba": 0.5,          # valeur = 0,5 × 5 − 1 = 1,5, au-dessus de tout seuil
        "mkt_cote": cote,
        "y_gagnant": gagne.astype(int),
    })
    if rapport is not None:
        d["rapport_reel"] = rapport
    return d


def test_sans_rapport_reel_la_simulation_estime_comme_avant():
    d = _paris()
    sim = ev.simulation(d, seuils_valeur=(0.0,)).iloc[0]
    assert not sim["mesure"]
    # retour = cote × (1 − 15 %) par gagnant
    attendu = d["y_gagnant"].sum() * 5.0 * 0.85
    assert sim["retour"] == pytest.approx(attendu, abs=0.02)


def test_avec_rapport_reel_on_paie_ce_qui_a_ete_paye():
    """
    Le rapport EST le montant perçu pour 1 € misé : aucun prélèvement à
    lui appliquer. L'y appliquer quand même serait le double comptage
    qu'on cherche justement à éliminer.
    """
    d = _paris(rapport=4.70)
    sim = ev.simulation(d, seuils_valeur=(0.0,), col_reel="rapport_reel").iloc[0]
    assert sim["mesure"]
    attendu = d["y_gagnant"].sum() * 4.70
    assert sim["retour"] == pytest.approx(attendu, abs=0.02)


def test_une_course_sans_rapport_est_ecartee_pas_comptee_perdante():
    """
    LE piège de cette mesure. Un rapport manquant sur une course
    gagnée serait lu comme un retour nul — une défaite imaginaire. Sur
    350 courses documentées parmi des milliers, l'erreur écraserait
    complètement le ROI.
    """
    d = _paris(n_courses=200, rapport=4.70)
    # La moitié des courses n'a pas de rapport collecté.
    d.loc[d["course_id"] >= 100, "rapport_reel"] = np.nan

    sim = ev.simulation(d, seuils_valeur=(0.0,), col_reel="rapport_reel").iloc[0]
    assert sim["n_paris"] == 100, "les courses sans rapport doivent sortir du lot"
    assert sim["courses_ecartees"] == 100

    connues = d[d["course_id"] < 100]
    assert sim["retour"] == pytest.approx(connues["y_gagnant"].sum() * 4.70, abs=0.02)
    assert sim["n_gagnants"] == int(connues["y_gagnant"].sum())


def test_le_roi_reel_differe_du_roi_estime_dans_le_bon_sens():
    """
    Cote 5,00, rapport réellement payé 4,70. L'estimation payait
    5,00 × 0,85 = 4,25, soit MOINS que la réalité : elle sous-estimait.
    """
    d = _paris(rapport=4.70)
    estime = ev.simulation(d, seuils_valeur=(0.0,)).iloc[0]["roi_pct"]
    mesure = ev.simulation(d, seuils_valeur=(0.0,),
                           col_reel="rapport_reel").iloc[0]["roi_pct"]
    assert mesure > estime
    assert (1 + mesure / 100) / (1 + estime / 100) == pytest.approx(4.70 / 4.25, rel=1e-3)


def test_l_ecart_type_suit_la_mesure_utilisee():
    """
    Un ROI sans son écart-type n'est pas lisible. Il doit être calculé
    sur les gains RÉELS, pas sur les gains estimés — sinon on habille
    une mesure d'une incertitude qui n'est pas la sienne.
    """
    d = _paris(rapport=9.40)   # rapport très supérieur à la cote : variance plus forte
    e = ev.simulation(d, seuils_valeur=(0.0,)).iloc[0]["roi_ecart_type_pct"]
    m = ev.simulation(d, seuils_valeur=(0.0,),
                      col_reel="rapport_reel").iloc[0]["roi_ecart_type_pct"]
    assert m > e


def test_le_rapport_expose_les_deux_lectures():
    """
    On ne remplace pas l'estimation par la mesure : on les affiche côte
    à côte. C'est l'écart entre les deux qui dit de combien on se
    trompait, et dans quel sens.
    """
    d = _paris(rapport=4.70)
    d["discipline"] = "ATTELE"
    rap = ev.rapport(d)
    assert rap["rentabilite"], "l'estimation doit rester"
    assert rap["rentabilite_reelle"], "la mesure doit s'ajouter"

    texte = ev.afficher(rap)
    assert "Rentabilité mesurée" in texte
    assert "rapports réellement payés" in texte
    assert "retour ESTIMÉ" in texte
    assert "2σ" in texte, "le rappel d'interprétation doit accompagner le chiffre"


def test_sans_rapports_collectes_le_rapport_ne_montre_qu_une_lecture():
    """Une base d'avant la 1.6 n'a aucun rapport : rien ne doit changer."""
    rap = ev.rapport(_paris())
    assert "rentabilite_reelle" not in rap
    assert "Rentabilité mesurée" not in ev.afficher(rap)


def test_le_rapport_reste_encodable_en_json():
    """
    Pandas rend des `np.bool_` et des `np.int64` que `json.dumps`
    refuse. Un rapport illisible en JSON casserait l'endpoint au moment
    exact où on vient voir si le modèle marche — le même défaut que le
    NaN de `/bilan`, qui avait rendu 500 en production.
    """
    import json

    d = _paris(rapport=4.70)
    d["discipline"] = "ATTELE"
    json.dumps(ev.rapport(d))     # ne doit pas lever
