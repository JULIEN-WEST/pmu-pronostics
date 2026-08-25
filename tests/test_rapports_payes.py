"""
Les rapports payés — trancher la question du prélèvement.

LA QUESTION

La simulation de rentabilité multiplie la cote par (1 − 15 %). Ça n'est
juste que si `mkt_cote` est un rapport BRUT. Dans un pari mutuel, le
rapport affiché est en général déjà NET — le prélèvement est retiré de
la masse avant répartition. Si c'est le cas, tous les ROI publiés sont
sous-estimés d'environ 18 %, et la conclusion « pas rentable » repose
sur une erreur d'unité, pas sur une mesure.

LE JUGE

Ce qui a été réellement payé. Pour un cheval GAGNANT, le rapport
définitif du Simple Gagnant est par définition la somme perçue pour
1 € misé. Son rapport à la cote relevée répond à la question sans
modèle et sans hypothèse.

CE QUE J'AI DÛ DEVINER, ET COMMENT ON S'EN PROTÈGE

La forme de la réponse `rapports-definitifs` n'a PAS pu être observée
en direct : le réseau de l'environnement de développement ne joint pas
l'API. Le parseur accepte donc les deux formes plausibles et ne lève
jamais. Les tests ci-dessous vérifient les deux formes ET le fait
qu'une forme inconnue rende une liste vide plutôt qu'une exception —
parce que si j'ai deviné faux, le symptôme doit être « zéro ligne, et
la commande de vérification le dit », pas un conteneur en boucle
d'erreurs.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pytest

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE / "src"))

from pmu import evaluate as ev  # noqa: E402
from pmu.normalize import parse_rapports_definitifs  # noqa: E402


# ---------------------------------------------------------------------
# 1. Le parseur, et les deux formes plausibles
# ---------------------------------------------------------------------

FORME_GROUPEE = [
    {"typePari": "SIMPLE_GAGNANT", "miseBase": 1,
     "rapports": [{"combinaison": [7], "dividendePourUnEuro": 4.5,
                   "nombreGagnants": 1284.0}]},
    {"typePari": "SIMPLE_PLACE", "miseBase": 1,
     "rapports": [{"combinaison": [7], "dividendePourUnEuro": 1.8},
                  {"combinaison": [3], "dividendePourUnEuro": 2.4},
                  {"combinaison": [11], "dividendePourUnEuro": 3.1}]},
    {"typePari": "COUPLE_GAGNANT", "miseBase": 1,
     "rapports": [{"combinaison": [7, 3], "dividendePourUnEuro": 18.7}]},
]

FORME_PLATE = [
    {"typePari": "SIMPLE_GAGNANT", "combinaison": "7", "rapport": 4.5, "miseBase": 1},
    {"typePari": "SIMPLE_PLACE", "combinaison": "7", "rapport": 1.8, "miseBase": 1},
]


def test_la_forme_groupee_se_lit():
    lignes = parse_rapports_definitifs(FORME_GROUPEE)
    par_cle = {(l["type_pari"], l["combinaison"]): l for l in lignes}
    assert par_cle[("SIMPLE_GAGNANT", "7")]["rapport"] == 4.5
    assert par_cle[("SIMPLE_GAGNANT", "7")]["nombre_gagnants"] == 1284.0
    assert par_cle[("SIMPLE_PLACE", "3")]["rapport"] == 2.4


def test_la_forme_plate_se_lit_aussi():
    lignes = parse_rapports_definitifs(FORME_PLATE)
    assert len(lignes) == 2
    assert lignes[0]["rapport"] == 4.5


def test_une_combinaison_multiple_devient_une_chaine():
    """« 7-3 » doit être lisible et stable : c'est une clé primaire."""
    lignes = parse_rapports_definitifs(FORME_GROUPEE)
    couples = [l for l in lignes if l["type_pari"] == "COUPLE_GAGNANT"]
    assert couples[0]["combinaison"] == "7-3"


def test_l_enveloppe_dict_est_acceptee():
    assert len(parse_rapports_definitifs({"rapports": FORME_PLATE})) == 2


@pytest.mark.parametrize("charge", [
    None, {}, [], "", 42, {"rapports": None}, [None, 3, "x"],
    [{"typePari": "X"}],                       # aucun montant
    [{"rapport": 4.5}],                        # aucune combinaison
    [{"typePari": "X", "rapports": "pas une liste"}],
])
def test_une_forme_inconnue_rend_une_liste_vide(charge):
    """
    LE point important, vu que la forme réelle n'a pas été observée : le
    symptôme d'une mauvaise hypothèse doit être « rien collecté », pas
    un conteneur qui plante en boucle.
    """
    assert parse_rapports_definitifs(charge) == []


def test_le_rapport_n_est_jamais_converti_a_l_aveugle():
    """
    450 (centimes) ne doit PAS être divisé par 100 ici. L'unité est
    déduite par le diagnostic, en comparant aux cotes ; convertir dans
    le parseur rendrait ce diagnostic circulaire — il vérifierait sa
    propre hypothèse.
    """
    lignes = parse_rapports_definitifs(
        [{"typePari": "SIMPLE_GAGNANT", "combinaison": "7", "rapport": 450}])
    assert lignes[0]["rapport"] == 450


# ---------------------------------------------------------------------
# 2. Le diagnostic — une base factice, mais un vrai verdict
# ---------------------------------------------------------------------

class FausseConn:
    """Rend les lignes qu'on lui a données, quelle que soit la requête."""

    def __init__(self, lignes):
        self.lignes = lignes

    def execute(self, sql, params=None):
        self._r = self.lignes
        return self

    def fetchall(self):
        return self._r

    def rollback(self):
        pass


def _lignes(ratio, n=120, cote=6.0, mise_base=1.0):
    """n gagnants dont le rapport payé vaut `ratio` × la cote."""
    return [(i, 7, cote, cote * ratio * mise_base, mise_base) for i in range(n)]


def test_une_cote_deja_nette_est_reconnue():
    """
    Le cas qui changerait la lecture du projet : rapport payé = cote.
    La simulation retire alors le prélèvement une SECONDE fois.
    """
    v = ev.verifier_rapports(FausseConn(_lignes(1.0)), date(2026, 1, 1), date(2026, 12, 31))
    assert v["verdict"] == "cote_nette"
    assert v["correction_roi"] == pytest.approx(1 / 0.85, rel=1e-3)
    assert "SOUS-ESTIME" in v["message"]


def test_une_cote_brute_est_reconnue():
    """L'autre cas : la simulation actuelle est juste, rien à corriger."""
    v = ev.verifier_rapports(FausseConn(_lignes(0.85)), date(2026, 1, 1), date(2026, 12, 31))
    assert v["verdict"] == "cote_brute"
    assert v["correction_roi"] == 1.0


def test_un_rapport_en_centimes_est_signale_comme_tel():
    """Une question d'unité ne doit jamais être lue comme une question d'économie."""
    v = ev.verifier_rapports(FausseConn(_lignes(100.0)), date(2026, 1, 1), date(2026, 12, 31))
    assert v["verdict"] == "unite"
    assert v["correction_roi"] is None


def test_un_ratio_inattendu_ne_conclut_rien():
    """
    0,6 n'est ni 1 ni 0,85. Le bon comportement est de refuser de
    trancher — pas de choisir le voisin le plus proche.
    """
    v = ev.verifier_rapports(FausseConn(_lignes(0.60)), date(2026, 1, 1), date(2026, 12, 31))
    assert v["verdict"] == "inattendu"
    assert v["correction_roi"] is None


def test_la_mise_de_base_est_neutralisee():
    """
    Une mise de base de 2 € double mécaniquement le rapport. Sans la
    diviser, un pari à 2 € ressemblerait exactement à une cote brute
    lue deux fois trop grande.
    """
    v = ev.verifier_rapports(FausseConn(_lignes(1.0, mise_base=2.0)),
                             date(2026, 1, 1), date(2026, 12, 31))
    assert v["verdict"] == "cote_nette", v


def test_un_echantillon_mince_ne_tranche_pas():
    """
    Cette question ne se tranche qu'une fois. Vingt gagnants ne
    suffisent pas, et un verdict prématuré serait recopié partout.
    """
    v = ev.verifier_rapports(FausseConn(_lignes(1.0, n=20)),
                             date(2026, 1, 1), date(2026, 12, 31))
    assert v["verdict"] == "insuffisant"
    assert "pmu.collect rapports" in v["message"]


def test_une_table_absente_ne_leve_pas():
    class Cassee(FausseConn):
        def execute(self, sql, params=None):
            raise RuntimeError('relation "rapport_definitif" does not exist')

    v = ev.verifier_rapports(Cassee([]), date(2026, 1, 1), date(2026, 12, 31))
    assert v["verdict"] == "indisponible"


def test_les_lignes_absurdes_sont_ecartees():
    """Cote ≤ 1 ou rapport nul : impossibles, donc écartés avant la médiane."""
    lignes = _lignes(1.0, n=100) + [(999, 7, 0.5, 4.0, 1.0), (998, 7, 6.0, 0.0, 1.0)]
    v = ev.verifier_rapports(FausseConn(lignes), date(2026, 1, 1), date(2026, 12, 31))
    assert v["n"] == 100
    assert v["verdict"] == "cote_nette"


def test_la_mediane_resiste_a_quelques_outsiders():
    """
    Un rapport à 300 € sur un outsider ne doit pas emporter le verdict.
    C'est pour ça qu'on lit une médiane et pas une moyenne.
    """
    lignes = _lignes(1.0, n=100) + [(900 + i, 7, 3.0, 300.0, 1.0) for i in range(8)]
    v = ev.verifier_rapports(FausseConn(lignes), date(2026, 1, 1), date(2026, 12, 31))
    assert v["verdict"] == "cote_nette"


# ---------------------------------------------------------------------
# 3. L'affichage
# ---------------------------------------------------------------------

def test_le_texte_dit_le_verdict_et_la_correction():
    v = ev.verifier_rapports(FausseConn(_lignes(1.0)), date(2026, 1, 1), date(2026, 12, 31))
    texte = ev.afficher_rapports(v)
    assert "rapport payé" in texte
    assert "1.1765" in texte or "multiplier" in texte


def test_un_diagnostic_impossible_s_affiche_quand_meme():
    for v in ({}, {"verdict": "insuffisant", "n": 3, "message": "pas assez"},
              {"verdict": "indisponible", "message": "table absente"}):
        assert ev.afficher_rapports(v).strip()
