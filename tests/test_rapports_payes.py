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

LA FORME, ENFIN OBSERVÉE

Elle avait d'abord été devinée — le réseau de l'environnement de
développement ne joint pas l'API PMU. La commande
`pmu.collect rapports --verifier` l'a relevée en production le
25/08/2026, et elle est recopiée telle quelle plus bas. Deux surprises,
qui auraient toutes deux faussé la mesure en silence :

  1. TOUT EST EN CENTIMES. `dividendePourUnEuro` = 190 vaut 1,90 €.
  2. TROIS CHAMPS DE DIVIDENDE COEXISTENT, dont un exprimé pour la mise
     de base (3 € sur un 2sur4). Le confondre avec le champ par euro
     triplerait le rapport.

Les tests gardent malgré tout leur volet « forme inconnue » : le jour
où le PMU changera d'unité, le symptôme doit rester « zéro ligne, et la
commande de vérification le dit », jamais un conteneur en boucle.
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

# LA CHARGE RÉELLE, relevée le 25/08/2026 sur R1C1 du 24/08 depuis le
# conteneur de production. Ce n'est plus une forme supposée : c'est la
# réponse de l'API, recopiée telle quelle.
CHARGE_REELLE = [
    {"typePari": "SIMPLE_GAGNANT", "miseBase": 200, "rembourse": False,
     "dividendeUnite": "PourUnEuro", "famillePari": "Simple",
     "rapports": [{"libelle": "Simple gagnant", "dividende": 190,
                   "dividendePourUnEuro": 190, "combinaison": "9",
                   "nombreGagnants": 29090.93,
                   "dividendePourUneMiseDeBase": 380,
                   "dividendeUnite": "PourUnEuro"}]},
    {"typePari": "SIMPLE_PLACE", "miseBase": 200, "rembourse": False,
     "rapports": [{"dividende": 140, "dividendePourUnEuro": 140,
                   "combinaison": "9", "nombreGagnants": 11743.45,
                   "dividendePourUneMiseDeBase": 280,
                   "dividendeUnite": "PourUnEuro"},
                  {"dividende": 250, "dividendePourUnEuro": 250,
                   "combinaison": "5", "dividendeUnite": "PourUnEuro"}]},
    {"typePari": "COUPLE_GAGNANT", "miseBase": 200, "rembourse": False,
     "rapports": [{"dividende": 1740, "dividendePourUnEuro": 1740,
                   "combinaison": "9-5", "nombreGagnants": 960.35,
                   "dividendePourUneMiseDeBase": 3480,
                   "dividendeUnite": "PourUnEuro"}]},
    # 2sur4 : mise de base à 3 €, et `dividende` exprimé POUR CETTE
    # MISE — pas pour un euro. C'est le piège du deuxième champ.
    {"typePari": "DEUX_SUR_QUATRE", "miseBase": 300, "rembourse": False,
     "rapports": [{"dividende": 570, "dividendePourUnEuro": 190,
                   "combinaison": "9-5", "nombreGagnants": 3038.37,
                   "dividendePourUneMiseDeBase": 570,
                   "dividendeUnite": "PourUneMiseDeBase"}]},
]

FORME_PLATE = [
    {"typePari": "SIMPLE_GAGNANT", "combinaison": "7", "rapport": 450, "miseBase": 200},
    {"typePari": "SIMPLE_PLACE", "combinaison": "7", "rapport": 180, "miseBase": 200},
]


def _par_cle(charge):
    return {(l["type_pari"], l["combinaison"]): l
            for l in parse_rapports_definitifs(charge)}


def test_les_centimes_deviennent_des_euros():
    """
    190 centimes = 1,90 € perçu pour 1 € misé. Lire 190 donnerait un
    cheval à 190 contre 1 — pour un favori à 29 000 tickets gagnants.
    L'erreur serait énorme, systématique, et jamais visible.
    """
    l = _par_cle(CHARGE_REELLE)[("SIMPLE_GAGNANT", "9")]
    assert l["rapport"] == pytest.approx(1.90)
    assert l["mise_base"] == pytest.approx(2.0)
    assert l["nombre_gagnants"] == 29090.93


def test_la_mise_de_base_ne_gonfle_pas_le_rapport():
    """
    LE piège du 2sur4 : `dividende` = 570 vaut pour une mise de 3 €,
    `dividendePourUnEuro` = 190. Prendre le premier triplerait le
    rapport. On doit retrouver 1,90 € dans les deux cas.
    """
    l = _par_cle(CHARGE_REELLE)[("DEUX_SUR_QUATRE", "9-5")]
    assert l["rapport"] == pytest.approx(1.90), "la mise de base a fuité dans le rapport"


def test_le_champ_pour_une_mise_de_base_est_converti_en_secours():
    """Si `dividendePourUnEuro` manquait, `dividendeUnite` sauve la lecture."""
    l = parse_rapports_definitifs([{
        "typePari": "DEUX_SUR_QUATRE", "miseBase": 300,
        "rapports": [{"combinaison": "9-5", "dividende": 570,
                      "dividendeUnite": "PourUneMiseDeBase"}]}])[0]
    assert l["rapport"] == pytest.approx(1.90)


def test_tous_les_types_de_paris_sont_gardes():
    cles = _par_cle(CHARGE_REELLE)
    assert ("SIMPLE_PLACE", "5") in cles
    assert ("COUPLE_GAGNANT", "9-5") in cles
    assert cles[("COUPLE_GAGNANT", "9-5")]["rapport"] == pytest.approx(17.40)


def test_la_forme_plate_se_lit_aussi():
    lignes = parse_rapports_definitifs(FORME_PLATE)
    assert len(lignes) == 2
    assert lignes[0]["rapport"] == pytest.approx(4.5)


def test_une_combinaison_de_liste_devient_une_chaine():
    """Une combinaison est une clé primaire : sa forme doit être stable."""
    l = parse_rapports_definitifs([{
        "typePari": "COUPLE_GAGNANT", "miseBase": 200,
        "rapports": [{"combinaison": [7, 3], "dividendePourUnEuro": 1870}]}])[0]
    assert l["combinaison"] == "7-3"


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


def test_le_diagnostic_garde_son_garde_fou_d_unite():
    """
    Tant que la forme n'avait pas été observée, ce fichier interdisait
    toute conversion : deviner l'unité aurait rendu le diagnostic
    circulaire. La forme est maintenant connue et la conversion est
    faite dans le parseur — mais le garde-fou reste, parce qu'il coûte
    trois lignes et qu'il attrapera un changement d'unité côté PMU.
    """
    v = ev.verifier_rapports(FausseConn(_lignes(100.0)),
                             date(2026, 1, 1), date(2026, 12, 31))
    assert v["verdict"] == "unite"


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
    """
    n gagnants dont le rapport payé vaut `ratio` × la cote.

    Le rapport est TOUJOURS ramené à 1 € misé par le parseur : la mise
    de base ne le multiplie donc pas, elle n'est qu'une information
    portée le long de la ligne.
    """
    return [(i, 7, cote, cote * ratio, mise_base) for i in range(n)]


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


def test_la_chaine_complete_sur_la_charge_reelle():
    """
    Bout en bout, avec la vraie charge : le parseur rend 1,90 €, et si
    la cote relevée valait elle aussi 1,90, le verdict doit être
    « cote déjà nette ».

    C'est le test qui compte : il enchaîne la conversion des centimes,
    l'absence de division par la mise de base, et le diagnostic. Une
    division de trop quelque part et le ratio tombe à 0,95 — soit
    exactement dans la zone « cote brute », donc un contresens complet
    et silencieux.
    """
    l = _par_cle(CHARGE_REELLE)[("SIMPLE_GAGNANT", "9")]
    lignes = [(i, 9, 1.90, l["rapport"], l["mise_base"]) for i in range(80)]
    v = ev.verifier_rapports(FausseConn(lignes), date(2026, 1, 1), date(2026, 12, 31))
    assert v["ratio_median"] == pytest.approx(1.0, abs=1e-6)
    assert v["verdict"] == "cote_nette", v


def test_la_mise_de_base_n_intervient_plus_dans_le_ratio():
    """
    Le rapport arrive DÉJÀ ramené à 1 € misé. Rediviser par la mise de
    base ferait passer une cote nette (ratio 1) pour une cote brute
    (ratio 0,5 avec une base à 2 €) — l'erreur qui inverserait la
    conclusion du projet.
    """
    v = ev.verifier_rapports(FausseConn(_lignes(1.0, mise_base=2.0)),
                             date(2026, 1, 1), date(2026, 12, 31))
    assert v["ratio_median"] == pytest.approx(1.0)
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
