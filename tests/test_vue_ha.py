"""
Les templates Jinja de la vue Home Assistant.

POURQUOI CE FICHIER EXISTE

Une carte markdown de Home Assistant a planté en production sur
`UndefinedError: 'dict object' has no attribute 'motifs'`. La cause :
le message MQTT retenu datait d'une version antérieure, où la clé
`motifs` n'existait pas — et un attribut manquant lève, il ne rend pas
une chaîne vide.

C'est le défaut typique d'un template : il n'est jamais exécuté avant
d'être collé dans Home Assistant, donc jamais testé. Ces tests le
rendent ici, avec les DEUX formes de charge utile — l'ancienne et la
nouvelle — et vérifient qu'aucune ne lève.

Ils ne prouvent pas que le rendu est joli. Ils prouvent qu'il a lieu.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")
jinja2 = pytest.importorskip("jinja2")

RACINE = Path(__file__).resolve().parent.parent
VUES = [RACINE / "homeassistant" / "lovelace_vue_SEULE.yaml",
        RACINE / "homeassistant" / "lovelace_vue_pronostics.yaml"]


# ---------------------------------------------------------------------
# Un Home Assistant de poche
# ---------------------------------------------------------------------

def _environnement() -> jinja2.Environment:
    env = jinja2.Environment()
    env.filters["as_datetime"] = lambda v: (
        v if isinstance(v, dt.datetime) else dt.datetime.fromisoformat(str(v)))
    env.filters["as_local"] = lambda d: d
    env.filters["float"] = lambda v, d=0.0: (d if v is None else float(v))
    env.filters["round"] = lambda v, p=0, m="common": (
        round(float(v), p) if p else round(float(v)))
    return env


def _cartes_markdown(cartes):
    for c in cartes:
        if c.get("type") == "markdown":
            yield c
        if "card" in c:
            yield from _cartes_markdown([c["card"]])


def _toutes_les_cartes(chemin: Path):
    charge = yaml.safe_load(chemin.read_text(encoding="utf-8"))
    vue = charge[0] if isinstance(charge, list) else charge
    for section in vue["sections"]:
        yield from _cartes_markdown(section["cards"])


# -- les trois formes de charge utile qu'on rencontre en vrai ---------

def _selection_ancienne():
    """Message MQTT d'avant la version 0.6 : aucune clé `motifs`."""
    return [
        {"num": 5, "cheval": "JULIE DU NORD", "proba": 0.218,
         "cote": None, "valeur": None, "arrivee": 1},
        {"num": 10, "cheval": "KORONA DES CHAMPS", "proba": 0.155,
         "cote": None, "valeur": None},
    ]


def _selection_nouvelle():
    s = _selection_ancienne()
    s[0]["motifs"] = [
        {"titre": "Chrono", "sens": "+",
         "details": ["meilleur chrono 1'12\"8 au km", "90 % du lot fait moins bien"]},
        {"titre": "Lignée", "sens": "−", "details": ["produits placés 28 %"]},
    ]
    s[1]["motifs"] = []
    return s


def _selection_mutilee():
    """
    Motifs présents mais incomplets — ce qui arrive si la charge a été
    tronquée pour tenir dans un attribut d'entité.
    """
    s = _selection_ancienne()
    s[0]["motifs"] = [{"titre": "Chrono"}, {"sens": "+"}, {}]
    return s


def _contexte(selection, *, prochaine=True, programme=True, courses="40"):
    pro = {"code": "R2C1", "hippodrome": "HIPPODROME DE LA CAPELLE",
           "libelle": "PRIX DES AMATEURS", "discipline": "ATTELE",
           "distance": 2750, "partants": None,
           "depart": "2026-08-25T11:52:00", "confiance": 0.062,
           "selection": selection}
    prog = [dict(pro, code=c, arrivee_connue=(i < 2))
            for i, c in enumerate(["R2C1", "R3C1", "R2C2"])] if programme else []
    attrs = {"prochaine": pro if prochaine else None, "programme": prog,
             "date": "2026-08-25", "amorcage": None, "message": ""}
    return {
        "state_attr": lambda e, a: attrs.get(a),
        "states": lambda e: courses if e == "sensor.pmu_courses" else "0.4",
    }


CAS = {
    "ancienne": _selection_ancienne,
    "nouvelle": _selection_nouvelle,
    "mutilée": _selection_mutilee,
}


# ---------------------------------------------------------------------

@pytest.mark.parametrize("chemin", VUES, ids=lambda p: p.name)
def test_le_yaml_survit_a_la_reserialisation(chemin):
    """
    Home Assistant réécrit le YAML quand on enregistre la vue. Un bloc
    littéral mal choisi peut voir ses retours à la ligne changer, et une
    table markdown s'effondre alors en une seule ligne.
    """
    d = yaml.safe_load(chemin.read_text(encoding="utf-8"))
    assert yaml.safe_load(yaml.safe_dump(d, allow_unicode=True)) == d


def test_les_deux_fichiers_decrivent_la_meme_vue():
    a = yaml.safe_load(VUES[0].read_text(encoding="utf-8"))
    b = yaml.safe_load(VUES[1].read_text(encoding="utf-8"))
    assert isinstance(b, list) and len(b) == 1
    assert a == b[0], "la forme liste et la forme seule ont divergé"


@pytest.mark.parametrize("cas", list(CAS), ids=list(CAS))
@pytest.mark.parametrize("chemin", VUES, ids=lambda p: p.name)
def test_toutes_les_cartes_se_rendent(chemin, cas):
    """
    LE test qui manquait. Un template n'est jamais exécuté avant d'être
    collé dans Home Assistant : celui-ci l'exécute ici.
    """
    env = _environnement()
    ctx = _contexte(CAS[cas]())
    for carte in _toutes_les_cartes(chemin):
        try:
            rendu = env.from_string(carte["content"]).render(**ctx)
        except jinja2.UndefinedError as exc:
            pytest.fail(f"{chemin.name} — charge « {cas} » : {exc}")
        assert "None" not in rendu, f"un « None » est rendu tel quel :\n{rendu}"


@pytest.mark.parametrize("chemin", VUES, ids=lambda p: p.name)
def test_les_cartes_se_rendent_sans_aucune_donnee(chemin):
    """Avant la première collecte, tous les attributs sont vides."""
    env = _environnement()
    ctx = _contexte([], prochaine=False, programme=False, courses="0")
    for carte in _toutes_les_cartes(chemin):
        env.from_string(carte["content"]).render(**ctx)


def test_le_tableau_du_programme_ne_deborde_pas():
    """
    Quatre colonnes au maximum. Au-delà, la dernière sort de l'écran
    dans une colonne de tableau de bord — c'est arrivé avec six.
    """
    env = _environnement()
    ctx = _contexte(_selection_nouvelle())
    for carte in _toutes_les_cartes(VUES[0]):
        rendu = env.from_string(carte["content"]).render(**ctx)
        for ligne in rendu.splitlines():
            if ligne.startswith("|") and ligne.endswith("|"):
                colonnes = ligne.strip("|").split("|")
                assert len(colonnes) <= 4, f"{len(colonnes)} colonnes : {ligne}"


def test_l_iframe_est_en_chemin_relatif():
    """
    Une URL http:// est refusée par le navigateur quand Home Assistant
    est servi en HTTPS, et le cadre reste blanc. Le chemin doit rester
    relatif, servi par Home Assistant lui-même.
    """
    for chemin in VUES:
        charge = yaml.safe_load(chemin.read_text(encoding="utf-8"))
        vue = charge[0] if isinstance(charge, list) else charge
        iframes = [c for s in vue["sections"] for c in s["cards"]
                   if c.get("type") == "iframe"]
        assert iframes, f"{chemin.name} : plus aucune iframe"
        for f in iframes:
            assert f["url"].startswith("/"), (
                f"{chemin.name} : l'iframe pointe sur {f['url']} — "
                "une URL absolue en http sera bloquée en HTTPS")


def test_le_package_declare_le_rapatriement():
    """
    Sans cette automatisation, le fichier n'arrive jamais dans `www` et
    l'iframe affiche une 404. Les deux vont ensemble.
    """
    pkg = yaml.safe_load(
        (RACINE / "homeassistant" / "package_pmu.yaml").read_text(encoding="utf-8"))
    autos = pkg["automation"]
    rapatriement = [a for a in autos if a["id"] == "pmu_rapatrier_vue"]
    assert rapatriement, "l'automatisation de rapatriement a disparu"
    actions = rapatriement[0]["actions"]
    assert any(a.get("action") == "downloader.download_file" for a in actions)
    d = actions[0]["data"]
    assert d["subdir"] == "pmu" and d["filename"] == "vue.html" and d["overwrite"]
    assert d["url"].endswith("/vue")
