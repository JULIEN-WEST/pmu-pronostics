"""
JSON PMU → lignes prêtes pour la base.

Tout ici est défensif : l'API n'a pas de contrat, les clés apparaissent et
disparaissent. Une clé manquante donne None, jamais une exception.

Les montants du PMU sont en CENTIMES d'euro (gains, allocations) et les
temps en MILLISECONDES. On conserve les millisecondes telles quelles
(entiers, pas d'arrondi) et on convertit les centimes en euros une seule
fois, ici, pour ne plus jamais avoir à y penser en aval.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import date, datetime, timezone
from typing import Any

# ---------------------------------------------------------------------
# Normalisation de libellés
# ---------------------------------------------------------------------

_PUNCT = re.compile(r"[^\w\s]", flags=re.UNICODE)
_SPACES = re.compile(r"\s+")


def strip_accents(text: str) -> str:
    return "".join(
        ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch)
    )


def norm_label(value: Any) -> str | None:
    """
    Clé de jointure stable pour un libellé libre.
    'M. Barzalona' et 'M.BARZALONA' doivent tomber sur la même personne.
    """
    if value is None:
        return None
    text = strip_accents(str(value)).upper()
    text = _PUNCT.sub(" ", text)
    text = _SPACES.sub(" ", text).strip()
    return text or None


norm_person = norm_label
norm_horse = norm_label


# ---------------------------------------------------------------------
# Types primitifs
# ---------------------------------------------------------------------

def ms_to_dt(value: Any) -> datetime | None:
    """Le PMU horodate en millisecondes epoch UTC."""
    if value in (None, "", 0):
        return None
    try:
        return datetime.fromtimestamp(int(value) / 1000, tz=timezone.utc)
    except (TypeError, ValueError, OSError, OverflowError):
        return None


def ms_to_date(value: Any) -> date | None:
    dt = ms_to_dt(value)
    return dt.date() if dt else None


def cents_to_eur(value: Any) -> float | None:
    """Gains et allocations arrivent en centimes."""
    if value is None:
        return None
    try:
        return round(int(value) / 100.0, 2)
    except (TypeError, ValueError):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None


def as_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def dig(obj: Any, *path: str, default: Any = None) -> Any:
    """Accès imbriqué tolérant : dig(p, 'gainsParticipant', 'gainsCarriere')."""
    cur = obj
    for key in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
        if cur is None:
            return default
    return cur


# ---------------------------------------------------------------------
# La musique
# ---------------------------------------------------------------------

# '1a 2a 0a Da 3m (25) 1a' — chiffre = place, lettre = discipline,
# 0 = hors des 10 premiers, D = disqualifié, T = tombé, A = arrêté,
# R = rétrogradé, (25) = changement de millésime.
_MUSIQUE_ITEM = re.compile(r"(\d{1,2}|[DTARN])\s*([a-zA-Z])?", flags=re.ASCII)

_DISCIPLINE_LETTRE = {
    "a": "ATTELE",
    "m": "MONTE",
    "p": "PLAT",
    "h": "HAIES",
    "s": "STEEPLE",
    "c": "CROSS",
    "o": "OBSTACLE",
}


def parse_musique(musique: Any) -> list[dict]:
    """
    Renvoie la musique décomposée, du plus récent au plus ancien.
    Chaque élément : {place: int|None, incident: str|None, discipline: str|None}

    On ne fabrique PAS de score agrégé ici. Un « score de musique » est un
    agrégat, et les agrégats se calculent dans features.py avec une borne
    temporelle — sinon on se retrouve à noter un cheval avec sa propre
    course du jour.
    """
    if not musique or not isinstance(musique, str):
        return []
    out: list[dict] = []
    for token in musique.replace("(", " (").split():
        if token.startswith("("):  # marqueur d'année, pas une performance
            continue
        m = _MUSIQUE_ITEM.fullmatch(token.strip())
        if not m:
            continue
        raw, lettre = m.group(1), (m.group(2) or "").lower()
        entry: dict[str, Any] = {
            "place": None,
            "incident": None,
            "discipline": _DISCIPLINE_LETTRE.get(lettre),
        }
        if raw.isdigit():
            place = int(raw)
            # '0' signifie « au-delà de la 10e place », pas « place zéro ».
            entry["place"] = place if place > 0 else None
            entry["incident"] = None if place > 0 else "NON_PLACE"
        else:
            entry["incident"] = {
                "D": "DISQUALIFIE",
                "T": "TOMBE",
                "A": "ARRETE",
                "R": "RETROGRADE",
                "N": "NON_PARTANT",
            }.get(raw.upper())
        out.append(entry)
    return out


# ---------------------------------------------------------------------
# Programme → réunions / courses
# ---------------------------------------------------------------------

def parse_reunion(reunion: dict) -> dict:
    return {
        "date_reunion": ms_to_date(reunion.get("dateReunion")),
        "num_officiel": as_int(reunion.get("numOfficiel")),
        "hippodrome_code": dig(reunion, "hippodrome", "code"),
        "nature": reunion.get("nature"),
        "audience": reunion.get("audience"),
        "statut": reunion.get("statut"),
        "pays_code": dig(reunion, "pays", "code"),
        "meteo": reunion.get("meteo"),
    }


def parse_hippodrome(reunion: dict) -> dict | None:
    h = reunion.get("hippodrome")
    if not isinstance(h, dict) or not h.get("code"):
        return None
    return {
        "code": h.get("code"),
        "libelle_court": h.get("libelleCourt"),
        "libelle_long": h.get("libelleLong"),
        "pays_code": dig(reunion, "pays", "code"),
        "pays_libelle": dig(reunion, "pays", "libelle"),
    }


def parse_course(course: dict, date_reunion: date | None) -> dict:
    ordre = course.get("ordreArrivee")
    # L'état du terrain se cache à deux endroits selon la discipline :
    # champ direct en trot, sous-objet pénétromètre en galop.
    penetro = course.get("penetrometre")
    penetro = penetro if isinstance(penetro, dict) else {}
    etat_terrain = course.get("etatTerrain") or penetro.get("intitule")

    return {
        "date_reunion": date_reunion,
        "num_reunion": as_int(course.get("numReunion")),
        "num_ordre": as_int(course.get("numOrdre")),
        "libelle": course.get("libelle"),
        "libelle_court": course.get("libelleCourt"),
        "discipline": course.get("discipline"),
        "specialite": course.get("specialite"),
        "categorie_particularite": course.get("categorieParticularite"),
        "categorie_statut": course.get("categorieStatut"),
        "conditions": course.get("conditions"),
        "condition_age": course.get("conditionAge"),
        "condition_sexe": course.get("conditionSexe"),
        "distance": as_int(course.get("distance")),
        "distance_unit": course.get("distanceUnit"),
        "corde": course.get("corde"),
        "depart_type": course.get("departType") or course.get("typeDepart"),
        "montant_prix": cents_to_eur(course.get("montantPrix")),
        "nombre_declares_partants": as_int(course.get("nombreDeclaresPartants")),
        "nombre_partants": as_int(course.get("nombrePartants")),
        "etat_terrain": etat_terrain,
        "penetrometre": as_float(penetro.get("valeurMesure")),
        "heure_depart": ms_to_dt(course.get("heureDepart")),
        "statut": course.get("statut"),
        "ordre_arrivee": ordre if isinstance(ordre, list) else None,
        "rapports_definitifs_disponibles": bool(course.get("rapportsDefinitifsDisponibles")),
    }


# ---------------------------------------------------------------------
# Participants
# ---------------------------------------------------------------------

def _place_from_ordre_arrivee(ordre: Any, num_pmu: int | None) -> int | None:
    """
    `ordreArrivee` est une liste de listes : [[3],[7],[1,9]] — le 3e rang
    est un ex æquo entre le 1 et le 9. On rend le rang (1-indexé).
    """
    if not isinstance(ordre, list) or num_pmu is None:
        return None
    for rang, groupe in enumerate(ordre, start=1):
        if isinstance(groupe, list) and num_pmu in [as_int(x) for x in groupe]:
            return rang
        if as_int(groupe) == num_pmu:
            return rang
    return None


def parse_participant(p: dict, ordre_arrivee: Any = None) -> dict:
    num_pmu = as_int(p.get("numPmu"))
    place = as_int(p.get("ordreArrivee")) or _place_from_ordre_arrivee(ordre_arrivee, num_pmu)

    return {
        "num_pmu": num_pmu,
        "id_cheval": as_int(p.get("idCheval")),
        "nom_cheval": p.get("nom"),
        "age": as_int(p.get("age")),
        "sexe": p.get("sexe"),
        "race": p.get("race"),
        "pays": p.get("pays"),
        # Généalogie : des noms, pas des identifiants.
        "nom_pere": p.get("nomPere"),
        "nom_mere": p.get("nomMere"),
        "nom_pere_mere": p.get("nomPereMere"),
        "eleveur": p.get("eleveur"),
        # Personnel
        "driver": p.get("driver") or p.get("jockey"),
        "entraineur": p.get("entraineur"),
        "proprietaire": p.get("proprietaire"),
        "driver_change": p.get("driverChange"),
        # Conditions
        "place_corde": as_int(p.get("placeCorde")),
        "handicap_poids": as_float(p.get("handicapPoids")),
        "handicap_valeur": as_float(p.get("handicapValeur")),
        "handicap_distance": as_int(p.get("handicapDistance")),
        "poids_condition_monte": as_float(p.get("poidsConditionMonteChange"))
        or as_float(p.get("poidsConditionMonte")),
        "oeilleres": p.get("oeilleres"),
        "deferre": p.get("deferre"),
        "supplement": cents_to_eur(p.get("supplement")),
        "engagement": p.get("engagement"),
        "jument_pleine": p.get("jumentPleine"),
        "indicateur_inedit": p.get("indicateurInedit"),
        "allure": p.get("allure"),
        "robe": dig(p, "robe", "libelleCourt"),
        # Palmarès déclaré (connu avant le départ → utilisable en feature)
        "musique": p.get("musique"),
        "nombre_courses": as_int(p.get("nombreCourses")),
        "nombre_victoires": as_int(p.get("nombreVictoires")),
        "nombre_places": as_int(p.get("nombrePlaces")),
        "nombre_places_second": as_int(p.get("nombrePlacesSecond")),
        "nombre_places_troisieme": as_int(p.get("nombrePlacesTroisieme")),
        "gains_carriere": cents_to_eur(dig(p, "gainsParticipant", "gainsCarriere")),
        "gains_victoires": cents_to_eur(dig(p, "gainsParticipant", "gainsVictoires")),
        "gains_place": cents_to_eur(dig(p, "gainsParticipant", "gainsPlace")),
        "gains_annee_en_cours": cents_to_eur(dig(p, "gainsParticipant", "gainsAnneeEnCours")),
        "gains_annee_precedente": cents_to_eur(dig(p, "gainsParticipant", "gainsAnneePrecedente")),
        # Résultat
        "statut": p.get("statut"),
        "ordre_arrivee": place,
        "statut_arrivee": p.get("statutArrivee"),
        "temps_officiel_ms": as_int(p.get("tempsObtenu")),
        "reduction_km_ms": as_int(p.get("reductionKilometrique")),
        "distance_cheval_precedent": p.get("distanceChevalPrecedent"),
        "commentaire_apres_course": p.get("commentaireApresCourse"),
    }


def parse_cotes(p: dict, releve_le: datetime) -> list[dict]:
    """
    Extrait les relevés de cote présents dans un objet participant.
    Une ligne par type de pari trouvé.
    """
    out = []
    for key in ("dernierRapportDirect", "dernierRapportReference"):
        rap = p.get(key)
        if not isinstance(rap, dict):
            continue
        rapport = as_float(rap.get("rapport"))
        if rapport is None:
            continue
        out.append(
            {
                "num_pmu": as_int(p.get("numPmu")),
                # dateRapport quand elle existe : c'est l'horodatage réel du
                # relevé, bien plus fiable que l'heure de notre appel.
                "releve_le": ms_to_dt(rap.get("dateRapport")) or releve_le,
                "type_pari": rap.get("typePari") or key,
                "rapport": rapport,
                "favoris": bool(rap.get("favoris")),
                "grosse_prise": bool(rap.get("grossePrise")),
                "tendance": as_int(rap.get("nombreIndicateurTendance")),
            }
        )
    return out


# ---------------------------------------------------------------------
# Performances détaillées
# ---------------------------------------------------------------------

def parse_performances(bloc: dict) -> list[dict]:
    """
    Un bloc = un participant du jour + ses `coursesCourues` passées.
    Renvoie une ligne par course passée.
    """
    id_cheval = as_int(bloc.get("idCheval"))
    if id_cheval is None:
        return []

    lignes = []
    for c in bloc.get("coursesCourues") or []:
        if not isinstance(c, dict):
            continue
        place_obj = c.get("place") if isinstance(c.get("place"), dict) else {}
        # participants[] porte parfois le détail du cheval concerné
        detail = {}
        for part in c.get("participants") or []:
            if isinstance(part, dict) and as_int(part.get("idCheval")) == id_cheval:
                detail = part
                break

        lignes.append(
            {
                "id_cheval": id_cheval,
                "date_course": ms_to_date(c.get("date")),
                "hippodrome_lib": c.get("hippodrome"),
                "hippodrome_code": c.get("codeHippodrome"),
                "nom_prix": c.get("nomPrix"),
                "discipline": c.get("discipline"),
                "specialite": c.get("specialite"),
                "distance": as_int(c.get("distance")),
                "allocation": cents_to_eur(c.get("allocation")),
                "nb_participants": as_int(c.get("nbParticipants")),
                "place": as_int(place_obj.get("place")),
                "statut_arrivee": place_obj.get("statusArrivee"),
                "corde": as_int(detail.get("corde") or c.get("corde")),
                "poids_jockey": as_float(detail.get("poidsJockey") or c.get("poidsJockey")),
                "nom_jockey": detail.get("nomJockey") or c.get("nomJockey"),
                "oeillere": detail.get("oeillere") or c.get("oeillere"),
                "deferre": detail.get("deferre") or c.get("deferre"),
                "etat_terrain": c.get("etatTerrain"),
                "temps_premier_ms": as_int(c.get("tempsDuPremier")),
                "reduction_km_ms": as_int(
                    detail.get("reductionKilometrique") or c.get("reductionKilometrique")
                ),
                "distance_avec_precedent": detail.get("distanceAvecPrecedent")
                or c.get("distanceAvecPrecedent"),
            }
        )
    return [l for l in lignes if l["date_course"] is not None]
