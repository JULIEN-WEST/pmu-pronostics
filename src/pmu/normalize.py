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

import json
import re
import unicodedata
from datetime import date, datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

# Les courses françaises sont datées en heure de Paris. Passer par une
# vraie base de fuseaux plutôt qu'un décalage figé : +1 h en hiver,
# +2 h en été, et le basculement se fait tout seul.
FUSEAU_COURSES = ZoneInfo("Europe/Paris")

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
    """
    Date CALENDAIRE d'un horodatage PMU.

    ⚠️ À lire en heure de Paris, jamais en UTC.

    Les champs de date du PMU (`dateReunion`, `date` d'une course passée)
    valent MINUIT heure locale. En été, minuit à Paris, c'est 22 h 00 UTC
    la VEILLE. Lus en UTC, ils renvoient donc systématiquement le jour
    précédent, et toute la base se décale d'un jour :

        1787608800000  lu en UTC   -> 2026-08-24  ✗
                       lu en Paris -> 2026-08-25  ✓

    Le symptôme est sournois : la collecte annonce « 40 courses » pour le
    jour J, elles sont rangées en J−1, et le calcul des pronostics du
    jour J ne trouve plus rien. Le PMU fournit d'ailleurs son
    `timezoneOffset` dans la réponse — c'était l'indice.

    Les horodatages de DÉPART (`heure_depart`) ne sont pas concernés :
    ce sont de vrais instants, stockés en timestamptz, et c'est eux —
    pas cette date — que le calcul des features utilise pour l'ordre
    chronologique. La fuite de données n'a donc jamais été en cause.
    """
    dt = ms_to_dt(value)
    return dt.astimezone(FUSEAU_COURSES).date() if dt else None


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


def as_texte(value: Any) -> str | None:
    """
    Ramène n'importe quelle valeur à du texte insérable en base.

    Indispensable : plusieurs champs que l'on croirait textuels sont en
    réalité des objets côté PMU —

        commentaireApresCourse   {"texte": "...", "source": "..."}
        distanceChevalPrecedent  {"libelleCourt": "1/2 L", "code": 3, ...}
        robe                     {"code": "001", "libelleLong": "ALEZAN"}

    Passer un dict à psycopg pour une colonne `text` lève
    « cannot adapt type dict », l'insertion casse, et comme l'exception
    remonte au milieu d'une transaction, TOUTE la journée de collecte est
    perdue — pas seulement la ligne fautive.

    On extrait donc le libellé utile, et à défaut on sérialise plutôt que
    de laisser passer un objet.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, dict):
        for cle in ("texte", "libelleLong", "libelleCourt", "libelle", "valeur", "code"):
            v = value.get(cle)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return json.dumps(value, ensure_ascii=False)[:500] or None
    if isinstance(value, (list, tuple)):
        morceaux = [as_texte(v) for v in value]
        return " | ".join([m for m in morceaux if m]) or None
    return str(value)


def as_id(value: Any) -> str | None:
    """
    Identifiant de cheval.

    ⚠️ `idCheval` N'EST PAS UN NOMBRE. Le PMU renvoie une chaîne composée
    du nom, de la mère et du père : « KHAMEPHIS GAME-AKITA-ZARAK ».
    C'est stable et unique — mais le convertir en entier donne None, et
    tout le fil généalogique s'effondre en silence.
    """
    t = as_texte(value)
    return t if t else None


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
        "id_cheval": as_id(p.get("idCheval")),
        "nom_cheval": as_texte(p.get("nom")),
        "age": as_int(p.get("age")),
        "sexe": as_texte(p.get("sexe")),
        "race": as_texte(p.get("race")),
        "pays": as_texte(p.get("pays")),
        # Généalogie : des noms, pas des identifiants.
        "nom_pere": as_texte(p.get("nomPere")),
        "nom_mere": as_texte(p.get("nomMere")),
        "nom_pere_mere": as_texte(p.get("nomPereMere")),
        "eleveur": as_texte(p.get("eleveur")),
        # Personnel
        "driver": as_texte(p.get("driver") or p.get("jockey")),
        "entraineur": as_texte(p.get("entraineur")),
        "proprietaire": as_texte(p.get("proprietaire")),
        "driver_change": p.get("driverChange"),
        # Conditions
        "place_corde": as_int(p.get("placeCorde")),
        "handicap_poids": as_float(p.get("handicapPoids")),
        "handicap_valeur": as_float(p.get("handicapValeur")),
        "handicap_distance": as_int(p.get("handicapDistance")),
        "poids_condition_monte": as_float(p.get("poidsConditionMonteChange"))
        or as_float(p.get("poidsConditionMonte")),
        "oeilleres": as_texte(p.get("oeilleres")),
        "deferre": as_texte(p.get("deferre")),
        "supplement": cents_to_eur(p.get("supplement")),
        "engagement": p.get("engagement"),
        "jument_pleine": p.get("jumentPleine"),
        "indicateur_inedit": p.get("indicateurInedit"),
        "allure": as_texte(p.get("allure")),
        "robe": as_texte(p.get("robe")),
        # Palmarès déclaré (connu avant le départ → utilisable en feature)
        "musique": as_texte(p.get("musique")),
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
        "statut": as_texte(p.get("statut")),
        "ordre_arrivee": place,
        "statut_arrivee": as_texte(p.get("statutArrivee")),
        "temps_officiel_ms": as_int(p.get("tempsObtenu")),
        "reduction_km_ms": as_int(p.get("reductionKilometrique")),
        # Ces deux-là sont des OBJETS côté PMU, pas des chaînes.
        "distance_cheval_precedent": as_texte(p.get("distanceChevalPrecedent")),
        "commentaire_apres_course": as_texte(p.get("commentaireApresCourse")),
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

def parse_performances(bloc: dict, id_cheval: str | None = None) -> list[dict]:
    """
    Un bloc = un partant du jour + ses `coursesCourues` passées.
    Renvoie une ligne par course passée.

    ⚠️ Les blocs de cet endpoint sont identifiés par `numPmu` et
    `nomCheval` — PAS par `idCheval`, qui n'y figure pas. L'appelant doit
    donc fournir l'identifiant, résolu depuis l'appel `participants` de la
    même course (cf. collect.collecte_course). Sans ça on obtient zéro
    ligne, sans la moindre erreur.
    """
    id_cheval = id_cheval or as_id(bloc.get("idCheval"))
    if not id_cheval:
        return []

    # Dans `coursesCourues[].participants[]`, le cheval concerné se
    # reconnaît au drapeau `itsHim` — le seul repère fiable, puisque les
    # identifiants n'y sont pas.
    lignes = []
    for c in bloc.get("coursesCourues") or []:
        if not isinstance(c, dict):
            continue
        place_obj = c.get("place") if isinstance(c.get("place"), dict) else {}

        detail: dict = {}
        for part in c.get("participants") or []:
            if not isinstance(part, dict):
                continue
            if part.get("itsHim") is True:
                detail = part
                break
            if as_texte(part.get("nomCheval")) == as_texte(bloc.get("nomCheval")):
                detail = part

        lignes.append(
            {
                "id_cheval": id_cheval,
                "date_course": ms_to_date(c.get("date")),
                "hippodrome_lib": as_texte(c.get("hippodrome")),
                "hippodrome_code": as_texte(c.get("codeHippodrome")),
                "nom_prix": as_texte(c.get("nomPrix")),
                "discipline": as_texte(c.get("discipline")),
                "specialite": as_texte(c.get("specialite")),
                "distance": as_int(c.get("distance")),
                "allocation": cents_to_eur(c.get("allocation")),
                "nb_participants": as_int(c.get("nbParticipants")),
                "place": as_int(place_obj.get("place")),
                "statut_arrivee": as_texte(place_obj.get("statusArrivee")),
                "corde": as_int(detail.get("corde") or c.get("corde")),
                "poids_jockey": as_float(detail.get("poidsJockey") or c.get("poidsJockey")),
                "nom_jockey": as_texte(detail.get("nomJockey") or c.get("nomJockey")),
                "oeillere": as_texte(detail.get("oeillere") or c.get("oeillere")),
                "deferre": as_texte(detail.get("deferre") or c.get("deferre")),
                "etat_terrain": as_texte(c.get("etatTerrain")),
                "temps_premier_ms": as_int(c.get("tempsDuPremier")),
                "reduction_km_ms": as_int(
                    detail.get("reductionKilometrique") or c.get("reductionKilometrique")
                ),
                # Objet côté PMU, comme sur les participants.
                "distance_avec_precedent": as_texte(
                    detail.get("distanceAvecPrecedent") or c.get("distanceAvecPrecedent")
                ),
            }
        )
    return [l for l in lignes if l["date_course"] is not None]
