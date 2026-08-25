"""
Météo réelle par hippodrome — l'état du terrain, mesuré au lieu d'être lu.

===================================================================
POURQUOI

`etat_terrain` est un adjectif posé par un commissaire : « bon »,
« souple », « lourd ». Il est grossier, subjectif, et il ne dit rien
de ce qui s'est passé la veille. Or c'est la pluie des dernières
vingt-quatre heures qui fait le terrain, pas celle du moment. Deux
« bon » séparés par un orage nocturne n'ont rien à voir.

Le pénétromètre, désormais exploité, couvre une partie du besoin —
mais il n'est renseigné qu'en galop, et pas partout.

===================================================================
LA SOURCE

Open-Meteo : gratuit, sans clé, sans inscription. Deux points d'entrée
au format identique — archive pour le passé, prévision pour le jour
même et les suivants. La géolocalisation des hippodromes passe par
leur service de géocodage, à partir du libellé PMU.

⚠️ CE QUI N'A PAS PU ÊTRE VÉRIFIÉ D'ICI
L'API d'archive refuse les robots, donc la forme exacte de sa réponse
n'a pas été confrontée en direct — seulement sa documentation. C'est
précisément le genre d'hypothèse qui a déjà coûté trois journées de
collecte à ce projet (identifiant en chaîne, champs en objet,
identifiant absent d'un bloc). D'où deux précautions :

  1. l'extraction est DÉFENSIVE : une forme inattendue rend « pas de
     météo », jamais une exception ;
  2. `python -m pmu.meteo verifier` interroge l'API pour de vrai et
     affiche ce qu'elle a renvoyé. À lancer une fois depuis le
     conteneur, là où le réseau est ouvert.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import requests

log = logging.getLogger("pmu.meteo")

GEOCODAGE = "https://geocoding-api.open-meteo.com/v1/search"
ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
PREVISION = "https://api.open-meteo.com/v1/forecast"

VARIABLES = ["temperature_2m", "precipitation", "wind_speed_10m",
             "relative_humidity_2m"]

# Fenêtre de course : on résume la journée entre 8 h et 20 h locales.
HEURE_DEBUT, HEURE_FIN = 8, 20

SQL_TABLES = """
CREATE TABLE IF NOT EXISTS meteo_lieu (
    hippodrome_code text PRIMARY KEY,
    libelle         text,
    latitude        double precision,
    longitude       double precision,
    -- Un géocodage qui échoue est mémorisé lui aussi : sans ça, chaque
    -- collecte retenterait indéfiniment les mêmes libellés introuvables.
    resolu          boolean NOT NULL DEFAULT false,
    tente_le        timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS meteo (
    hippodrome_code text NOT NULL,
    date_course     date NOT NULL,
    temperature     numeric(5,1),
    pluie_jour      numeric(6,2),   -- mm cumulés sur la journée de course
    pluie_24h       numeric(6,2),   -- mm cumulés sur les 24 h précédentes
    vent_max        numeric(5,1),
    humidite        numeric(5,1),
    source          text,
    collecte_le     timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (hippodrome_code, date_course)
);
"""


# ---------------------------------------------------------------------
# Géocodage
# ---------------------------------------------------------------------

_PREFIXES = ("HIPPODROME DE LA ", "HIPPODROME DE L'", "HIPPODROME DES ",
             "HIPPODROME DE ", "HIPPODROME D'", "HIPPODROME DU ",
             "HIPPODROME ")


def candidats(libelle: str) -> list[str]:
    """
    Noms de commune à essayer, du plus précis au plus large.

    Les libellés PMU sont hétérogènes : « HIPPODROME DE LA CAPELLE »,
    « PARIS-VINCENNES », « CAGNES-SUR-MER MIDI ». On dégrade donc par
    étapes plutôt que d'espérer qu'une seule règle marche partout.
    """
    if not libelle:
        return []
    t = " ".join(str(libelle).upper().split())
    for p in _PREFIXES:
        if t.startswith(p):
            t = t[len(p):]
            break
    sortie = [t]
    if "-" in t:
        # « PARIS-VINCENNES » → « VINCENNES » : le second terme est le
        # lieu réel, le premier n'est qu'un rattachement administratif.
        sortie.append(t.rsplit("-", 1)[-1].strip())
        sortie.append(t.split("-", 1)[0].strip())
    mots = t.split()
    if len(mots) > 1:
        sortie.append(mots[0])
    vus, uniques = set(), []
    for c in sortie:
        c = c.strip()
        if c and c not in vus:
            vus.add(c)
            uniques.append(c)
    return uniques


@dataclass
class ClientMeteo:
    """Client Open-Meteo. Aucune clé, aucun compte."""
    session: requests.Session = field(default_factory=requests.Session)
    timeout: float = 20.0

    def _get(self, url: str, params: dict) -> Any:
        r = self.session.get(url, params=params, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def geocoder(self, nom: str) -> tuple[float, float] | None:
        for candidat in candidats(nom):
            try:
                data = self._get(GEOCODAGE, {"name": candidat, "country": "FR",
                                             "count": 1, "language": "fr",
                                             "format": "json"})
            except (requests.RequestException, ValueError) as exc:
                log.warning("géocodage de %r impossible : %s", candidat, exc)
                return None
            resultats = data.get("results") if isinstance(data, dict) else None
            if resultats:
                r = resultats[0]
                lat, lon = r.get("latitude"), r.get("longitude")
                if lat is not None and lon is not None:
                    log.info("hippodrome %r → %s (%.4f, %.4f)",
                             nom, r.get("name"), lat, lon)
                    return float(lat), float(lon)
        return None

    def journee(self, lat: float, lon: float, jour: date,
                *, aujourdhui: date | None = None) -> dict | None:
        """
        Résumé météo d'une journée de course, avec la pluie de la veille.

        Le choix du point d'entrée dépend de la date : l'archive ne
        couvre pas le jour même (elle accuse plusieurs jours de retard),
        la prévision ne remonte que de quelques jours.
        """
        aujourdhui = aujourdhui or date.today()
        veille = jour - timedelta(days=1)
        commun = {"latitude": lat, "longitude": lon,
                  "hourly": ",".join(VARIABLES), "timezone": "Europe/Paris"}
        recent = (aujourdhui - jour).days < 5
        try:
            if recent:
                data = self._get(PREVISION, {**commun, "past_days": 7,
                                             "forecast_days": 3})
                source = "prevision"
            else:
                data = self._get(ARCHIVE, {**commun,
                                           "start_date": veille.isoformat(),
                                           "end_date": jour.isoformat()})
                source = "archive"
        except (requests.RequestException, ValueError) as exc:
            log.warning("météo indisponible pour %s : %s", jour, exc)
            return None
        resume = resumer(data, jour)
        if resume:
            resume["source"] = source
        return resume


# ---------------------------------------------------------------------
# Extraction — volontairement méfiante
# ---------------------------------------------------------------------

def resumer(data: Any, jour: date) -> dict | None:
    """
    Réduit la réponse horaire à un résumé par journée de course.

    Aucune confiance n'est faite à la forme reçue : tout est vérifié,
    et le moindre écart rend None plutôt que de lever. Une météo
    manquante coûte quelques features vides ; une exception ici ferait
    tomber toute la collecte du jour.
    """
    if not isinstance(data, dict):
        return None
    horaire = data.get("hourly")
    if not isinstance(horaire, dict):
        return None
    temps = horaire.get("time")
    if not isinstance(temps, list) or not temps:
        return None

    def colonne(nom: str) -> list:
        v = horaire.get(nom)
        return v if isinstance(v, list) and len(v) == len(temps) else [None] * len(temps)

    temp = colonne("temperature_2m")
    pluie = colonne("precipitation")
    vent = colonne("wind_speed_10m")
    humid = colonne("relative_humidity_2m")

    jour_txt = jour.isoformat()
    veille_txt = (jour - timedelta(days=1)).isoformat()

    def nombres(valeurs) -> list[float]:
        return [float(v) for v in valeurs
                if isinstance(v, (int, float)) and not isinstance(v, bool)]

    t_jour, p_jour, v_jour, h_jour, p_24h = [], [], [], [], []
    for i, horodatage in enumerate(temps):
        if not isinstance(horodatage, str) or "T" not in horodatage:
            continue
        j, heure = horodatage.split("T", 1)
        try:
            h = int(heure[:2])
        except ValueError:
            continue
        # Les 24 h qui précèdent la journée de course : c'est cette
        # pluie-là qui a fait le terrain, pas celle de l'après-midi.
        if j == veille_txt:
            p_24h.append(pluie[i])
        if j == jour_txt:
            if h < HEURE_DEBUT:
                p_24h.append(pluie[i])
            if HEURE_DEBUT <= h <= HEURE_FIN:
                t_jour.append(temp[i])
                p_jour.append(pluie[i])
                v_jour.append(vent[i])
                h_jour.append(humid[i])

    t, p, v, hh, p24 = (nombres(t_jour), nombres(p_jour), nombres(v_jour),
                        nombres(h_jour), nombres(p_24h))
    if not (t or p or v or hh or p24):
        return None
    return {
        "temperature": round(sum(t) / len(t), 1) if t else None,
        "pluie_jour": round(sum(p), 2) if p else None,
        "pluie_24h": round(sum(p24), 2) if p24 else None,
        "vent_max": round(max(v), 1) if v else None,
        "humidite": round(sum(hh) / len(hh), 1) if hh else None,
    }


# ---------------------------------------------------------------------
# Collecte
# ---------------------------------------------------------------------

def _lieu(conn, client: ClientMeteo, code: str, libelle: str | None):
    """Coordonnées de l'hippodrome, géocodées une fois pour toutes."""
    row = conn.execute(
        "SELECT latitude, longitude, resolu FROM meteo_lieu WHERE hippodrome_code = %s",
        (code,)).fetchone()
    if row is not None:
        if not row["resolu"]:
            return None
        return row["latitude"], row["longitude"]

    coord = client.geocoder(libelle or code)
    conn.execute(
        """INSERT INTO meteo_lieu (hippodrome_code, libelle, latitude, longitude, resolu)
           VALUES (%s, %s, %s, %s, %s)
           ON CONFLICT (hippodrome_code) DO UPDATE SET
             libelle = EXCLUDED.libelle, latitude = EXCLUDED.latitude,
             longitude = EXCLUDED.longitude, resolu = EXCLUDED.resolu,
             tente_le = now()""",
        (code, libelle, coord[0] if coord else None,
         coord[1] if coord else None, coord is not None))
    conn.commit()
    if coord is None:
        log.warning("hippodrome %s (%r) non géocodé — sans météo", code, libelle)
    return coord


def enrichir(conn, jour: date, *, client: ClientMeteo | None = None) -> int:
    """
    Renseigne la météo des hippodromes qui courent ce jour-là.
    Renvoie le nombre de lignes écrites. Ne lève jamais : une météo
    absente ne doit pas empêcher une collecte de se terminer.
    """
    client = client or ClientMeteo()
    conn.execute(SQL_TABLES)
    conn.commit()

    lieux = conn.execute(
        """SELECT DISTINCT r.hippodrome_code AS code, h.libelle_long AS libelle
             FROM reunion r
             LEFT JOIN hippodrome h ON h.code = r.hippodrome_code
            WHERE r.date_reunion = %s AND r.hippodrome_code IS NOT NULL""",
        (jour,)).fetchall()

    ecrits = 0
    for l in lieux:
        deja = conn.execute(
            "SELECT 1 FROM meteo WHERE hippodrome_code = %s AND date_course = %s",
            (l["code"], jour)).fetchone()
        if deja:
            continue
        try:
            coord = _lieu(conn, client, l["code"], l["libelle"])
            if not coord:
                continue
            resume = client.journee(coord[0], coord[1], jour)
        except Exception as exc:  # noqa: BLE001 — la météo est un bonus
            log.warning("météo %s le %s : %s", l["code"], jour, exc)
            conn.rollback()
            continue
        if not resume:
            continue
        conn.execute(
            """INSERT INTO meteo (hippodrome_code, date_course, temperature,
                                  pluie_jour, pluie_24h, vent_max, humidite, source)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
               ON CONFLICT (hippodrome_code, date_course) DO UPDATE SET
                 temperature = EXCLUDED.temperature, pluie_jour = EXCLUDED.pluie_jour,
                 pluie_24h = EXCLUDED.pluie_24h, vent_max = EXCLUDED.vent_max,
                 humidite = EXCLUDED.humidite, source = EXCLUDED.source,
                 collecte_le = now()""",
            (l["code"], jour, resume.get("temperature"), resume.get("pluie_jour"),
             resume.get("pluie_24h"), resume.get("vent_max"),
             resume.get("humidite"), resume.get("source")))
        conn.commit()
        ecrits += 1
    if ecrits:
        log.info("météo : %d hippodrome(s) renseigné(s) le %s", ecrits, jour)
    return ecrits


# ---------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Météo des hippodromes")
    ap.add_argument("mode", choices=["verifier", "jour"])
    ap.add_argument("--date", type=lambda s: date.fromisoformat(s))
    ap.add_argument("--lieu", default="HIPPODROME DE LA CAPELLE")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s")

    if args.mode == "verifier":
        # ⚠️ À lancer depuis le conteneur. La forme de la réponse
        # d'archive n'a pas pu être confrontée en direct au moment
        # d'écrire ce module : cette commande le fait pour de vrai.
        c = ClientMeteo()
        print(f"Géocodage de {args.lieu!r}")
        print("  candidats essayés :", candidats(args.lieu))
        coord = c.geocoder(args.lieu)
        print("  →", coord or "ÉCHEC")
        if not coord:
            raise SystemExit(1)
        jour = args.date or (date.today() - timedelta(days=10))
        print(f"\nMétéo du {jour} (archive)")
        resume = c.journee(coord[0], coord[1], jour)
        print("  →", resume or "ÉCHEC — forme de réponse inattendue")
        aujourd = date.today()
        print(f"\nMétéo du {aujourd} (prévision)")
        print("  →", c.journee(coord[0], coord[1], aujourd) or "ÉCHEC")
        raise SystemExit(0 if resume else 2)

    from . import db
    with db.connect() as conn:
        n = enrichir(conn, args.date or date.today())
        print(f"{n} hippodrome(s) renseigné(s)")


if __name__ == "__main__":
    main()
