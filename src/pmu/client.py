"""
Client HTTP pour l'API turfinfo du PMU.

⚠️ API NON OFFICIELLE. Elle n'est couverte par aucun contrat : le PMU peut
la modifier ou la fermer sans préavis, et rien n'oblige à en autoriser
l'usage. Deux conséquences pratiques :

  1. On throttle sérieusement (défaut : 2 req/s) et on met en cache sur
     disque. Un backfill qui martèle le serveur se fait couper, et c'est
     mérité.
  2. Le parsing est défensif de bout en bout (`.get()` partout, jamais
     d'accès direct par clé). Le jour où une clé disparaît, la collecte
     dégrade au lieu de planter.

Le segment `/client/62/` est un numéro de version d'app mobile. Il évolue.
S'il renvoie 404 ou 426, essayer les valeurs voisines (61, 63, 64…) :
`PmuClient.detect_client_version()` fait ça automatiquement.
"""

from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import requests

log = logging.getLogger(__name__)

BASE_URL = "https://online.turfinfo.api.pmu.fr/rest/client"
DEFAULT_CLIENT_VERSION = 62

# Le PMU ne publie pas de limite. 2 req/s avec un jitter est un compromis
# raisonnable : ~7000 requêtes/heure, largement de quoi tenir un backfill
# nocturne sans se faire remarquer.
DEFAULT_RPS = 2.0


class PmuError(RuntimeError):
    pass


class PmuNotFound(PmuError):
    """404 — la ressource n'existe pas (jour sans course, course annulée)."""


@dataclass
class PmuClient:
    client_version: int = DEFAULT_CLIENT_VERSION
    rps: float = DEFAULT_RPS
    timeout: float = 20.0
    max_retries: int = 4
    cache_dir: Path | None = None
    session: requests.Session = field(default_factory=requests.Session)
    _last_call: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        self.session.headers.update(
            {
                # Se présenter honnêtement plutôt que d'usurper un navigateur.
                "User-Agent": "pmu-pronostics/0.1 (projet de recherche personnel)",
                "Accept": "application/json",
            }
        )
        if self.cache_dir:
            self.cache_dir = Path(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    # -- plomberie ----------------------------------------------------

    def _throttle(self) -> None:
        if self.rps <= 0:
            return
        gap = 1.0 / self.rps
        wait = gap - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait + random.uniform(0, gap * 0.25))
        self._last_call = time.monotonic()

    def _cache_path(self, key: str) -> Path | None:
        if not self.cache_dir:
            return None
        safe = key.strip("/").replace("/", "__") or "root"
        return self.cache_dir / f"{safe}.json"

    def get(self, path: str, *, use_cache: bool = True) -> Any:
        """GET sur un chemin relatif, ex. 'programme/23082026/R1/C1/participants'."""
        cache = self._cache_path(path) if use_cache else None
        if cache and cache.exists():
            try:
                return json.loads(cache.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                cache.unlink(missing_ok=True)  # cache corrompu, on refait

        url = f"{BASE_URL}/{self.client_version}/{path.lstrip('/')}"
        last_exc: Exception | None = None

        for attempt in range(self.max_retries):
            self._throttle()
            try:
                resp = self.session.get(url, timeout=self.timeout)
            except requests.RequestException as exc:
                last_exc = exc
                log.warning("réseau KO (%s) sur %s — tentative %d", exc, path, attempt + 1)
                time.sleep(2**attempt)
                continue

            if resp.status_code == 404:
                raise PmuNotFound(path)
            if resp.status_code in (429, 500, 502, 503, 504):
                delay = 2**attempt + random.uniform(0, 1)
                log.warning("HTTP %s sur %s — pause %.1fs", resp.status_code, path, delay)
                time.sleep(delay)
                last_exc = PmuError(f"HTTP {resp.status_code}")
                continue
            if not resp.ok:
                raise PmuError(f"HTTP {resp.status_code} sur {path}")

            # Certains endpoints répondent 200 avec un corps vide.
            if not resp.content.strip():
                raise PmuNotFound(path)
            try:
                data = resp.json()
            except json.JSONDecodeError as exc:
                raise PmuError(f"réponse non-JSON sur {path}") from exc

            if cache:
                cache.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            return data

        raise PmuError(f"échec définitif sur {path}") from last_exc

    # -- endpoints ----------------------------------------------------

    @staticmethod
    def fmt_date(d: date) -> str:
        """L'API attend DDMMYYYY, sans séparateur."""
        return d.strftime("%d%m%Y")

    def programme(self, d: date, *, use_cache: bool = True) -> dict:
        """Toutes les réunions et courses d'une journée."""
        data = self.get(f"programme/{self.fmt_date(d)}", use_cache=use_cache)
        return (data or {}).get("programme", {}) or {}

    def participants(self, d: date, r: int, c: int, *, use_cache: bool = True) -> list[dict]:
        """Partants d'une course. Après l'arrivée, contient aussi ordreArrivee."""
        data = self.get(
            f"programme/{self.fmt_date(d)}/R{r}/C{c}/participants", use_cache=use_cache
        )
        return (data or {}).get("participants", []) or []

    def performances_detaillees(
        self, d: date, r: int, c: int, *, use_cache: bool = True
    ) -> list[dict]:
        """
        Courses PASSÉES de chaque partant. C'est la mine d'or du projet :
        elle donne de la profondeur historique sans backfill jour par jour.
        """
        data = self.get(
            f"programme/{self.fmt_date(d)}/R{r}/C{c}/performances-detaillees/pretty",
            use_cache=use_cache,
        )
        return (data or {}).get("participants", []) or []

    def rapports_definitifs(self, d: date, r: int, c: int, *, use_cache: bool = True) -> list[dict]:
        """Rapports payés, disponibles seulement une fois la course arrivée."""
        data = self.get(
            f"programme/{self.fmt_date(d)}/R{r}/C{c}/rapports-definitifs", use_cache=use_cache
        )
        return data if isinstance(data, list) else (data or {}).get("rapports", []) or []

    def citations(self, d: date, r: int, c: int, *, use_cache: bool = True) -> Any:
        """Cotes en direct (relevé instantané). À échantillonner avant le départ."""
        return self.get(
            f"programme/{self.fmt_date(d)}/R{r}/C{c}/citations", use_cache=use_cache
        )

    # -- robustesse ---------------------------------------------------

    def detect_client_version(self, probe_date: date, candidates: range | None = None) -> int:
        """
        Le segment /client/<n>/ suit les versions de l'app PMU et finit par
        être retiré. On balaie les valeurs voisines jusqu'à en trouver une
        qui répond, et on l'adopte pour la suite de la session.
        """
        candidates = candidates or range(DEFAULT_CLIENT_VERSION - 4, DEFAULT_CLIENT_VERSION + 12)
        for version in candidates:
            self.client_version = version
            try:
                prog = self.programme(probe_date, use_cache=False)
            except (PmuError, PmuNotFound):
                continue
            if prog.get("reunions"):
                log.info("version client retenue : %d", version)
                return version
        raise PmuError(
            "aucune version /client/<n>/ ne répond — l'API a probablement changé de forme"
        )
