"""
API de service — FastAPI.

    uvicorn pmu.api:app --host 0.0.0.0 --port 8100

Elle ne calcule rien : elle relit la table `pronostic`, alimentée par le
job `pmu.predict`. Une requête HTTP ne doit jamais déclencher trente
secondes de calcul de features.

Endpoints :
    GET /sante                     état de la pile, volumétrie
    GET /pronostics                journée complète
    GET /pronostics/{code}         une course, ex. R1C3
    GET /ha/resume                 charge utile compacte pour Home Assistant
    GET /ha/prochaine              la course à venir, formatée pour l'affichage

L'API est en lecture seule et sans authentification : elle est prévue pour
un réseau local. Si elle doit sortir du LAN, mettre un reverse proxy avec
authentification devant — ne pas bricoler un jeton ici.
"""

from __future__ import annotations

import logging
import os
from datetime import date, datetime, timedelta, timezone

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import JSONResponse

from . import dataset, db
from .predict import lire_pronostics

log = logging.getLogger("pmu.api")

app = FastAPI(
    title="Pronostics PMU",
    version="0.1.0",
    description="Probabilités calibrées par partant. Lecture seule, réseau local.",
)

MODELE_DEFAUT = os.environ.get("PMU_MODELE", "sans_marche")


def _jour(param: str | None) -> date:
    if not param:
        return date.today()
    try:
        return date.fromisoformat(param)
    except ValueError:
        raise HTTPException(400, f"date illisible : {param!r} (attendu AAAA-MM-JJ)")


# ---------------------------------------------------------------------

@app.get("/sante")
def sante():
    """Volumétrie et fraîcheur. C'est ce que surveille Home Assistant."""
    try:
        with db.connect() as conn:
            s = dataset.stats(conn)
            row = conn.execute(
                "SELECT max(calcule_le) AS dernier FROM pronostic"
            ).fetchone() if _table_existe(conn, "pronostic") else None
    except Exception as exc:  # noqa: BLE001 — la santé ne doit jamais lever
        log.exception("santé indisponible")
        return JSONResponse({"ok": False, "erreur": str(exc)}, status_code=503)

    dernier = row["dernier"] if row else None
    age_h = None
    if dernier:
        age_h = round((datetime.now(timezone.utc) - dernier).total_seconds() / 3600, 1)

    return {
        "ok": True,
        # Un pronostic de plus de 24 h n'est plus un pronostic.
        "frais": age_h is not None and age_h < 24,
        "dernier_calcul": dernier.isoformat() if dernier else None,
        "age_heures": age_h,
        "modele": MODELE_DEFAUT,
        **{k: (v.isoformat() if isinstance(v, date) else v) for k, v in s.items()},
    }


def _table_existe(conn, nom: str) -> bool:
    row = conn.execute(
        "SELECT to_regclass(%s) IS NOT NULL AS existe", (f"pmu.{nom}",)
    ).fetchone()
    return bool(row and row["existe"])


@app.get("/pronostics")
def pronostics(
    date_: str | None = Query(None, alias="date", description="AAAA-MM-JJ, défaut aujourd'hui"),
    modele: str = Query(MODELE_DEFAUT),
    top: int = Query(0, ge=0, le=30, description="ne garder que les N premiers par course"),
):
    jour = _jour(date_)
    with db.connect() as conn:
        if not _table_existe(conn, "pronostic"):
            raise HTTPException(503, "table pronostic absente — lancer `pmu.predict jour`")
        courses = lire_pronostics(conn, jour, modele)
    if top:
        for c in courses:
            c["selection"] = c["selection"][:top]
    return {"date": jour.isoformat(), "modele": modele,
            "courses": len(courses), "programme": courses}


@app.get("/pronostics/{code}")
def pronostic_course(
    code: str,
    date_: str | None = Query(None, alias="date"),
    modele: str = Query(MODELE_DEFAUT),
):
    """`code` au format R1C3."""
    jour = _jour(date_)
    with db.connect() as conn:
        courses = lire_pronostics(conn, jour, modele)
    for c in courses:
        if c["code"].upper() == code.upper():
            return c
    raise HTTPException(404, f"{code} introuvable le {jour}")


# ---------------------------------------------------------------------
# Home Assistant
# ---------------------------------------------------------------------

def _prochaine(courses: list[dict]) -> dict | None:
    """Première course non arrivée dont le départ est encore devant nous."""
    maintenant = datetime.now(timezone.utc)
    futures = [
        c for c in courses
        if c["depart"] and not c["arrivee_connue"]
        and datetime.fromisoformat(c["depart"]) > maintenant - timedelta(minutes=5)
    ]
    return min(futures, key=lambda c: c["depart"]) if futures else None


@app.get("/ha/resume")
def ha_resume(modele: str = Query(MODELE_DEFAUT)):
    """
    Charge utile unique pour Home Assistant.

    Volontairement compacte : les attributs d'entité HA sont limités à
    16 ko après sérialisation, et un capteur trop lourd ralentit tout le
    moteur d'état. On plafonne donc à 6 partants par course et 12 courses.
    """
    jour = date.today()
    with db.connect() as conn:
        if not _table_existe(conn, "pronostic"):
            return {"ok": False, "courses": 0, "programme": [], "prochaine": None}
        courses = lire_pronostics(conn, jour, modele)

    compact = []
    for c in courses[:12]:
        compact.append({
            "code": c["code"],
            "hippodrome": c["hippodrome"],
            "libelle": (c["libelle"] or "")[:48],
            "discipline": c["discipline"],
            "distance": c["distance"],
            "depart": c["depart"],
            "partants": c["partants"],
            "confiance": round(c["confiance"], 3),
            "arrivee_connue": c["arrivee_connue"],
            "selection": [
                {k: s[k] for k in ("num", "cheval", "proba", "cote", "valeur", "arrivee")}
                for s in c["selection"][:6]
            ],
        })

    proch = _prochaine(courses)
    return {
        "ok": True,
        "date": jour.isoformat(),
        "modele": modele,
        "courses": len(courses),
        "prochaine": {
            "code": proch["code"],
            "hippodrome": proch["hippodrome"],
            "depart": proch["depart"],
            "confiance": round(proch["confiance"], 3),
            "selection": proch["selection"][:6],
        } if proch else None,
        "programme": compact,
    }


@app.get("/ha/prochaine")
def ha_prochaine(modele: str = Query(MODELE_DEFAUT)):
    """La prochaine course seule — pour un capteur léger rafraîchi souvent."""
    with db.connect() as conn:
        if not _table_existe(conn, "pronostic"):
            raise HTTPException(503, "pas encore de pronostic")
        courses = lire_pronostics(conn, date.today(), modele)
    proch = _prochaine(courses)
    if not proch:
        return {"ok": True, "prochaine": None, "message": "plus de course aujourd'hui"}
    return {"ok": True, "prochaine": proch}
