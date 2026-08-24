"""
Sonde l'API PMU et capture de vrais payloads.

    python scripts/probe.py --date 2026-08-22

Pourquoi c'est la PREMIÈRE chose à lancer : le parsing de ce dépôt a été
écrit à partir de la forme observée de l'API, pas d'une spécification —
il n'en existe aucune. Les noms de clés peuvent différer selon la
discipline, le pays ou la version. Cette sonde imprime ce que l'API
renvoie VRAIMENT et signale les clés que le parsing ignore.

Les payloads sont écrits dans `fixtures/` pour servir de base à des tests
de contrat.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE / "src"))

from pmu import normalize as nz          # noqa: E402
from pmu.client import PmuClient, PmuError, PmuNotFound  # noqa: E402


def montrer(titre: str, obj) -> None:
    print(f"\n{'─' * 66}\n  {titre}\n{'─' * 66}")
    print(json.dumps(obj, indent=2, ensure_ascii=False)[:2200])


def cles_ignorees(brut: dict, parse: dict, correspondances: dict) -> list[str]:
    """Clés présentes dans le JSON que le parsing ne reprend nulle part."""
    connues = set(correspondances)
    return sorted(k for k in brut if k not in connues)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", type=lambda s: date.fromisoformat(s),
                    default=date.today() - timedelta(days=1))
    ap.add_argument("--sortie", default="fixtures")
    args = ap.parse_args()

    sortie = RACINE / args.sortie
    sortie.mkdir(exist_ok=True)
    client = PmuClient(cache_dir=None, rps=1.0)

    print(f"Sonde du {args.date} · version client {client.client_version}")

    try:
        prog = client.programme(args.date)
    except (PmuError, PmuNotFound) as exc:
        print(f"\n⚠ version {client.client_version} muette ({exc}) — détection…")
        client.detect_client_version(args.date)
        prog = client.programme(args.date)

    reunions = prog.get("reunions") or []
    print(f"\n✓ {len(reunions)} réunion(s)")
    if not reunions:
        print("Aucune course ce jour-là. Essayer une autre date.")
        return

    (sortie / "programme.json").write_text(
        json.dumps(prog, ensure_ascii=False, indent=2)[:400_000], encoding="utf-8"
    )

    r0 = reunions[0]
    print(f"  clés d'une réunion : {sorted(r0)}")
    montrer("Réunion normalisée", {**nz.parse_reunion(r0), "meteo": "…"})

    courses = r0.get("courses") or []
    if not courses:
        print("Réunion sans course.")
        return
    c0 = courses[0]
    print(f"\n  clés d'une course : {sorted(c0)}")
    montrer("Course normalisée", {
        k: str(v) for k, v in nz.parse_course(c0, nz.ms_to_date(r0.get("dateReunion"))).items()
    })

    num_r = nz.as_int(c0.get("numReunion")) or nz.as_int(r0.get("numOfficiel"))
    num_c = nz.as_int(c0.get("numOrdre"))

    # --- Participants ---
    try:
        participants = client.participants(args.date, num_r, num_c)
    except (PmuError, PmuNotFound) as exc:
        print(f"\n⚠ participants indisponibles : {exc}")
        participants = []

    if participants:
        (sortie / "participants.json").write_text(
            json.dumps(participants, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        p0 = participants[0]
        print(f"\n✓ {len(participants)} partants · clés disponibles :")
        for k in sorted(p0):
            apercu = str(p0[k])[:52]
            print(f"    {k:<32} {apercu}")
        montrer("Partant normalisé", {
            k: str(v) for k, v in nz.parse_participant(p0, c0.get("ordreArrivee")).items()
        })
        # Alerte sur les champs que le parsing laisse vides alors que la
        # source contient quelque chose : signe d'un renommage côté PMU.
        parse = nz.parse_participant(p0, c0.get("ordreArrivee"))
        vides = [k for k, v in parse.items() if v is None]
        if vides:
            print(f"\n⚠ champs normalisés à None : {vides}")
            print("  → si l'un d'eux figure dans la liste des clés ci-dessus "
                  "sous un autre nom, adapter normalize.parse_participant")

    # --- Performances détaillées ---
    try:
        perfs = client.performances_detaillees(args.date, num_r, num_c)
    except (PmuError, PmuNotFound) as exc:
        print(f"\n⚠ performances détaillées indisponibles : {exc}")
        perfs = []

    if perfs:
        (sortie / "performances.json").write_text(
            json.dumps(perfs, ensure_ascii=False, indent=2)[:400_000], encoding="utf-8"
        )
        b0 = perfs[0]
        courues = b0.get("coursesCourues") or []
        print(f"\n✓ performances : {len(perfs)} chevaux, "
              f"{len(courues)} courses passées pour le premier")
        if courues:
            print(f"  clés d'une course passée : {sorted(courues[0])}")
            lignes = nz.parse_performances(b0)
            montrer("Performance normalisée", {k: str(v) for k, v in lignes[0].items()}
                    if lignes else {})
        total = sum(len(b.get("coursesCourues") or []) for b in perfs)
        print(f"\n  → cette SEULE course apporte {total} lignes d'historique.")
        print("    C'est l'argument pour amorcer par les performances détaillées")
        print("    plutôt que par un backfill chronologique.")

    print(f"\nPayloads écrits dans {sortie}/")


if __name__ == "__main__":
    main()
