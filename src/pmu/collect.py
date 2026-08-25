"""
Collecte : programme → participants → performances détaillées.

Trois modes :

  backfill   remonte le temps jour par jour depuis une date
  jour       collecte une journée précise (ou aujourd'hui)
  live       échantillonne les cotes des courses à venir, toutes les N minutes

Le journal de collecte rend l'opération reprenable : relancer un backfill
interrompu ne refait pas ce qui est déjà marqué OK.

--- Sur la profondeur d'historique ---
Ne PAS backfiller des années. `performances-detaillees` renvoie, pour chaque
partant du jour, ses courses passées — souvent 15 à 20 lignes remontant
plusieurs saisons. En collectant un mois de programmes, on récupère
mécaniquement l'historique de plusieurs milliers de chevaux actifs.
C'est 50 à 100 fois moins de requêtes qu'un backfill chronologique, pour
une couverture comparable sur les chevaux qui courent encore — les seuls
sur lesquels on aura à prédire.
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import date, datetime, timedelta, timezone

from . import db, normalize as nz
from .client import PmuClient, PmuError, PmuNotFound

log = logging.getLogger("pmu.collect")


def _personnes(conn, p: dict) -> tuple[int | None, int | None, int | None, int | None]:
    """Insère driver / entraîneur / propriétaire / éleveur, renvoie leurs id."""
    return (
        db.upsert_personne(conn, p.get("driver"), nz.norm_person(p.get("driver"))),
        db.upsert_personne(conn, p.get("entraineur"), nz.norm_person(p.get("entraineur"))),
        db.upsert_personne(conn, p.get("proprietaire"), nz.norm_person(p.get("proprietaire"))),
        db.upsert_personne(conn, p.get("eleveur"), nz.norm_person(p.get("eleveur"))),
    )


def collecte_course(
    conn, client: PmuClient, jour: date, num_r: int, num_c: int,
    course_id: int, ordre_arrivee, *, avec_perfs: bool = True,
) -> dict:
    """Partants + cotes + performances passées d'une course. Renvoie un compte."""
    cle = f"{client.fmt_date(jour)}/R{num_r}/C{num_c}"
    stats = {"partants": 0, "cotes": 0, "perfs": 0, "ignores": 0}
    releve_le = datetime.now(timezone.utc)

    try:
        participants = client.participants(jour, num_r, num_c)
    except PmuNotFound:
        db.journal(conn, "participants", cle, "VIDE", 404)
        return stats
    except PmuError as exc:
        db.journal(conn, "participants", cle, "ERREUR", None, str(exc))
        return stats

    # numPmu → idCheval, pour rattacher les performances passées : l'endpoint
    # performances-detaillees n'expose PAS idCheval, seulement numPmu.
    id_par_num: dict[int, str] = {}

    for raw in participants:
        p = nz.parse_participant(raw, ordre_arrivee)
        if p["num_pmu"] is None:
            continue
        cotes = nz.parse_cotes(raw, releve_le)
        try:
            # Point de reprise par partant : une ligne malformée est
            # écartée sans emporter la course entière ni laisser la
            # transaction en échec. Sans ça, un seul champ inattendu
            # coûte une journée de collecte.
            with conn.transaction():
                id_driver, id_entr, id_prop, id_elev = _personnes(conn, p)
                db.upsert_cheval(conn, p, id_eleveur=id_elev)
                db.upsert_partant(conn, course_id, p, id_driver, id_entr, id_prop)
                db.insert_cotes(conn, course_id, cotes)
        except Exception as exc:  # noqa: BLE001
            log.warning("partant %s de %s ignoré : %s", p["num_pmu"], cle, exc)
            stats["ignores"] += 1
            continue
        if p["id_cheval"]:
            id_par_num[p["num_pmu"]] = p["id_cheval"]
        stats["partants"] += 1
        stats["cotes"] += len(cotes)

    db.journal(conn, "participants", cle, "OK", 200)

    if avec_perfs and not db.deja_collecte(conn, "perf", cle):
        try:
            blocs = client.performances_detaillees(jour, num_r, num_c)
        except (PmuNotFound, PmuError) as exc:
            db.journal(conn, "perf", cle, "VIDE", None, str(exc))
        else:
            lignes = []
            for bloc in blocs:
                num = nz.as_int(bloc.get("numPmu"))
                lignes += nz.parse_performances(bloc, id_par_num.get(num))
            try:
                with conn.transaction():
                    stats["perfs"] = db.insert_performances(conn, lignes)
                db.journal(conn, "perf", cle, "OK", 200)
            except Exception as exc:  # noqa: BLE001
                log.warning("performances de %s ignorées : %s", cle, exc)
                db.journal(conn, "perf", cle, "ERREUR", None, str(exc))

    return stats


def collecte_jour(conn, client: PmuClient, jour: date, *, avec_perfs: bool = True,
                  force: bool = False) -> dict:
    cle_jour = client.fmt_date(jour)
    total = {"reunions": 0, "courses": 0, "partants": 0, "cotes": 0,
             "perfs": 0, "ignores": 0}

    try:
        # Le programme se recharge toujours : les arrivées et les statuts
        # évoluent au fil de la journée.
        prog = client.programme(jour, use_cache=False)
    except PmuNotFound:
        db.journal(conn, "programme", cle_jour, "VIDE", 404)
        log.info("%s — pas de programme", jour)
        return total
    except PmuError as exc:
        db.journal(conn, "programme", cle_jour, "ERREUR", None, str(exc))
        log.warning("%s — programme KO : %s", jour, exc)
        return total

    for raw_reunion in prog.get("reunions") or []:
        hip = nz.parse_hippodrome(raw_reunion)
        if hip:
            db.upsert_hippodrome(conn, hip)
        r = nz.parse_reunion(raw_reunion)
        # Ceinture ET bretelles : on a DEMANDÉ le programme de `jour`,
        # c'est donc `jour` la date de référence. Ne pas dépendre d'un
        # horodatage à retraduire en fuseau — la source d'autorité est
        # l'URL qu'on vient d'appeler.
        r["date_reunion"] = jour
        if r["num_officiel"] is None:
            continue
        db.upsert_reunion(conn, r)
        total["reunions"] += 1

        for raw_course in raw_reunion.get("courses") or []:
            c = nz.parse_course(raw_course, r["date_reunion"])
            if c["num_reunion"] is None:
                c["num_reunion"] = r["num_officiel"]
            if c["num_ordre"] is None:
                continue
            course_id = db.upsert_course(conn, c)
            total["courses"] += 1
            if course_id is None:
                continue

            cle = f"{cle_jour}/R{c['num_reunion']}/C{c['num_ordre']}"
            # Une course déjà collectée ET arrivée n'a plus rien à livrer.
            if not force and c["ordre_arrivee"] and db.deja_collecte(conn, "participants", cle):
                continue

            s = collecte_course(
                conn, client, jour, c["num_reunion"], c["num_ordre"],
                course_id, c["ordre_arrivee"], avec_perfs=avec_perfs,
            )
            for k, v in s.items():
                total[k] += v

        conn.commit()

    db.journal(conn, "programme", cle_jour, "OK", 200)
    conn.commit()
    log.info(
        "%s — %d réunions, %d courses, %d partants, %d perfs importées%s",
        jour, total["reunions"], total["courses"], total["partants"], total["perfs"],
        f", {total['ignores']} partants ignorés" if total["ignores"] else "",
    )
    return total


def backfill(conn, client: PmuClient, depuis: date, jusqua: date,
             *, avec_perfs: bool = True, force: bool = False) -> dict:
    """Remonte le temps du plus récent au plus ancien."""
    cumul = {"jours": 0, "reunions": 0, "courses": 0, "partants": 0, "cotes": 0, "perfs": 0}
    jour = jusqua
    while jour >= depuis:
        if not force and db.deja_collecte(conn, "programme", client.fmt_date(jour)):
            log.debug("%s déjà fait", jour)
        else:
            t = collecte_jour(conn, client, jour, avec_perfs=avec_perfs, force=force)
            for k, v in t.items():
                cumul[k] += v
        cumul["jours"] += 1
        jour -= timedelta(days=1)
    db.link_genealogie(conn)
    conn.commit()
    return cumul


def live_cotes(conn, client: PmuClient, jour: date | None = None,
               *, fenetre_min: int = 45, periode_s: int = 300,
               duree_max_s: int = 3600 * 6) -> None:
    """
    Échantillonne les cotes des courses qui partent dans les `fenetre_min`
    prochaines minutes. C'est ce qui alimente la dérive de cote — impossible
    à reconstituer après coup, il FAUT l'enregistrer en direct.
    """
    jour = jour or datetime.now(timezone.utc).date()
    fin = time.monotonic() + duree_max_s

    while time.monotonic() < fin:
        maintenant = datetime.now(timezone.utc)
        rows = conn.execute(
            """
            SELECT course_id, num_reunion, num_ordre, heure_depart
              FROM course
             WHERE date_reunion = %s
               AND heure_depart IS NOT NULL
               AND heure_depart BETWEEN %s AND %s
               AND ordre_arrivee IS NULL
             ORDER BY heure_depart
            """,
            (jour, maintenant, maintenant + timedelta(minutes=fenetre_min)),
        ).fetchall()

        for row in rows:
            try:
                participants = client.participants(
                    jour, row["num_reunion"], row["num_ordre"], use_cache=False
                )
            except (PmuNotFound, PmuError):
                continue
            releve = datetime.now(timezone.utc)
            for raw in participants:
                db.insert_cotes(conn, row["course_id"], nz.parse_cotes(raw, releve))
        conn.commit()
        if rows:
            log.info("cotes relevées sur %d course(s)", len(rows))
        time.sleep(periode_s)


# ---------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Collecte PMU")
    ap.add_argument("mode", choices=["jour", "backfill", "live", "init"])
    ap.add_argument("--date", type=lambda s: date.fromisoformat(s))
    ap.add_argument("--depuis", type=lambda s: date.fromisoformat(s))
    ap.add_argument("--jusqua", type=lambda s: date.fromisoformat(s))
    ap.add_argument("--cache", default=".cache/pmu")
    ap.add_argument("--rps", type=float, default=2.0)
    ap.add_argument("--sans-perfs", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    )

    client = PmuClient(cache_dir=args.cache, rps=args.rps)

    with db.connect() as conn:
        if args.mode == "init":
            db.apply_schema(conn)
            log.info("schéma appliqué")
            return

        # Vérifie que la version d'API répond avant de lancer quoi que ce soit.
        sonde = args.date or args.jusqua or (date.today() - timedelta(days=1))
        try:
            client.programme(sonde, use_cache=False)
        except (PmuError, PmuNotFound):
            log.warning("version client %d muette — détection automatique", client.client_version)
            client.detect_client_version(sonde)

        if args.mode == "jour":
            collecte_jour(conn, client, args.date or date.today(),
                          avec_perfs=not args.sans_perfs, force=args.force)
            db.link_genealogie(conn)
            conn.commit()
        elif args.mode == "backfill":
            jusqua = args.jusqua or date.today()
            depuis = args.depuis or (jusqua - timedelta(days=30))
            cumul = backfill(conn, client, depuis, jusqua,
                             avec_perfs=not args.sans_perfs, force=args.force)
            log.info("backfill terminé : %s", cumul)
        elif args.mode == "live":
            live_cotes(conn, client, args.date)


if __name__ == "__main__":
    main()


def rafraichir_arrivees(conn, client, jour) -> int:
    """
    Met à jour les arrivées du jour au moindre coût.

    UN SEUL appel à l'API — le programme de la journée, qui porte
    l'ordre d'arrivée de chaque course — puis une projection SQL vers
    les partants. Aucun appel par course, donc c'est assez léger pour
    tourner à chaque tour du planificateur.

    C'est ce qui manquait : les arrivées ne descendaient au niveau des
    partants qu'à 23 h 30, et entre-temps le tableau de bord affichait
    tous les favoris comme battus.
    """
    try:
        prog = client.programme(jour, use_cache=False)
    except (PmuNotFound, PmuError) as exc:
        log.warning("programme du %s indisponible : %s", jour, exc)
        return 0

    reunions = nz.dig(prog, "programme", "reunions", default=[]) or []
    for raw_r in reunions:
        r = nz.parse_reunion(raw_r)
        for raw_course in raw_r.get("courses") or []:
            c = nz.parse_course(raw_course, r["date_reunion"])
            if c["num_reunion"] is None:
                c["num_reunion"] = r["num_officiel"]
            if c["num_ordre"] is None or not c["ordre_arrivee"]:
                continue
            db.upsert_course(conn, c)
    n = db.propager_arrivees(conn, jour)
    conn.commit()
    return n
