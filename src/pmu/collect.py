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
    stats = {"partants": 0, "cotes": 0, "perfs": 0, "ignores": 0, "expert": 0}
    releve_le = datetime.now(timezone.utc)

    try:
        participants = client.participants(jour, num_r, num_c)
    except PmuNotFound:
        db.journal(conn, "participants", cle, "VIDE", 404)
        return stats
    except PmuError as exc:
        db.journal(conn, "participants", cle, "ERREUR", None, str(exc))
        return stats

    # Avis de l'analyste : un classement complet, publié avant la course
    # et rétroactif. Jamais bloquant — c'est un bonus, pas un socle.
    try:
        sel = client.pronostics_expert(jour, num_r, num_c)
        cribles = client.cribles_expert(jour, num_r, num_c)
        stats["expert"] = db.insert_pronostics_expert(
            conn, course_id, nz.parse_pronostic_expert(sel, cribles))
    except (PmuNotFound, PmuError):
        pass
    except Exception as exc:  # noqa: BLE001
        log.warning("avis expert %s : %s", cle, exc)

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
    # Les compteurs sont initialisés pour que le journal de fin puisse
    # les lire, mais l'ADDITION plus bas ne suppose PAS cette liste :
    # `collecte_course` a gagné une clé (`expert`) en 1.3 et l'oubli ici
    # a fait planter chaque tour de collecte sur un KeyError. Une source
    # de vérité, pas deux à tenir synchronisées.
    total = {"reunions": 0, "courses": 0, "partants": 0, "cotes": 0,
             "perfs": 0, "ignores": 0, "expert": 0}

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
                total[k] = total.get(k, 0) + v

        conn.commit()

    db.journal(conn, "programme", cle_jour, "OK", 200)
    conn.commit()
    log.info(
        "%s — %d réunions, %d courses, %d partants, %d perfs importées%s",
        jour, total["reunions"], total["courses"], total["partants"], total["perfs"],
        f", {total['ignores']} partants ignorés" if total["ignores"] else "",
    )
    return total


SQL_COURSES_SANS_RAPPORT = """
    SELECT c.course_id, c.num_reunion, c.num_ordre, c.date_reunion
      FROM course c
     WHERE c.date_reunion BETWEEN %s AND %s
       AND c.ordre_arrivee IS NOT NULL
       AND NOT EXISTS (SELECT 1 FROM rapport_definitif r
                        WHERE r.course_id = c.course_id)
     ORDER BY c.date_reunion DESC, c.num_reunion, c.num_ordre
     LIMIT %s
"""


def rafraichir_rapports(conn, client: PmuClient, depuis: date, jusqua: date,
                        *, limite: int = 400) -> dict:
    """
    Récupère les rapports payés des courses déjà arrivées.

    Séparé de la collecte du jour à dessein : les rapports ne sont
    publiés qu'APRÈS l'arrivée, souvent avec quelques minutes de retard.
    Les demander pendant la collecte du programme les manquerait
    systématiquement pour la course qui vient de partir.
    """
    rows = conn.execute(SQL_COURSES_SANS_RAPPORT, (depuis, jusqua, limite)).fetchall()
    out = {"courses": 0, "lignes": 0, "vides": 0}
    # ⚠️ `db.connect()` ouvre des curseurs en `dict_row`. Déballer une
    # ligne comme un tuple (`for a, b, c in rows`) itère alors sur les
    # NOMS de colonnes, et la requête suivante reçoit la chaîne
    # « course_id » à la place d'un entier. On accède par clé, toujours.
    for r in rows:
        course_id = r["course_id"]
        num_r, num_c = r["num_reunion"], r["num_ordre"]
        jour = r["date_reunion"]
        try:
            charge = client.rapports_definitifs(jour, num_r, num_c, use_cache=False)
        except (PmuNotFound, PmuError):
            out["vides"] += 1
            continue
        except Exception as exc:  # noqa: BLE001
            log.warning("rapports R%sC%s du %s : %s", num_r, num_c, jour, exc)
            out["vides"] += 1
            continue
        lignes = nz.parse_rapports_definitifs(charge)
        if not lignes:
            out["vides"] += 1
            continue
        try:
            with conn.transaction():
                out["lignes"] += db.insert_rapports_definitifs(conn, course_id, lignes)
            out["courses"] += 1
        except Exception as exc:  # noqa: BLE001
            log.warning("rapports de la course %s ignorés : %s", course_id, exc)
    conn.commit()
    log.info("rapports définitifs : %d courses, %d lignes, %d sans rapport",
             out["courses"], out["lignes"], out["vides"])
    return out


def inspecter_montants(conn, client: PmuClient, jour: date) -> str:
    """
    L'allocation affichée est-elle juste ?

    Une course de Plat à Deauville sur 3 200 m s'est affichée à 259 €,
    ce qui est impossible : une allocation ne descend pas sous quelques
    milliers d'euros. Soit `montantPrix` n'est pas en centimes contrairement
    à ce que suppose `cents_to_eur`, soit il est divisé deux fois.

    On ne devine pas : on met la valeur BRUTE de l'API à côté de celle
    stockée en base, et l'écart saute aux yeux.
    """
    L = ["── Allocation : brut API contre base " + "─" * 22, ""]
    try:
        prog = client.programme(jour, use_cache=False)
    except (PmuNotFound, PmuError) as exc:
        return "\n".join(L + [f"  programme injoignable : {exc}"])

    brut = {}
    for r in (prog.get("reunions") or [])[:3]:
        nr = r.get("numOfficiel")
        for c in (r.get("courses") or [])[:3]:
            nc = c.get("numOrdre")
            if nr is None or nc is None:
                continue
            brut[(nr, nc)] = {k: v for k, v in c.items()
                              if "ontant" in k or k in ("libelle", "distance")}

    rows = conn.execute(
        """SELECT num_reunion, num_ordre, libelle, distance, montant_prix
             FROM course WHERE date_reunion = %s
            ORDER BY num_reunion, num_ordre LIMIT 9""", (jour,)).fetchall()
    en_base = {(r["num_reunion"], r["num_ordre"]): r for r in rows}

    L.append(f"  {'course':<8} {'distance':>9} {'API brut':>14} {'en base':>12}")
    for cle in sorted(set(brut) | set(en_base)):
        b = brut.get(cle, {})
        d = en_base.get(cle, {})
        mp = b.get("montantPrix")
        L.append(f"  R{cle[0]}C{cle[1]:<5} {str(b.get('distance') or d.get('distance') or '—'):>9}"
                 f" {str(mp if mp is not None else '—'):>14}"
                 f" {str(d.get('montant_prix') or '—'):>12}")
    L += ["",
          "  Lecture : « API brut » et « en base » doivent être ÉGAUX.",
          "  montantPrix est en EUROS (relevé en production le 25/08/2026) ;",
          "  un écart d'un facteur 100 signale une conversion de trop.",
          ""]

    # L'allocation des performances passées vient d'un AUTRE endpoint,
    # et rien ne garantit qu'elle suive la même unité. On la montre au
    # lieu de la supposer.
    try:
        anciennes = conn.execute(
            """SELECT allocation, count(*) AS n
                 FROM performance_passee
                WHERE allocation IS NOT NULL
             GROUP BY 1 ORDER BY 2 DESC LIMIT 6""").fetchall()
        L.append("-- Allocation des performances passees " + "-" * 20)
        L.append("  (autre endpoint, unite a confirmer - valeurs les plus frequentes)")
        for r in anciennes:
            L.append(f"  {float(r['allocation']):>12,.2f} EUR   ({r['n']} lignes)")
        L.append("  Si ces montants sont 100 fois trop petits, parse_performances")
        L.append("  applique cents_to_eur a tort, comme le faisait montantPrix.")
    except Exception as exc:  # noqa: BLE001
        conn.rollback()
        L.append(f"  (performances passees illisibles : {exc})")
    return "\n".join(L)


def inspecter_rapports(client: PmuClient, jour: date, num_r: int, num_c: int) -> str:
    """
    Imprime la charge BRUTE d'une course arrivée, puis ce que le parseur
    en tire.

    Raison d'être : la forme de cette réponse n'a pas pu être observée
    depuis l'environnement de développement. Plutôt que de deviner en
    silence et d'enregistrer des zéros, on donne de quoi trancher en
    trente secondes depuis le conteneur.
    """
    import json
    charge = client.rapports_definitifs(jour, num_r, num_c, use_cache=False)
    lignes = nz.parse_rapports_definitifs(charge)
    brut = json.dumps(charge, ensure_ascii=False, indent=1)
    L = ["── Charge brute " + "─" * 44, brut[:3000],
         "", "── Ce que le parseur en tire " + "─" * 31,
         f"  {len(lignes)} ligne(s)"]
    for l in lignes[:12]:
        L.append(f"  {l['type_pari']:<22} {l['combinaison']:<10} "
                 f"rapport={l['rapport']} mise_base={l['mise_base']}")
    if not lignes:
        L.append("  AUCUNE — la forme attendue ne correspond pas.")
        L.append("  Copier la charge brute ci-dessus et la transmettre.")
    return "\n".join(L)


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
                cumul[k] = cumul.get(k, 0) + v
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
    ap.add_argument("mode", choices=["jour", "backfill", "live", "init",
                                     "rapports", "montants"])
    ap.add_argument("--date", type=lambda s: date.fromisoformat(s))
    ap.add_argument("--depuis", type=lambda s: date.fromisoformat(s))
    ap.add_argument("--jusqua", type=lambda s: date.fromisoformat(s))
    ap.add_argument("--cache", default=".cache/pmu")
    ap.add_argument("--rps", type=float, default=2.0)
    ap.add_argument("--sans-perfs", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    # `rapports --verifier R1 C1` imprime la charge brute d'une course
    # arrivée : la forme de cette réponse n'a pas pu être observée
    # depuis l'environnement de développement.
    ap.add_argument("--verifier", nargs=2, metavar=("R", "C"), type=int)
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
        elif args.mode == "montants":
            print(inspecter_montants(conn, client, args.date or date.today()))
        elif args.mode == "rapports":
            if args.verifier:
                jour = args.date or (date.today() - timedelta(days=1))
                print(inspecter_rapports(client, jour, *args.verifier))
                return
            jusqua = args.jusqua or date.today()
            depuis = args.depuis or (jusqua - timedelta(days=60))
            rafraichir_rapports(conn, client, depuis, jusqua)
            from . import evaluate as ev
            print(ev.afficher_surcote(ev.surcote(conn, depuis, jusqua)))
            print()
            print(ev.afficher_rapports(ev.verifier_rapports(conn, depuis, jusqua)))


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
