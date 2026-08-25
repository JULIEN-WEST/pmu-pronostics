"""
Ordonnanceur — le service qui fait tourner la pile tout seul.

    python -m pmu.planificateur

Rien n'est à lancer à la main. Au premier démarrage, le conteneur :

    1. vérifie l'API PMU, la base et le broker MQTT, et écrit un compte
       rendu lisible dans ses journaux ;
    2. rattrape l'historique (PMU_BACKFILL_JOURS jours, 30 par défaut) ;
    3. entraîne le premier modèle dès qu'il a assez de données ;
    4. entre en régime de croisière.

L'amorçage est REPRENABLE : si le conteneur redémarre au milieu du
rattrapage, il reprend là où il en était. Le drapeau de fin n'est posé
qu'une fois tout terminé.

Journée type, ensuite :

    08h00   collecte du programme du jour
    08h05   pronostics + publication MQTT
    puis    toutes les 5 min : relevé des cotes des courses à venir,
            re-pronostic si une cote a bougé, publication
    23h30   collecte des arrivées (rattrapage)
    dimanche 04h00  ré-entraînement complet

Le relevé de cotes est le seul travail réellement contraint en temps :
la dérive de cote ne se reconstitue jamais après coup.
"""

from __future__ import annotations

import logging
import os
import time
import traceback
from datetime import date, datetime, time as dtime, timedelta, timezone

import psycopg

from . import collect, meteo as mt, db, mqtt_ha
from .client import PmuClient, PmuError, PmuNotFound
from .predict import entrainer, pronostiquer

log = logging.getLogger("pmu.planificateur")

FUSEAU = timezone(timedelta(hours=2))            # Europe/Paris en été
CACHE = os.environ.get("PMU_CACHE", "/data/cache")
RPS = float(os.environ.get("PMU_RPS", "2"))
PERIODE_COTES = int(os.environ.get("PMU_PERIODE_COTES", "300"))
FENETRE_COTES = int(os.environ.get("PMU_FENETRE_COTES", "45"))
MQTT_ACTIF = os.environ.get("MQTT_HOST", "") != ""
JOUR_REENTRAINEMENT = int(os.environ.get("PMU_JOUR_REENTRAINEMENT", "6"))  # 6 = dimanche

# Amorçage automatique. Mettre PMU_BACKFILL_JOURS=0 pour le désactiver et
# tout piloter à la main.
BACKFILL_JOURS = int(os.environ.get("PMU_BACKFILL_JOURS", "30"))
MIN_PARTANTS = int(os.environ.get("PMU_MIN_PARTANTS", "15000"))


def _maintenant() -> datetime:
    return datetime.now(FUSEAU)


def _sur(message: str) -> None:
    """Journalise une exception sans tuer la boucle."""
    log.error("%s\n%s", message, traceback.format_exc())


def _cadre(titre: str, lignes: list[str]) -> str:
    """Encadré lisible dans les journaux Portainer."""
    largeur = max([len(titre)] + [len(l) for l in lignes]) + 4
    haut = "┌" + "─" * largeur + "┐"
    bas = "└" + "─" * largeur + "┘"
    # « │ » + 2 espaces = 3 caractères avant le texte, 1 après :
    # le remplissage vaut donc largeur − 3, pas largeur − 2.
    corps = [f"│  {titre.ljust(largeur - 2)}│", "├" + "─" * largeur + "┤"]
    corps += [f"│  {l.ljust(largeur - 2)}│" for l in lignes]
    return "\n" + "\n".join([haut] + corps + [bas])


# ---------------------------------------------------------------------
# Auto-diagnostic
# ---------------------------------------------------------------------

def diagnostic(client: PmuClient) -> bool:
    """
    Vérifie les trois dépendances externes et écrit un compte rendu en
    clair. Renvoie False si l'API PMU est injoignable — dans ce cas rien
    ne sert de continuer, et le message dit quoi faire.
    """
    lignes: list[str] = []
    ok_api = False

    # -- Base ------------------------------------------------------
    try:
        with db.connect() as conn:
            conn.execute("SELECT 1")
        lignes.append("BASE DE DONNEES  ..... OK")
    except Exception as exc:  # noqa: BLE001
        lignes.append("BASE DE DONNEES  ..... ECHEC")
        lignes.append(f"   {str(exc)[:70]}")
        lignes.append("   -> verifier la variable POSTGRES_PASSWORD de la stack")

    # -- API PMU ---------------------------------------------------
    sonde = date.today() - timedelta(days=1)
    try:
        prog = client.programme(sonde, use_cache=False)
        n = len(prog.get("reunions") or [])
        lignes.append(f"API PMU          ..... OK  ({n} reunions le {sonde})")
        ok_api = True
    except (PmuError, PmuNotFound):
        try:
            version = client.detect_client_version(sonde)
            lignes.append(f"API PMU          ..... OK  (version {version} detectee)")
            ok_api = True
        except PmuError:
            lignes.append("API PMU          ..... ECHEC")
            lignes.append("   aucune version /client/<n>/ ne repond")
            lignes.append("   -> l'API a change de forme, previens Claude")

    # -- MQTT ------------------------------------------------------
    if MQTT_ACTIF:
        try:
            detail = mqtt_ha.verifier()
            lignes.append(f"BROKER MQTT      ..... OK  ({detail})")
        except Exception as exc:  # noqa: BLE001
            lignes.append("BROKER MQTT      ..... ECHEC")
            for morceau in str(exc).split(" — "):
                lignes.append(f"   {morceau[:70]}")
            lignes.append("   -> la pile fonctionne quand meme, sans entites HA")
    else:
        lignes.append("BROKER MQTT      ..... DESACTIVE (MQTT_HOST vide)")
        lignes.append("   -> aucune entite ne remontera dans Home Assistant")

    log.info(_cadre("AUTO-DIAGNOSTIC AU DEMARRAGE", lignes))
    return ok_api


# ---------------------------------------------------------------------
# Amorçage
# ---------------------------------------------------------------------

def _volumetrie(conn) -> int:
    row = conn.execute("SELECT count(*) AS n FROM partant").fetchone()
    return int(row["n"]) if row else 0


def amorcer(conn, client: PmuClient) -> None:
    """
    Rattrapage d'historique puis premier entraînement, une seule fois.

    Le drapeau est posé dans `collecte_journal`, donc un redémarrage en
    cours de route reprend proprement : les jours déjà collectés sont
    sautés et le drapeau n'existe pas encore.
    """
    if BACKFILL_JOURS <= 0:
        log.info("amorçage automatique désactivé (PMU_BACKFILL_JOURS=0)")
        return
    if db.deja_collecte(conn, "amorcage", "termine"):
        return

    jusqua = date.today()
    depuis = jusqua - timedelta(days=BACKFILL_JOURS)
    log.info(_cadre("PREMIER DEMARRAGE - AMORCAGE AUTOMATIQUE", [
        f"Rattrapage de {BACKFILL_JOURS} jours : {depuis} -> {jusqua}",
        "",
        "Compter 1 a 2 heures. C'est normal et ca ne se reproduira pas.",
        "Tu peux fermer cette fenetre, le conteneur continue.",
        "",
        "Ensuite le premier modele s'entrainera tout seul,",
        f"des que la base contiendra {MIN_PARTANTS} partants.",
    ]))

    # Signale à Home Assistant que la collecte tourne, pour que la vue
    # n'affiche pas un vide inexpliqué pendant deux heures.
    if MQTT_ACTIF:
        try:
            mqtt_ha.publier_amorcage(depuis, jusqua)
        except Exception:  # noqa: BLE001
            pass

    jour = jusqua
    while jour >= depuis:
        if not db.deja_collecte(conn, "programme", client.fmt_date(jour)):
            try:
                collect.collecte_jour(conn, client, jour, avec_perfs=True)
                try:
                    mt.enrichir(conn, jour)
                except Exception as exc:  # noqa: BLE001
                    log.warning("météo non collectée pour %s : %s", jour, exc)
                    conn.rollback()
            except Exception:  # noqa: BLE001
                # Le rollback n'est pas optionnel : après une erreur SQL,
                # psycopg refuse toute instruction suivante tant que la
                # transaction n'est pas annulée. Sans lui, un échec sur un
                # jour condamne silencieusement tous les jours suivants.
                conn.rollback()
                _sur(f"collecte du {jour} en échec, on continue")
        jour -= timedelta(days=1)

    db.link_genealogie(conn)
    conn.commit()

    n = _volumetrie(conn)
    log.info("rattrapage terminé : %d partants en base", n)

    if n < MIN_PARTANTS:
        log.warning(_cadre("PAS ASSEZ DE DONNEES POUR ENTRAINER", [
            f"{n} partants collectes, {MIN_PARTANTS} attendus.",
            "",
            "Le conteneur va continuer a collecter chaque jour et",
            "reessaiera tout seul. Pour aller plus vite, augmenter",
            "PMU_BACKFILL_JOURS dans les variables de la stack.",
        ]))
        return

    log.info("── premier entraînement (5 à 15 min)")
    try:
        entrainer(conn, avec_marche=False)
        entrainer(conn, avec_marche=True)
    except Exception:  # noqa: BLE001
        _sur("premier entraînement en échec")
        return

    db.journal(conn, "amorcage", "termine", "OK")
    conn.commit()
    log.info(_cadre("AMORCAGE TERMINE", [
        f"{n} partants en base, modeles entraines.",
        "",
        "Le rapport d'evaluation est dans /data/modeles/*/rapport.txt",
        "La pile passe en regime de croisiere.",
    ]))


def rattraper_entrainement(conn) -> bool:
    """
    Appelé chaque tour tant que l'amorçage n'est pas terminé : si la
    collecte quotidienne a fini par franchir le seuil, on entraîne.
    """
    if BACKFILL_JOURS <= 0 or db.deja_collecte(conn, "amorcage", "termine"):
        return False
    n = _volumetrie(conn)
    if n < MIN_PARTANTS:
        return False
    log.info("seuil atteint (%d partants) — entraînement du premier modèle", n)
    try:
        entrainer(conn, avec_marche=False)
        entrainer(conn, avec_marche=True)
    except Exception:  # noqa: BLE001
        _sur("entraînement de rattrapage en échec")
        return False
    db.journal(conn, "amorcage", "termine", "OK")
    conn.commit()
    return True


# ---------------------------------------------------------------------

class Planificateur:
    def __init__(self, client: PmuClient) -> None:
        self.client = client
        self.dernier_programme: date | None = None
        self.dernier_reentrainement: date | None = None
        self.derniere_arrivee: date | None = None

    # -- tâches ------------------------------------------------------

    def collecte_programme(self, conn, jour: date) -> None:
        log.info("── collecte du programme %s", jour)
        collect.collecte_jour(conn, self.client, jour, avec_perfs=True)
        # Météo : bonus, jamais bloquant. Une erreur ici ne doit pas
        # empêcher la collecte de se terminer.
        try:
            mt.enrichir(conn, jour)
        except Exception as exc:  # noqa: BLE001
            log.warning("météo non collectée pour %s : %s", jour, exc)
            conn.rollback()

        db.link_genealogie(conn)
        conn.commit()

    def pronostics(self, conn, jour: date) -> None:
        try:
            tout = pronostiquer(conn, jour)
        except RuntimeError as exc:
            log.warning("pronostic impossible : %s", exc)
            return
        if MQTT_ACTIF and len(tout):
            try:
                mqtt_ha.publier(jour)
            except Exception:  # noqa: BLE001
                _sur("publication MQTT en échec")

    def cotes(self, conn, jour: date) -> int:
        """Relève les cotes des courses proches du départ. Renvoie le nombre traité."""
        from . import normalize as nz

        maintenant = datetime.now(timezone.utc)
        rows = conn.execute(
            """
            SELECT course_id, num_reunion, num_ordre
              FROM course
             WHERE date_reunion = %s
               AND heure_depart BETWEEN %s AND %s
               AND ordre_arrivee IS NULL
             ORDER BY heure_depart
            """,
            (jour, maintenant, maintenant + timedelta(minutes=FENETRE_COTES)),
        ).fetchall()

        for row in rows:
            try:
                participants = self.client.participants(
                    jour, row["num_reunion"], row["num_ordre"], use_cache=False)
            except Exception:  # noqa: BLE001
                continue
            releve = datetime.now(timezone.utc)
            for raw in participants:
                db.insert_cotes(conn, row["course_id"], nz.parse_cotes(raw, releve))
        conn.commit()
        if rows:
            log.info("cotes relevées sur %d course(s)", len(rows))
        return len(rows)

    def arrivees(self, conn, jour: date) -> None:
        log.info("── rattrapage des arrivées %s", jour)
        collect.collecte_jour(conn, self.client, jour, avec_perfs=False, force=True)

    def reentrainer(self, conn) -> None:
        log.info("── ré-entraînement hebdomadaire")
        try:
            entrainer(conn, avec_marche=False)
            entrainer(conn, avec_marche=True)
        except Exception:  # noqa: BLE001
            _sur("ré-entraînement en échec")

    # -- boucle ------------------------------------------------------

    def tour(self, conn) -> None:
        n = _maintenant()
        jour = n.date()

        if self.dernier_programme != jour and n.time() >= dtime(8, 0):
            self.collecte_programme(conn, jour)
            self.dernier_programme = jour
            self.pronostics(conn, jour)

        # Tant que le premier modèle n'existe pas, on retente dès que la
        # base a suffisamment grossi.
        if rattraper_entrainement(conn):
            self.pronostics(conn, jour)

        if (self.dernier_reentrainement != jour
                and n.weekday() == JOUR_REENTRAINEMENT and n.time() >= dtime(4, 0)):
            self.reentrainer(conn)
            self.dernier_reentrainement = jour

        if self.dernier_programme == jour and self.cotes(conn, jour) > 0:
            self.pronostics(conn, jour)

        if self.derniere_arrivee != jour and n.time() >= dtime(23, 30):
            self.arrivees(conn, jour)
            self.derniere_arrivee = jour

    def boucler(self) -> None:
        log.info("régime de croisière — vérification toutes les %d s", PERIODE_COTES)
        amorce = True
        while True:
            try:
                with db.connect() as conn:
                    if amorce:
                        # Au démarrage on ne attend pas 8 h : on rattrape.
                        jour = _maintenant().date()
                        self.collecte_programme(conn, jour)
                        self.dernier_programme = jour
                        self.pronostics(conn, jour)
                        amorce = False
                    else:
                        self.tour(conn)
            except Exception:  # noqa: BLE001 — la boucle ne meurt jamais
                _sur("tour en échec, on continue")
            time.sleep(PERIODE_COTES)


# ---------------------------------------------------------------------

def _veille(raison: str) -> None:
    """
    Met le conteneur en veille au lieu de sortir en erreur.

    Sortir déclencherait le redémarrage automatique de Portainer, donc une
    boucle : le message d'erreur défilerait sans fin et deviendrait
    illisible. En veille, il reste en haut du journal, là où on le lit.
    """
    log.error("%s — le conteneur reste en veille. Corrige, puis redémarre-le "
              "depuis Portainer.", raison)
    while True:
        time.sleep(3600)


def _preparer_base() -> bool:
    """
    Applique le schéma, en attendant que PostgreSQL soit prêt.

    ⚠️ On ne réessaie QUE sur une erreur de connexion. Un `except
    Exception` large ici transformerait toute erreur de configuration en
    « base pas encore prête », répété douze fois puis abandonné — le vrai
    message, celui qui dit quoi faire, ne serait jamais affiché. C'est
    exactement ce qui est arrivé avec le contrôle de schéma.
    """
    from .predict import SQL_TABLE

    chemin = os.environ.get("PMU_SCHEMA", "sql/001_schema.sql")
    derniere: Exception | None = None

    jeton_reset = os.environ.get("PMU_RESET", "").strip()

    for tentative in range(12):
        try:
            with db.connect() as conn:
                remis_a_zero = db.reinitialiser(conn, jeton_reset)
                db.apply_schema(conn, chemin)
                conn.execute(SQL_TABLE)
                if remis_a_zero:
                    # Consigné APRÈS recréation du schéma, sinon la trace
                    # partirait avec la table qu'on vient de supprimer.
                    db.journal(conn, "reset", jeton_reset, "OK")
                conn.commit()
            return True
        except psycopg.OperationalError as exc:
            derniere = exc
            log.info("base pas encore prête, nouvelle tentative dans 5 s (%d/12)",
                     tentative + 1)
            time.sleep(5)
        except RuntimeError as exc:
            # Erreur de configuration : réessayer n'y changera rien.
            log.error("%s", exc)
            return False
        except Exception:  # noqa: BLE001
            _sur("préparation de la base en échec")
            return False

    log.error("base injoignable après 1 minute : %s", derniere)
    return False


def main() -> None:
    logging.basicConfig(
        level=os.environ.get("PMU_LOG", "INFO"),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    )

    client = PmuClient(cache_dir=CACHE, rps=RPS)

    if not _preparer_base():
        _veille("base de données inutilisable")

    if not diagnostic(client):
        _veille("API PMU injoignable")

    with db.connect() as conn:
        amorcer(conn, client)

    Planificateur(client).boucler()


if __name__ == "__main__":
    main()
