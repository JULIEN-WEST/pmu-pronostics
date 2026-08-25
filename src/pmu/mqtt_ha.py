"""
Publication vers Home Assistant par découverte MQTT.

Pourquoi MQTT plutôt qu'un capteur REST côté HA :

  - les entités apparaissent toutes seules, groupées sous un appareil
    « Pronostics PMU », sans toucher à configuration.yaml ;
  - les messages sont RETENUS : après un redémarrage de HA, ou si la pile
    Docker est éteinte, les dernières valeurs sont toujours là. Un capteur
    REST, lui, passe à `unavailable` dès que l'API ne répond plus ;
  - pas de scrutation : HA reçoit la mise à jour au moment où elle a lieu.

Le broker est celui de zigbee2mqtt — rien de nouveau à installer.

⚠️ Limite d'attributs. Home Assistant tronque les attributs d'entité au-delà
de 16 ko et ralentit sensiblement bien avant. La charge utile est donc
plafonnée ici même, et pas seulement côté API.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import date, datetime, timezone

import paho.mqtt.client as mqtt

from . import db
from .predict import lire_pronostics

log = logging.getLogger("pmu.mqtt")

HOTE = os.environ.get("MQTT_HOST", "192.168.1.153")
PORT = int(os.environ.get("MQTT_PORT", "1883"))
UTILISATEUR = os.environ.get("MQTT_USER") or None
MOTDEPASSE = os.environ.get("MQTT_PASSWORD") or None
PREFIXE = os.environ.get("MQTT_DISCOVERY_PREFIX", "homeassistant")
MODELE = os.environ.get("PMU_MODELE", "sans_marche")

BASE = "pmu_pronostics"
T_ETAT = f"{BASE}/etat"
T_DISPO = f"{BASE}/dispo"

APPAREIL = {
    "identifiers": ["pmu_pronostics"],
    "name": "Pronostics PMU",
    "manufacturer": "pmu-pronostics",
    "model": "Modèle calibré multi-disciplines",
    "sw_version": "0.1.0",
}

# clé, nom, icône, unité, template de valeur
CAPTEURS = [
    ("courses", "Courses du jour", "mdi:horse-variant", None,
     "{{ value_json.courses }}"),
    ("prochaine", "Prochaine course", "mdi:clock-fast", None,
     "{{ value_json.prochaine.libelle if value_json.prochaine else 'aucune' }}"),
    ("prochaine_depart", "Départ prochaine course", "mdi:timer-outline", None,
     "{{ value_json.prochaine.depart if value_json.prochaine else None }}"),
    ("favori", "Favori du modèle", "mdi:trophy-outline", None,
     "{{ value_json.prochaine.selection[0].cheval if value_json.prochaine else 'aucun' }}"),
    ("favori_proba", "Probabilité du favori", "mdi:percent-outline", "%",
     "{{ (value_json.prochaine.selection[0].proba * 100) | round(1)"
     " if value_json.prochaine else 0 }}"),
    ("confiance", "Confiance", "mdi:gauge", "%",
     "{{ (value_json.prochaine.confiance * 100) | round(1)"
     " if value_json.prochaine else 0 }}"),
    ("age_calcul", "Âge du calcul", "mdi:update", "h",
     "{{ value_json.age_heures | default(0) }}"),
]


class MqttIndisponible(RuntimeError):
    pass


def _client() -> mqtt.Client:
    """
    Connexion au broker, avec attente EFFECTIVE de l'acquittement.

    `connect()` rend la main avant que le broker n'ait répondu. Publier
    dans la foulée puis se déconnecter fait perdre les messages : ils
    partent dans une file que personne ne vide. C'est ce qui faisait que
    la découverte était « publiée » sans qu'aucune entité n'apparaisse
    dans Home Assistant.
    """
    etat: dict = {"code": None}

    def on_connect(client, userdata, flags, reason_code, properties=None):
        etat["code"] = reason_code
        if reason_code != 0:
            log.error("connexion MQTT refusée par %s : %s", HOTE, reason_code)

    c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="pmu-pronostics")
    c.on_connect = on_connect
    if UTILISATEUR:
        c.username_pw_set(UTILISATEUR, MOTDEPASSE)
    # Testament : si le conteneur meurt, HA passe les entités en indisponible
    # au lieu d'afficher éternellement le dernier pronostic comme s'il était frais.
    c.will_set(T_DISPO, "offline", retain=True)
    c.connect(HOTE, PORT, keepalive=60)
    c.loop_start()

    for _ in range(50):                     # 5 s maximum
        if c.is_connected():
            return c
        if etat["code"] not in (None, 0):
            break
        time.sleep(0.1)

    c.loop_stop()
    c.disconnect()
    raise MqttIndisponible(
        f"broker {HOTE}:{PORT} injoignable ou refuse la connexion "
        f"(code {etat['code']}) — vérifier MQTT_HOST / MQTT_USER / MQTT_PASSWORD"
    )


def _publier(c: mqtt.Client, topic: str, charge: str, *, retain: bool = True):
    """
    Publie en QoS 1 et rend l'objet de suivi.

    QoS 0 ne garantit rien : le message est « envoyé » même si le broker
    ne l'a jamais reçu. QoS 1 impose un acquittement, qu'on attend ensuite
    dans `_attendre()`.
    """
    return c.publish(topic, charge, qos=1, retain=retain)


def _attendre(infos: list, quoi: str) -> None:
    """Bloque jusqu'à l'acquittement effectif de chaque message."""
    perdus = 0
    for info in infos:
        try:
            info.wait_for_publish(timeout=5)
        except (ValueError, RuntimeError):
            perdus += 1
        else:
            if not info.is_published():
                perdus += 1
    if perdus:
        log.warning("%s : %d message(s) sur %d non acquittés par le broker",
                    quoi, perdus, len(infos))
    else:
        log.info("%s : %d message(s) acquittés", quoi, len(infos))


def publier_decouverte(c: mqtt.Client) -> list:
    infos = []
    for cle, nom, icone, unite, tpl in CAPTEURS:
        conf = {
            "name": nom,
            "unique_id": f"pmu_pronostics_{cle}",
            "object_id": f"pmu_{cle}",
            "state_topic": T_ETAT,
            "availability_topic": T_DISPO,
            "value_template": tpl,
            # Toute la journée est portée en attributs par le capteur
            # « courses » : c'est lui que la vue Lovelace lit.
            "device": APPAREIL,
        }
        if unite:
            conf["unit_of_measurement"] = unite
        if icone:
            conf["icon"] = icone
        if cle == "courses":
            conf["json_attributes_topic"] = T_ETAT
            conf["json_attributes_template"] = "{{ value_json | tojson }}"
        if cle == "prochaine_depart":
            conf["device_class"] = "timestamp"
        infos.append(_publier(c, f"{PREFIXE}/sensor/pmu_pronostics/{cle}/config",
                              json.dumps(conf, ensure_ascii=False)))

    binaire = {
        "name": "Pronostics à jour",
        "unique_id": "pmu_pronostics_frais",
        "object_id": "pmu_frais",
        "state_topic": T_ETAT,
        "availability_topic": T_DISPO,
        "value_template": "{{ 'ON' if value_json.frais else 'OFF' }}",
        "device_class": "problem",
        "payload_on": "OFF",     # « problème » = PAS frais
        "payload_off": "ON",
        "device": APPAREIL,
    }
    infos.append(_publier(c, f"{PREFIXE}/binary_sensor/pmu_pronostics/frais/config",
                          json.dumps(binaire, ensure_ascii=False)))
    return infos


def construire_charge(conn, jour: date | None = None) -> dict:
    jour = jour or date.today()
    courses = lire_pronostics(conn, jour, MODELE)

    maintenant = datetime.now(timezone.utc)
    futures = [c for c in courses
               if c["depart"] and not c["arrivee_connue"]
               and datetime.fromisoformat(c["depart"]) > maintenant]
    proch = min(futures, key=lambda c: c["depart"]) if futures else None

    age_h = None
    if courses and courses[0].get("calcule_le"):
        age_h = round(
            (maintenant - datetime.fromisoformat(courses[0]["calcule_le"])).total_seconds()
            / 3600, 1)

    def _motifs(s: dict) -> list:
        """
        Justification, RADICALEMENT taillée pour MQTT.

        Les attributs d'entité Home Assistant partent dans la base du
        recorder à chaque changement d'état. Y déverser les faits
        complets de quinze partants sur quarante courses ferait gonfler
        la base de plusieurs mégaoctets par jour, pour un texte que
        personne ne lit. Trois motifs, deux détails chacun, et
        seulement pour la course à venir : le reste se consulte sur la
        page /vue, qui n'a pas cette contrainte.
        """
        return [
            {"titre": m.get("titre"), "sens": m.get("sens"),
             "details": [str(d)[:90] for d in (m.get("details") or [])[:2]]}
            for m in (s.get("motifs") or [])[:3]
        ]

    def compacter(c: dict, n_partants: int, *, avec_motifs: bool = False) -> dict:
        selection = []
        for s in c["selection"][:n_partants]:
            ligne = {"num": s["num"], "cheval": s["cheval"],
                     "proba": s["proba"], "cote": s["cote"],
                     "valeur": s["valeur"], "arrivee": s["arrivee"]}
            if avec_motifs:
                ligne["driver"] = s.get("driver")
                ligne["musique"] = s.get("musique")
                ligne["motifs"] = _motifs(s)
            selection.append(ligne)
        return {
            "code": c["code"],
            "hippodrome": c["hippodrome"],
            "libelle": (c["libelle"] or "")[:44],
            "discipline": c["discipline"],
            "distance": c["distance"],
            "depart": c["depart"],
            "partants": c["partants"],
            "confiance": round(c["confiance"], 3),
            "arrivee_connue": c["arrivee_connue"],
            "publiable": c.get("publiable", True),
            "selection": selection,
        }

    return {
        "date": jour.isoformat(),
        "modele": MODELE,
        "courses": len(courses),
        "age_heures": age_h if age_h is not None else 999,
        "frais": age_h is not None and age_h < 24,
        "maj": maintenant.isoformat(),
        # La prochaine course est détaillée et justifiée ; les autres
        # sont résumées, sans motifs.
        "prochaine": compacter(proch, 8, avec_motifs=True) if proch else None,
        "programme": [compacter(c, 4) for c in courses[:10]],
    }


def verifier() -> str:
    """
    Test de bout en bout : connexion + publication acquittée.

    Se contenter de vérifier la connexion ne suffit pas. Un broker peut
    accepter la connexion puis refuser silencieusement les écritures sur
    `homeassistant/#` si une liste de contrôle d'accès est en place — la
    découverte part alors dans le vide, sans la moindre erreur.

    Seul un acquittement en QoS 1 prouve que le message est bien arrivé.
    """
    c = _client()
    try:
        info = _publier(c, f"{BASE}/verification", "ping", retain=False)
        info.wait_for_publish(timeout=5)
        if not info.is_published():
            raise MqttIndisponible(
                f"connecté à {HOTE} mais la publication n'est pas acquittée — "
                "l'utilisateur MQTT n'a probablement pas le droit d'écrire"
            )
        return f"{HOTE}, écriture confirmée"
    finally:
        c.loop_stop()
        c.disconnect()


def publier_amorcage(depuis: date, jusqua: date) -> None:
    """
    Signale à Home Assistant que le rattrapage initial tourne.

    Sans ça, la vue Lovelace resterait vide pendant une à deux heures au
    premier démarrage, sans que rien n'explique pourquoi. Un écran vide
    ressemble à une panne.
    """
    c = _client()
    try:
        infos = publier_decouverte(c)
        infos.append(_publier(c, T_ETAT, json.dumps({
            "date": date.today().isoformat(),
            "modele": MODELE,
            "courses": 0,
            "age_heures": 0,
            "frais": True,
            "amorcage": True,
            "message": f"Rattrapage de l'historique en cours ({depuis} → {jusqua}). "
                       "Compter 1 à 2 heures, une seule fois.",
            "prochaine": None,
            "programme": [],
        }, ensure_ascii=False)))
        infos.append(_publier(c, T_DISPO, "online"))
        _attendre(infos, "découverte + amorçage")
    finally:
        c.loop_stop()
        c.disconnect()


def publier(jour: date | None = None) -> dict:
    c = _client()
    try:
        infos = publier_decouverte(c)
        with db.connect() as conn:
            charge = construire_charge(conn, jour)

        brut = json.dumps(charge, ensure_ascii=False)
        if len(brut.encode()) > 15000:
            # Dégradation contrôlée plutôt que troncature silencieuse côté HA.
            charge["programme"] = [
                {k: v for k, v in c_.items() if k != "selection"}
                for c_ in charge["programme"]
            ]
            brut = json.dumps(charge, ensure_ascii=False)
            log.warning("charge utile réduite à %d octets (programme sans sélections)",
                        len(brut.encode()))

        infos.append(_publier(c, T_ETAT, brut))
        infos.append(_publier(c, T_DISPO, "online"))
        _attendre(infos, f"publication de {charge['courses']} courses "
                         f"({len(brut.encode())} octets)")
        return charge
    finally:
        c.loop_stop()
        c.disconnect()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s")
    publier()
