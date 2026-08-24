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


def _client() -> mqtt.Client:
    c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="pmu-pronostics")
    if UTILISATEUR:
        c.username_pw_set(UTILISATEUR, MOTDEPASSE)
    # Testament : si le conteneur meurt, HA passe les entités en indisponible
    # au lieu d'afficher éternellement le dernier pronostic comme s'il était frais.
    c.will_set(T_DISPO, "offline", retain=True)
    c.connect(HOTE, PORT, keepalive=60)
    return c


def publier_decouverte(c: mqtt.Client) -> None:
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
        c.publish(f"{PREFIXE}/sensor/pmu_pronostics/{cle}/config",
                  json.dumps(conf, ensure_ascii=False), retain=True)

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
    c.publish(f"{PREFIXE}/binary_sensor/pmu_pronostics/frais/config",
              json.dumps(binaire, ensure_ascii=False), retain=True)
    log.info("découverte MQTT publiée (%d capteurs)", len(CAPTEURS) + 1)


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

    def compacter(c: dict, n_partants: int) -> dict:
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
            "selection": [
                {"num": s["num"], "cheval": s["cheval"],
                 "proba": s["proba"], "cote": s["cote"],
                 "valeur": s["valeur"], "arrivee": s["arrivee"]}
                for s in c["selection"][:n_partants]
            ],
        }

    return {
        "date": jour.isoformat(),
        "modele": MODELE,
        "courses": len(courses),
        "age_heures": age_h if age_h is not None else 999,
        "frais": age_h is not None and age_h < 24,
        "maj": maintenant.isoformat(),
        # La prochaine course est détaillée ; les autres sont résumées.
        "prochaine": compacter(proch, 8) if proch else None,
        "programme": [compacter(c, 4) for c in courses[:10]],
    }


def publier_amorcage(depuis: date, jusqua: date) -> None:
    """
    Signale à Home Assistant que le rattrapage initial tourne.

    Sans ça, la vue Lovelace resterait vide pendant une à deux heures au
    premier démarrage, sans que rien n'explique pourquoi. Un écran vide
    ressemble à une panne.
    """
    c = _client()
    c.loop_start()
    try:
        publier_decouverte(c)
        c.publish(T_ETAT, json.dumps({
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
        }, ensure_ascii=False), retain=True)
        c.publish(T_DISPO, "online", retain=True)
        log.info("amorçage signalé à Home Assistant")
    finally:
        c.loop_stop()
        c.disconnect()


def publier(jour: date | None = None) -> dict:
    c = _client()
    c.loop_start()
    try:
        publier_decouverte(c)
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

        c.publish(T_ETAT, brut, retain=True)
        c.publish(T_DISPO, "online", retain=True)
        log.info("publié : %d courses, %d octets", charge["courses"], len(brut.encode()))
        return charge
    finally:
        c.loop_stop()
        c.disconnect()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s")
    publier()
