"""
Tests MQTT contre un VRAI broker.

Ces tests existent à cause d'un bug qu'aucune vérification en mémoire
n'aurait attrapé : la découverte était « publiée » — le journal le disait —
et aucune entité n'apparaissait dans Home Assistant.

La cause : `publish()` en QoS 0 met le message dans une file locale, puis
`disconnect()` immédiat jetait cette file avant qu'elle ne parte. Le code
ne mentait pas, il n'avait simplement aucun moyen de savoir.

D'où la règle appliquée ici : on ne considère un message publié que
lorsque le broker l'a ACQUITTÉ. Et on le vérifie en s'abonnant vraiment.

Ignorés si aucun broker n'est joignable — passer PMU_TEST_MQTT pour
activer, ex. PMU_TEST_MQTT=127.0.0.1:1884
"""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import date

import pytest

BROKER = os.environ.get("PMU_TEST_MQTT")
pytestmark = pytest.mark.skipif(not BROKER, reason="PMU_TEST_MQTT non défini")

import paho.mqtt.client as mqtt  # noqa: E402

from pmu import mqtt_ha  # noqa: E402

if BROKER:
    # On surcharge les constantes du module, pas seulement l'environnement :
    # `mqtt_ha` peut déjà avoir été importé par un autre test, auquel cas
    # ses constantes sont figées et la variable d'environnement arriverait
    # trop tard.
    hote, _, port = BROKER.partition(":")
    mqtt_ha.HOTE = hote
    mqtt_ha.PORT = int(port or 1883)
    mqtt_ha.UTILISATEUR = None
    mqtt_ha.MOTDEPASSE = None


@pytest.fixture
def espion():
    """
    Abonné qui écoute tout et enregistre ce qui arrive RÉELLEMENT.

    C'est le seul juge acceptable : la seule preuve qu'un message est
    parti, c'est qu'un autre client le reçoive.
    """
    recus: dict[str, str] = {}
    pret = threading.Event()

    def on_connect(client, userdata, flags, rc, properties=None):
        client.subscribe("#", qos=1)
        pret.set()

    def on_message(client, userdata, msg):
        recus[msg.topic] = msg.payload.decode("utf-8", "replace")

    c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="espion-test")
    c.on_connect = on_connect
    c.on_message = on_message
    c.connect(mqtt_ha.HOTE, mqtt_ha.PORT, keepalive=30)
    c.loop_start()
    assert pret.wait(5), "l'espion n'a pas pu se connecter au broker"
    time.sleep(0.3)
    yield recus
    c.loop_stop()
    c.disconnect()


def _purger():
    """Efface les messages retenus d'un test précédent."""
    c = mqtt_ha._client()
    for t in [f"{mqtt_ha.PREFIXE}/sensor/pmu_pronostics/{k}/config"
              for k, *_ in mqtt_ha.CAPTEURS]:
        c.publish(t, "", qos=1, retain=True)
    c.publish(f"{mqtt_ha.PREFIXE}/binary_sensor/pmu_pronostics/frais/config",
              "", qos=1, retain=True)
    c.publish(mqtt_ha.T_ETAT, "", qos=1, retain=True)
    time.sleep(0.4)
    c.loop_stop()
    c.disconnect()


# ---------------------------------------------------------------------

def test_connexion_attend_l_acquittement():
    """`_client()` ne doit rendre la main qu'une fois vraiment connecté."""
    c = mqtt_ha._client()
    try:
        assert c.is_connected()
    finally:
        c.loop_stop()
        c.disconnect()


def test_verifier_confirme_l_ecriture():
    detail = mqtt_ha.verifier()
    assert mqtt_ha.HOTE in detail
    assert "écriture confirmée" in detail


def test_broker_injoignable_leve_une_erreur_claire(monkeypatch):
    """Un port fermé doit produire un message exploitable, pas une trace."""
    monkeypatch.setattr(mqtt_ha, "PORT", 1)
    with pytest.raises(Exception) as exc:
        mqtt_ha.verifier()
    assert "1" in str(exc.value) or "refus" in str(exc.value).lower()


def test_la_decouverte_arrive_vraiment_au_broker(espion):
    """
    LE test qui manquait. On publie la découverte, puis on vérifie qu'un
    abonné indépendant a bien reçu les huit configurations.
    """
    _purger()
    mqtt_ha.publier_amorcage(date(2026, 7, 26), date(2026, 8, 25))
    time.sleep(0.8)

    attendus = [f"{mqtt_ha.PREFIXE}/sensor/pmu_pronostics/{cle}/config"
                for cle, *_ in mqtt_ha.CAPTEURS]
    attendus.append(f"{mqtt_ha.PREFIXE}/binary_sensor/pmu_pronostics/frais/config")

    manquants = [t for t in attendus if not espion.get(t)]
    assert not manquants, (
        f"{len(manquants)} configuration(s) jamais arrivée(s) au broker : "
        f"{[t.rsplit('/', 2)[1] for t in manquants]}"
    )


def test_les_configurations_sont_exploitables_par_home_assistant(espion):
    """
    Chaque configuration doit porter ce que HA exige pour créer l'entité :
    un identifiant unique, un sujet d'état, et un appareil de rattachement.
    """
    _purger()
    mqtt_ha.publier_amorcage(date(2026, 7, 26), date(2026, 8, 25))
    time.sleep(0.8)

    sujet = f"{mqtt_ha.PREFIXE}/sensor/pmu_pronostics/courses/config"
    conf = json.loads(espion[sujet])
    assert conf["unique_id"] == "pmu_pronostics_courses"
    assert conf["state_topic"] == mqtt_ha.T_ETAT
    assert conf["device"]["identifiers"] == ["pmu_pronostics"]
    # Le capteur « courses » porte toute la journée en attributs : c'est
    # lui que lit la vue Lovelace.
    assert conf["json_attributes_topic"] == mqtt_ha.T_ETAT


def test_l_etat_d_amorcage_arrive_et_est_lisible(espion):
    _purger()
    mqtt_ha.publier_amorcage(date(2026, 7, 26), date(2026, 8, 25))
    time.sleep(0.8)

    etat = json.loads(espion[mqtt_ha.T_ETAT])
    assert etat["amorcage"] is True
    assert etat["courses"] == 0
    assert "Rattrapage" in etat["message"]
    assert espion[mqtt_ha.T_DISPO] == "online"


def test_les_messages_sont_retenus(espion):
    """
    Sans le drapeau « retenu », un redémarrage de Home Assistant perdrait
    toutes les entités jusqu'à la publication suivante.
    """
    _purger()
    mqtt_ha.publier_amorcage(date(2026, 7, 26), date(2026, 8, 25))
    time.sleep(0.8)

    recus: dict[str, bool] = {}
    pret = threading.Event()

    def on_connect(client, userdata, flags, rc, properties=None):
        client.subscribe(f"{mqtt_ha.PREFIXE}/#", qos=1)
        client.subscribe(f"{mqtt_ha.BASE}/#", qos=1)
        pret.set()

    def on_message(client, userdata, msg):
        recus[msg.topic] = msg.retain == 1

    tardif = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="tardif")
    tardif.on_connect = on_connect
    tardif.on_message = on_message
    tardif.connect(mqtt_ha.HOTE, mqtt_ha.PORT)
    tardif.loop_start()
    assert pret.wait(5)
    time.sleep(0.8)
    tardif.loop_stop()
    tardif.disconnect()

    # Un client qui arrive APRÈS coup doit tout recevoir quand même.
    assert recus, "aucun message retenu : HA ne retrouverait rien au redémarrage"
    assert recus.get(mqtt_ha.T_ETAT) is True
