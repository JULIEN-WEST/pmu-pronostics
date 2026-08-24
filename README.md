# pmu-pronostics

Probabilités calibrées par partant, toutes disciplines, à partir de l'API
PMU — avec le classement qui en découle et un degré de confiance qui veut
dire quelque chose.

Successeur de [`api_equide`](https://github.com/Projets-finaux-Simplon-2024/api_equide)
et [`build_bdd_equide`](https://github.com/Projets-finaux-Simplon-2024/build_bdd_equide).
Ce qui change : la généalogie n'est plus la finalité, elle devient un bloc
de features parmi d'autres, et tout le projet s'articule autour d'une seule
contrainte — **ne jamais laisser le futur entrer dans le passé**.

---

## Essayer tout de suite, sans rien collecter

```bash
pip install -r requirements.txt
python scripts/demo.py
```

Génère un univers de courses dont on connaît la vérité, construit les
73 features, entraîne les deux variantes du modèle et imprime le rapport
complet. Trois minutes, aucune dépendance externe.

C'est aussi le meilleur moyen de comprendre ce que le projet produit avant
d'investir dans la collecte.

---

## Mise en production

La pile tourne en Docker sur une machine qui reste allumée — un conteneur
LXC Proxmox convient très bien.

**Pas sur la box Home Assistant.** HAOS est un appareil verrouillé qui
n'exécute pas de conteneurs arbitraires ; il faudrait en faire un module
complémentaire, et y loger PostgreSQL plus un ré-entraînement hebdomadaire
alourdirait la domotique pour rien. Home Assistant reste ce qu'il fait de
mieux ici : l'affichage et les notifications.

```bash
# 1. Voir ce que l'API PMU renvoie VRAIMENT aujourd'hui
python scripts/probe.py --date 2026-08-22

# 2. Configurer et démarrer
cp .env.example .env          # POSTGRES_PASSWORD et MQTT_HOST au minimum
docker compose up -d --build

# 3. Amorcer l'historique — un mois suffit (voir plus bas pourquoi)
docker compose exec collecteur \
  python -m pmu.collect backfill --depuis 2026-07-23 --jusqua 2026-08-22

# 4. Premier entraînement
docker compose exec collecteur python -m pmu.predict entrainer --avec-marche

# 5. Vérifier
curl -s http://localhost:8100/sante | jq
```

L'ordonnanceur prend ensuite le relais : programme le matin, relevé des
cotes toutes les 5 minutes dans les 45 minutes avant chaque départ,
pronostic à chaque mouvement, arrivées le soir, ré-entraînement le
dimanche. Le service redémarre tout seul et ne meurt jamais sur une
erreur de tour.

### Les trois services

| Service | Rôle |
|---|---|
| `db` | PostgreSQL 16. Port non publié — la base n'a pas à être joignable. |
| `collecteur` | Ordonnanceur : collecte, cotes, pronostics, publication MQTT. |
| `api` | FastAPI en lecture seule sur `:8100`. Ne calcule rien, relit la table. |

### Home Assistant

Deux fichiers dans `homeassistant/` :

- **`package_pmu.yaml`** → `/config/packages/pmu.yaml`. Commandes REST,
  script de rafraîchissement, alerte si la pile décroche, résumé Telegram
  du midi, alerte sur les courses à forte confiance.
- **`lovelace_vue_pronostics.yaml`** → à coller dans un tableau de bord.
  Cartes natives uniquement, aucune carte personnalisée requise.

Les entités `sensor.pmu_*` n'ont **rien** à déclarer : elles arrivent par
découverte MQTT depuis le conteneur, groupées sous un appareil
« Pronostics PMU ».

MQTT plutôt qu'un capteur REST parce que les messages sont **retenus** :
après un redémarrage de HA, ou pile éteinte, les dernières valeurs sont
toujours là. Un capteur REST passe à `unavailable` dès que l'API ne répond
plus. Et le testament MQTT bascule les entités en indisponible si le
conteneur meurt — pas de vieux pronostic affiché comme s'il était frais.

```
┌─────────────── hôte Docker (LXC Proxmox) ───────────────┐
│  db ← collecteur → api :8100                            │
└───────────────────────┬─────────────────────────────────┘
                        │ MQTT (retenu) + REST
              ┌─────────▼──────────┐
              │  Home Assistant    │  vue Lovelace + Telegram
              └────────────────────┘
```

---

## Trois décisions qui structurent tout

### 1. L'historique s'amorce par les performances détaillées, pas par un backfill

L'endpoint `/performances-detaillees` renvoie, pour chaque partant du jour,
le détail de **ses courses passées** — souvent quinze à vingt lignes
remontant plusieurs saisons.

Collecter un mois de programmes récupère donc mécaniquement l'historique de
plusieurs milliers de chevaux **actifs** : précisément ceux sur lesquels il
faudra prédire. Un backfill chronologique sur trois ans demanderait cinquante
à cent fois plus de requêtes pour une couverture équivalente là où elle
compte, et ferait couper l'accès en chemin.

Ces lignes importées sont plus pauvres (ni cote, ni gains détaillés). Elles
sont marquées comme telles et la vue `v_historique_cheval` préfère toujours
la ligne collectée en direct quand elle existe.

### 2. Aucun agrégat n'est stocké en base

La base ne contient que des faits horodatés. Les taux de réussite, la forme
récente, l'aptitude au terrain sont recalculés à la demande, avec une borne
temporelle explicite.

Stocker « taux de victoire du cheval » dans une colonne, c'est se garantir
une fuite le jour où on le rafraîchit sans y penser.

### 3. Deux modèles, toujours

| Variante | Voit la cote | Ce qu'elle sert à |
|---|---|---|
| `sans_marche` | non | La seule qui puisse révéler un écart exploitable |
| `avec_marche` | oui | Le plafond : ce qui est prévisible au mieux |

Un modèle qui voit la cote apprend surtout à la recopier. Il affiche de
belles métriques et n'apporte rien, puisque la cote est déjà connue de tous.
L'écart entre les deux mesure ce que le public sait déjà.

---

## La règle anti-fuite

> Aucune feature d'une ligne ne peut dépendre du résultat de cette ligne,
> **ni d'aucune course partie en même temps ou après elle**.

La seconde moitié est celle qu'on oublie. Deux demi-frères par le même père
courent l'un contre l'autre : avec un `shift(1)` naïf, le second voit le
résultat du premier, alors qu'ils sont partis ensemble. Discret, et
suffisant pour fausser toutes les features de lignée, d'entraîneur et
d'écurie.

D'où le calcul unique de `features._taux_glissant` :

```
  cumul sur la clé jusqu'ici
− cumul sur (clé, course) jusqu'ici
= cumul sur les seules courses strictement antérieures
```

Deux garde-fous automatiques :

- `tests/test_fuite.py` — cible **purement aléatoire**, le modèle doit
  plafonner à 0,50 d'AUC. Vérifié : injecter une moyenne calculée sur tout
  l'historique fait bondir l'AUC à 0,90, le test se déclenche.
- `features.colonnes_features()` refuse de laisser passer une colonne de
  résultat.

---

## Les features

73 colonnes, sept familles.

| Préfixe | Famille | Exemples |
|---|---|---|
| `c_` | Contexte de course | distance, terrain, nb partants, corde relative |
| `p_` | Partant | âge, poids, gains par course, jours de repos |
| `mus_` | Musique déclarée | moyenne, taux de top 3, incidents |
| `r_` | **Rang dans le lot** | rang de gains, de musique, de taux de victoire |
| `h_` | Historique glissant | cheval, driver, entraîneur, couple, attelage |
| `a_` | **Aptitudes** | terrain, distance, hippodrome, discipline — et leur `_delta` |
| `g_` | **Lignée** | père, père de mère, croisement, père × terrain |

Trois remarques sur les choix qui comptent :

**Les `r_` (rang intra-course).** Ce qui prédit n'est pas le niveau absolu
d'un cheval mais son niveau **relatif aux autres partants de sa course**.
Un cheval à 12 000 € de gains est excellent dans un lot à 8 000, médiocre
dans un lot à 40 000.

**Les `_delta`.** `a_terrain` seul ne fait que redire le niveau général du
cheval. `a_terrain_delta` — l'écart entre sa réussite sur ce terrain et sa
réussite globale — répond à la vraie question : *ce cheval est-il meilleur
que lui-même dans le lourd ?* C'est l'axe « talent sur type de sol ».

**Les `g_`.** L'axe généalogie du projet d'origine, mesuré sur les courses
antérieures des autres produits du même étalon. `g_pere_terrain_delta`
teste une transmission d'aptitude **spécifique**, au-delà de la qualité
moyenne de l'étalon.

---

## L'évaluation

L'AUC ne suffit pas : elle mesure une capacité à ordonner, pas la justesse
des probabilités. Or c'est la justesse qui a été demandée.

1. **Justesse** — Brier, log-loss, Brier skill score
2. **Calibration** — quand le modèle annonce 20 %, est-ce que ça arrive
   20 % du temps ? (courbe par décile, ECE)
3. **Face au marché** — bat-on le favori du public ? *C'est la référence.*
4. **Rentabilité** — l'écart survit-il au prélèvement ?

Le rapport est aussi **stratifié par discipline**. C'est le contrôle qui
dira si le modèle unique tient ou s'il faut le scinder — mesuré, pas deviné.

### Ce que la démo montre

Sur l'univers synthétique, calibré pour que le favori gagne 31 % du temps
comme en courses réelles :

```
  ECE                   0.00728        calibration excellente
  top-1 modèle           25.70%
  top-1 favori public    31.30%
  bat le marché ?           NON
  ROI                    -30.6%
```

**Le modèle est bien calibré et perd de l'argent.** Les deux à la fois, et
ce n'est pas contradictoire : produire des probabilités justes est un
problème résolu, battre le consensus d'un marché mutuel en est un autre.

C'est le résultat honnête. Un dépôt qui afficherait +40 % de retour aurait
un bug — la première version de ce simulateur construisait la cote à partir
du score **après** tirage de l'aléa, donc d'un marché qui avait déjà vu
l'arrivée : le favori gagnait 82 % du temps et miser sur lui rapportait
+72 %. `tests/test_simulateur.py` empêche la rechute.

---

## Ce contre quoi il faut se prémunir

Le PMU est un pari **mutuel**, pas un bookmaker :

- la cote **est** la répartition des mises du public — battre le marché,
  c'est battre tous les autres parieurs, dont beaucoup sont outillés ;
- le prélèvement est retiré **avant** répartition. Égaler le marché fait
  donc perdre ce pourcentage à chaque tour ;
- vos propres mises font baisser la cote que vous venez de viser.

Le seuil de réussite n'est donc pas « mieux que le hasard », ni même
« mieux que le favori ». C'est *mieux que le marché, d'une marge supérieure
au prélèvement, et de façon stable sur plusieurs centaines de courses*.
Ce harnais est fait pour mesurer ça honnêtement — y compris, et surtout,
quand la réponse est non.

La constante `evaluate.PRELEVEMENT_DEFAUT` vaut 0,15 : **à vérifier** sur
vos propres rapports, elle varie selon le type de pari.

---

## L'axe élevage

Le modèle de mariages du mémoire d'origine n'est pas abandonné, il est
repoussé — et il devient beaucoup plus facile une fois ce socle en place.

Le modèle de pronostic estime, pour chaque cheval, une valeur intrinsèque
nettoyée du contexte de course. Agréger cette estimation par croisement
père × père de mère donne l'évaluation de mariage, sans repartir de zéro :
une requête sur les sorties du modèle, pas un second projet.

Les features `g_*` sont déjà l'ossature de cette agrégation.

---

## Structure

```
sql/001_schema.sql        schéma PostgreSQL commenté
src/pmu/
  client.py               HTTP : throttling, reprise, détection de version
  normalize.py            JSON → lignes, défensif de bout en bout
  db.py                   upserts idempotents, résolution de généalogie
  collect.py              backfill / jour / live      (CLI)
  features.py             ⭐ le feature store anti-fuite
  train.py                découpage chronologique, calibration isotonique
  evaluate.py             justesse, calibration, marché, rentabilité
scripts/
  probe.py                ⭐ à lancer en premier : que renvoie l'API ?
  simulateur.py           univers synthétique étalonné sur le réel
  demo.py                 chaîne complète, sans base de données
tests/                    49 tests, dont le canari anti-fuite
```

---

## Note sur l'API

`online.turfinfo.api.pmu.fr` n'est **pas** une API publique documentée. Elle
peut changer ou fermer sans préavis, et rien n'en garantit l'usage. D'où :

- 2 requêtes/seconde par défaut, cache disque, `User-Agent` honnête ;
- parsing intégralement défensif — une clé qui disparaît dégrade la
  collecte, elle ne la casse pas ;
- `PmuClient.detect_client_version()` balaie les valeurs voisines du
  segment `/client/62/` quand celle-ci cesse de répondre.

Lancer `scripts/probe.py` **avant** toute collecte : c'est ce qui dira si le
contrat de parsing tient encore.
