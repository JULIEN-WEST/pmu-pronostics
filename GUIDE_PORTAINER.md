# Déploiement sur Portainer — marche à suivre

Cinq étapes, uniquement des clics dans un navigateur. Environ 20 minutes
de manipulation, puis une à deux heures pendant lesquelles la machine
travaille seule.

**Rien à taper dans un terminal.** Le conteneur rattrape l'historique et
entraîne son premier modèle tout seul au premier démarrage.

---

## Ce qu'il te faut

- un compte GitHub (gratuit) — c'est là que Portainer ira chercher le code ;
- ton Portainer habituel ;
- le fichier `.zip` décompressé quelque part.

---

## Étape 1 — Déposer le code sur GitHub

Un dépôt GitHub, ici, c'est simplement un dossier de fichiers hébergé en
ligne. Portainer sait aller y chercher le code — c'est la seule raison
pour laquelle on passe par là.

1. <https://github.com/new>
2. **Repository name** : `pmu-pronostics`
3. Coche **Public**
4. **Create repository**
5. Sur la page suivante, clique **uploading an existing file**
6. Ouvre le dossier `pmu-pronostics` décompressé, **sélectionne tout ce
   qu'il y a dedans** (Ctrl+A / Cmd+A) et glisse la sélection dans la page
7. En bas : **Commit changes**

> **Le piège** : glisse le *contenu* du dossier, pas le dossier lui-même.
> Après l'envoi, tu dois voir `docker-compose.yml` et `Dockerfile`
> directement dans la liste, pas rangés dans un sous-dossier. Si c'est le
> cas, supprime le dépôt et recommence — plus rapide que de corriger.

> **Pourquoi public** : un dépôt privé oblige à créer un jeton d'accès et
> à le renseigner dans Portainer. Deux étapes de plus, deux occasions de
> se tromper. Ce code ne contient aucun mot de passe.

---

## Étape 2 — Créer la stack

**Stacks** → **+ Add stack**, méthode **Repository**.

| Champ | Valeur |
|---|---|
| Name | `pmu` |
| Repository URL | `https://github.com/<ton-compte>/pmu-pronostics` |
| Repository reference | `refs/heads/main` |
| Compose path | `docker-compose.yml` |
| Authentication | désactivé (dépôt public) |
| GitOps updates | désactivé |

Ne clique pas encore sur Deploy.

---

## Étape 3 — Deux variables, puis déployer

Descends jusqu'à **Environment variables** → **+ Add an environment
variable**. C'est ici que vivent les mots de passe, jamais dans GitHub.

| Name | Value | |
|---|---|---|
| `POSTGRES_PASSWORD` | un mot de passe long | **obligatoire** |
| `MQTT_HOST` | `192.168.1.153` | pour Home Assistant |
| `MQTT_USER` | ton identifiant MQTT | si ton broker en exige un |
| `MQTT_PASSWORD` | son mot de passe | idem |
| `TZ` | `Europe/Paris` | conseillé |

Puis **Deploy the stack**. La première construction prend 5 à 10 minutes
avec beaucoup de texte qui défile : c'est normal.

Si `POSTGRES_PASSWORD` manque, le déploiement refuse de démarrer et le dit
en clair, plutôt que de lancer une base sans mot de passe.

---

## Étape 4 — Lire les journaux, puis laisser travailler

**Containers** — trois lignes vertes : `pmu-db`, `pmu-collecteur`,
`pmu-api`. Clique l'icône **Logs** sur `pmu-collecteur`.

Le programme se présente lui-même :

```
┌─────────────────────────────────────────────────────────┐
│  AUTO-DIAGNOSTIC AU DEMARRAGE                           │
├─────────────────────────────────────────────────────────┤
│  BASE DE DONNEES  ..... OK                              │
│  API PMU          ..... OK  (8 reunions le 2026-08-23)  │
│  BROKER MQTT      ..... OK  (192.168.1.153)             │
└─────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│  PREMIER DEMARRAGE - AMORCAGE AUTOMATIQUE                        │
├──────────────────────────────────────────────────────────────────┤
│  Rattrapage de 30 jours : 2026-07-25 -> 2026-08-24               │
│                                                                  │
│  Compter 1 a 2 heures. C'est normal et ca ne se reproduira pas.  │
│  Tu peux fermer cette fenetre, le conteneur continue.            │
└──────────────────────────────────────────────────────────────────┘
```

**Il n'y a rien à faire.** Ferme l'onglet. Reviens dans deux heures : tu
dois trouver `AMORCAGE TERMINE`.

Pendant ce temps, tu peux vérifier que l'API répond en ouvrant dans ton
navigateur `http://<ip-de-ta-machine-docker>:8100/sante` — une page de
texte avec `"ok": true` signifie que tout va bien.

---

## Étape 5 — Brancher Home Assistant

Les capteurs apparaissent **tout seuls** : le conteneur les annonce au
broker MQTT que tu utilises déjà pour Zigbee. Vérifie dans *Paramètres →
Appareils et services → MQTT* : un appareil **Pronostics PMU** avec sept
capteurs.

Reste l'affichage. Deux fichiers, dans le dossier `homeassistant/` :

**`lovelace_vue_pronostics.yaml`** — ouvre-le dans un éditeur de texte,
copie tout. Dans HA : un tableau de bord → ⋮ *Modifier* → ⋮ *Éditeur de
configuration brut*, colle le bloc à la suite des autres vues sous
`views:`.

**`package_pmu.yaml`** — à déposer dans `/config/packages/pmu.yaml` via le
module **File editor** ou **Studio Code Server**. Remplace-y
`192.168.1.42` par l'IP de ta machine Docker. Puis *Outils de
développement → YAML → Recharger toute la configuration*.

> Si le second te bloque, saute-le : la vue fonctionne sans. Il n'ajoute
> que les extras (alerte si la pile s'arrête, résumé Telegram du midi,
> notification avant les courses à forte confiance).

---

## Si ça coince

| Ce que tu vois | Ce que ça veut dire | Quoi faire |
|---|---|---|
| `required variable POSTGRES_PASSWORD is missing` | La variable n'a pas été enregistrée. | Stack → *Editor* → remplir → *Update the stack*. |
| `failed to read dockerfile` / `no such file` | Les fichiers sont dans un sous-dossier sur GitHub. | Vérifie que `docker-compose.yml` est à la racine du dépôt. Sinon refais l'étape 1. |
| `BROKER MQTT ..... ECHEC` | Mauvaise adresse, identifiants manquants, ou droits d'écriture refusés. | Corrige les variables `MQTT_*`. Le message précise s'il s'agit d'un refus de connexion ou d'un refus d'écriture. La pile tourne quand même, sans entités HA. |
| Les capteurs `sensor.pmu_*` n'existent pas dans HA | La découverte n'est pas arrivée, ou HA ne l'a pas traitée. | Dans HA : *Paramètres → Appareils et services → MQTT → Configurer → Écouter un sujet*, saisir `homeassistant/sensor/pmu_pronostics/#`. Les messages étant retenus, ils s'affichent immédiatement s'ils sont bien sur le broker — le problème est alors côté HA, pas côté pile. |
| `API PMU ..... ECHEC` | L'API du PMU a changé de forme. | Le conteneur se met en veille au lieu de boucler. Copie-moi le message. |
| `PAS ASSEZ DE DONNEES POUR ENTRAINER` | Moins de 15 000 partants récupérés. | Rien à faire : il continue à collecter et réessaiera seul. Pour accélérer, mets `PMU_BACKFILL_JOURS` à `60`. |
| `pull access denied for pmu-pronostics` | La case *Re-pull image* était cochée. | Décoche-la et reclique *Pull and redeploy*. L'image se construit localement, elle n'est dans aucun registre. |
| `schéma obsolète … recréation automatique` | La base venait d'une version antérieure et était vide. | Rien à faire, c'est le message normal : elle a été recréée au bon format. |

---

## Ce qui se passe ensuite, sans toi

Le programme le matin, les cotes toutes les 5 minutes dans les 45 minutes
avant chaque départ, un pronostic recalculé à chaque mouvement, les
arrivées le soir. Le dimanche, le modèle se ré-entraîne.

Si la machine redémarre, la pile repart seule et reprend où elle en
était : le rattrapage initial ne se refait pas, il est marqué comme fait
en base.

**Mettre à jour** : renvoie les fichiers sur GitHub, puis Stacks → `pmu` →
*Pull and redeploy*.

> ⚠️ **Ne coche JAMAIS « Re-pull image ».** L'image est construite sur ta
> machine à partir du Dockerfile ; elle n'existe dans aucun registre
> public. Cocher cette case fait chercher `pmu-pronostics` sur Docker Hub,
> où elle n'est évidemment pas, et le déploiement échoue sur
> `pull access denied`. Le bouton seul suffit : il récupère le dépôt et
> reconstruit.

Les volumes sont conservés : base et modèles survivent.

---

## Pour aller plus loin (facultatif)

Ces commandes ne sont **pas** nécessaires — tout se fait automatiquement.
Elles servent si tu veux reprendre la main un jour.

Console d'un conteneur : *Containers* → `pmu-collecteur` → *Console* →
*Connect* (commande `/bin/bash`, utilisateur `pmu`).

```bash
# Que renvoie l'API PMU aujourd'hui ?
python scripts/probe.py --date 2026-08-22

# Combien de données en base ?
python -c "
from pmu import db, dataset
with db.connect() as c: print(dataset.stats(c))"

# Forcer un ré-entraînement
python -m pmu.predict entrainer --avec-marche

# Recalculer les pronostics du jour
python -m pmu.predict jour
```

Collecte longue à lancer depuis l'hôte Docker (survit à la fermeture de
l'onglet) :

```bash
docker exec -d pmu-collecteur \
  python -m pmu.collect backfill --depuis 2026-06-01 --jusqua 2026-08-23
```

Sauvegarde de la base — le seul contenu irremplaçable est l'historique des
cotes, qui ne se reconstitue jamais après coup :

```bash
docker exec pmu-db pg_dump -U pmu pmu | gzip > pmu-$(date +%F).sql.gz
```
