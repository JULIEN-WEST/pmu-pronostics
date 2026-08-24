# syntax=docker/dockerfile:1
#
# Image unique pour les deux services : le collecteur (ordonnanceur) et
# l'API. Ils ne diffèrent que par la commande, définie dans le compose.

FROM python:3.12-slim

# libgomp : LightGBM en dépend, et sans lui l'import échoue au démarrage
#           avec un message peu parlant.
# curl    : utilisé par le healthcheck de l'API.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 curl \
 && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src \
    PMU_CACHE=/data/cache \
    PMU_MODELES=/data/modeles \
    PMU_SCHEMA=/app/sql/001_schema.sql

WORKDIR /app

# Couche de dépendances isolée : elle ne se reconstruit que si les
# fichiers requirements changent, pas à chaque modification de code.
COPY requirements.txt requirements-service.txt ./
RUN pip install --no-cache-dir -r requirements-service.txt

COPY sql/ ./sql/
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY tests/ ./tests/

# Utilisateur non privilégié : rien ici n'a besoin de root.
#
# Le chown de /data AVANT la déclaration du volume n'est pas cosmétique :
# Docker recopie la propriété du répertoire de l'image dans un volume
# nommé vide au premier montage. Sans lui, /data appartiendrait à root
# dans le volume et le conteneur ne pourrait ni écrire son cache ni
# sauvegarder ses modèles — en échouant tard, au premier entraînement.
RUN useradd --create-home --uid 1000 pmu \
 && mkdir -p /data/cache /data/modeles \
 && chown -R pmu:pmu /data /app
USER pmu

# Pas de directive VOLUME : elle créerait un volume anonyme à chaque
# `docker run` ad hoc. Le montage est déclaré explicitement dans le compose.

EXPOSE 8100

# Par défaut l'ordonnanceur ; l'API surcharge la commande dans le compose.
CMD ["python", "-m", "pmu.planificateur"]
