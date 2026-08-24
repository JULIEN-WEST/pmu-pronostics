"""
Générateur de courses synthétiques AVEC signal réel.

À quoi ça sert : valider tout le pipeline sans attendre des semaines de
collecte. On fabrique un univers dont on connaît la vérité — chaque cheval
a une valeur intrinsèque, héritée en partie de son père, plus une aptitude
au terrain — puis on vérifie que le modèle la retrouve.

Si le modèle n'apprend rien ici, le problème est dans le code, pas dans les
données. Si le modèle apprend ici mais échoue sur les vraies courses, c'est
que le monde réel est plus dur — ce qui est l'hypothèse par défaut.

Le générateur produit aussi une cote de marché : une version bruitée de la
vraie probabilité, majorée du prélèvement. Elle reproduit la situation
réelle — un public qui a globalement raison, avec des erreurs exploitables.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DISCIPLINES = ["ATTELE", "MONTE", "PLAT"]
TERRAINS = ["BON", "SOUPLE", "COLLANT", "LOURD"]
HIPPODROMES = ["VIN", "ENG", "CAG", "DEA", "CHA", "LYO"]


def generer(
    n_courses: int = 3000,
    n_chevaux: int = 2500,
    n_peres: int = 60,
    heritabilite: float = 0.45,
    echelle: float = 1.00,
    bruit_marche: float = 0.25,
    prelevement: float = 0.15,
    graine: int = 42,
) -> pd.DataFrame:
    """
    `echelle` règle l'écart de niveau entre partants, donc le caractère
    prévisible des courses. Étalonné pour que le meilleur cheval gagne
    environ une fois sur trois — le taux observé en courses réelles.
    Le monter rend le simulateur complaisant ; le baisser rend les courses
    illisibles.

    `bruit_marche` règle l'imperfection du public. À 0, le marché est
    parfait et rien n'est exploitable ; plus il monte, plus il laisse des
    erreurs à saisir. 0,30 laisse une marge volontairement optimiste :
    considérez-la comme une borne haute, pas comme une prévision.
    """
    rng = np.random.default_rng(graine)

    # --- L'univers ---
    peres = [f"ETALON_{i:02d}" for i in range(n_peres)]
    peres_mere = [f"PM_{i:02d}" for i in range(n_peres // 2)]

    # Valeur transmise par l'étalon, et son aptitude propre au terrain lourd.
    valeur_pere = {p: rng.normal(0, 1) for p in peres}
    aptitude_pere_lourd = {p: rng.normal(0, 0.8) for p in peres}

    pere_de = {c: rng.choice(peres) for c in range(n_chevaux)}
    pm_de = {c: rng.choice(peres_mere) for c in range(n_chevaux)}

    # Valeur du cheval = part héritée + part propre.
    valeur = {
        c: heritabilite * valeur_pere[pere_de[c]]
        + np.sqrt(1 - heritabilite**2) * rng.normal(0, 1)
        for c in range(n_chevaux)
    }
    # Aptitude au lourd : héritée du père + variation individuelle.
    apt_lourd = {
        c: 0.6 * aptitude_pere_lourd[pere_de[c]] + 0.4 * rng.normal(0, 1)
        for c in range(n_chevaux)
    }
    # Distance de prédilection.
    dist_pref = {c: rng.choice([1600, 2100, 2700, 3200]) for c in range(n_chevaux)}
    disc_de = {c: rng.choice(DISCIPLINES) for c in range(n_chevaux)}

    valeur_driver = {d: rng.normal(0, 0.45) for d in range(120)}
    valeur_entr = {e: rng.normal(0, 0.30) for e in range(60)}

    # Compteurs de carrière, mis à jour au fil de l'eau (jamais depuis le futur).
    carriere = {c: {"n": 0, "v": 0, "p": 0, "gains": 0.0, "musique": []}
                for c in range(n_chevaux)}

    lignes = []
    t0 = pd.Timestamp("2023-01-01", tz="UTC")

    for i in range(n_courses):
        n = int(rng.integers(8, 19))
        partants = rng.choice(n_chevaux, size=n, replace=False)
        heure = t0 + pd.Timedelta(hours=5 * i + int(rng.integers(0, 4)))
        terrain = rng.choice(TERRAINS, p=[0.45, 0.28, 0.17, 0.10])
        distance = int(rng.choice([1600, 2100, 2700, 3200]))
        discipline = rng.choice(DISCIPLINES, p=[0.5, 0.15, 0.35])
        hippo = rng.choice(HIPPODROMES)
        allocation = float(rng.integers(12000, 120000))

        drivers = rng.choice(120, size=n, replace=False)
        entraineurs = rng.choice(60, size=n)

        # --- Ce qui est CONNAISSABLE avant le départ ---
        # Séparation essentielle : le score latent ne contient que ce qu'un
        # observateur parfait pourrait savoir. L'aléa de la course s'ajoute
        # APRÈS, et n'entre ni dans la probabilité vraie ni dans le marché.
        # Confondre les deux, c'est fabriquer un marché qui connaît déjà
        # l'arrivée — et un simulateur qui annonce 90 % de réussite.
        lourd = 1.0 if terrain in ("COLLANT", "LOURD") else 0.0
        latents = echelle * np.array([
            valeur[c]
            + apt_lourd[c] * lourd
            - 0.30 * abs(distance - dist_pref[c]) / 800.0
            + (0.35 if disc_de[c] == discipline else -0.15)
            + valeur_driver[d]
            + valeur_entr[e]
            for c, d, e in zip(partants, drivers, entraineurs)
        ])

        # Probabilité vraie = softmax des latents.
        expo = np.exp(latents - latents.max())
        proba_vraie = expo / expo.sum()

        # --- L'aléa de la course ---
        # Bruit de Gumbel : c'est le seul qui rende argmax(latent + bruit)
        # EXACTEMENT distribué selon softmax(latent) — le modèle de
        # Plackett-Luce. La probabilité annoncée est alors la vraie
        # fréquence, par construction, et non une approximation.
        perturbe = latents + rng.gumbel(0.0, 1.0, n)
        classement = np.argsort(-perturbe)            # 0 = vainqueur
        place_de = {int(partants[idx]): rang + 1 for rang, idx in enumerate(classement)}

        # --- Le marché : la vérité, vue à travers un verre dépoli ---
        proba_publique = proba_vraie * np.exp(rng.normal(0, bruit_marche, n))
        proba_publique /= proba_publique.sum()
        cotes = np.clip((1 - prelevement) / proba_publique, 1.1, 250.0)
        # Cote d'ouverture : encore plus bruitée — le marché s'affine.
        ouverture = np.clip(cotes * np.exp(rng.normal(0, 0.22, n)), 1.1, 300.0)

        seuil = 3 if n >= 8 else 2
        for j, c in enumerate(partants):
            c = int(c)
            car = carriere[c]
            place = place_de[c]
            lignes.append({
                "course_id": i,
                "heure_depart": heure,
                "num_pmu": j + 1,
                "id_cheval": c,
                "id_driver": int(drivers[j]),
                "id_entraineur": int(entraineurs[j]),
                "nom_pere": pere_de[c],
                "nom_pere_mere": pm_de[c],
                "discipline": discipline,
                "specialite": None,
                "distance": distance,
                "etat_terrain": terrain,
                "hippodrome_code": hippo,
                "nombre_partants": n,
                "montant_prix": allocation,
                "age": int(3 + min(car["n"] // 12, 9)),
                "sexe": rng.choice(["MALES", "FEMELLES", "HONGRES"]),
                "place_corde": j + 1,
                "handicap_poids": float(rng.integers(50, 62)),
                "deferre": rng.choice([None, "DEFERRE_ANTERIEURS", "DEFERRE_QUATRE_PIEDS"]),
                "oeilleres": rng.choice([None, "OEILLERES_CLASSIQUES"]),
                # Palmarès tel qu'il serait DÉCLARÉ avant le départ :
                # construit uniquement à partir des courses déjà disputées.
                "musique": " ".join(car["musique"][-6:][::-1]) or None,
                "nombre_courses": car["n"],
                "nombre_victoires": car["v"],
                "nombre_places": car["p"],
                "gains_carriere": round(car["gains"], 2),
                "gains_annee_en_cours": round(car["gains"] * 0.4, 2),
                "statut": "PARTANT",
                "ordre_arrivee": place,
                "cote_finale": round(float(cotes[j]), 2),
                "cote_ouverture": round(float(ouverture[j]), 2),
            })

            # Mise à jour APRÈS avoir écrit la ligne : le palmarès de la
            # course suivante inclura celle-ci, jamais l'inverse.
            car["n"] += 1
            car["v"] += int(place == 1)
            car["p"] += int(place <= seuil)
            car["gains"] += float(allocation * (0.5 if place == 1 else 0.12 if place <= seuil else 0))
            lettre = {"ATTELE": "a", "MONTE": "m", "PLAT": "p"}[discipline]
            car["musique"].append(f"{place if place <= 9 else 0}{lettre}")

    return pd.DataFrame(lignes)


if __name__ == "__main__":
    df = generer(n_courses=200)
    print(df.head())
    print(f"\n{len(df)} partants sur {df['course_id'].nunique()} courses")
