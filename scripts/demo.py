"""
Démonstration de bout en bout, sans base de données ni collecte.

    python scripts/demo.py

Fabrique un univers de courses dont on connaît la vérité, construit les
features, entraîne les deux variantes du modèle (avec et sans le marché),
et imprime le rapport d'évaluation complet.

C'est le moyen le plus rapide de vérifier que la chaîne tient debout —
et de voir à quoi ressemblera la sortie sur de vraies données.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE / "src"))
sys.path.insert(0, str(RACINE / "scripts"))

from pmu import evaluate as ev, features as ft          # noqa: E402
from pmu.train import Decoupage, ModelePmu, importances  # noqa: E402
from simulateur import generer                           # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
pd.set_option("display.width", 120)


def main() -> None:
    print("\n" + "=" * 66)
    print("  1. Génération de l'univers synthétique")
    print("=" * 66)
    brut = generer(n_courses=3000, n_chevaux=2500)
    print(f"  {len(brut):,} partants · {brut['course_id'].nunique():,} courses"
          f" · {brut['id_cheval'].nunique():,} chevaux".replace(",", " "))

    print("\n" + "=" * 66)
    print("  2. Construction des features")
    print("=" * 66)
    df = ft.construire(brut, avec_marche=True)
    df = df[df["est_exploitable"]].copy()
    sans = ft.colonnes_features(df, avec_marche=False)
    avec = ft.colonnes_features(df, avec_marche=True)
    print(f"  {len(sans)} features hors marché, {len(avec)} avec le marché")

    decoupage = Decoupage.par_proportions(df["heure_depart"], 0.6, 0.2)
    _, _, m_test = decoupage.masques(df["heure_depart"])
    print(f"  entraînement jusqu'au {decoupage.fin_train:%Y-%m-%d}, "
          f"calibration jusqu'au {decoupage.fin_calib:%Y-%m-%d}")
    print(f"  test : {m_test.sum():,} partants".replace(",", " "))

    resultats = {}
    for nom, avec_marche in [("SANS le marché", False), ("AVEC le marché", True)]:
        print("\n" + "=" * 66)
        print(f"  3. Modèle {nom}")
        print("=" * 66)

        modele = ModelePmu(cible="y_gagnant", avec_marche=avec_marche)
        modele.entrainer(df, decoupage)

        test = df[m_test].copy()
        pred = modele.predire(test)
        test["proba"] = pred["proba"].reindex(test.index)
        test["rang_modele"] = pred["rang"].reindex(test.index)
        test["ecart_top2"] = pred["ecart_top2"].reindex(test.index)

        rapport = ev.rapport(test)
        print(ev.afficher(rapport))
        resultats[nom] = (modele, test, rapport)

    # --- Ce que le modèle a réellement utilisé ---
    print("\n" + "=" * 66)
    print("  4. Features les plus utiles (modèle sans marché)")
    print("=" * 66)
    modele, test, _ = resultats["SANS le marché"]
    imp = importances(modele, test.sample(min(6000, len(test)), random_state=0), n=18)
    for _, r in imp.iterrows():
        barre = "█" * max(1, int(r["importance"] / imp["importance"].max() * 34))
        print(f"  {r['feature']:<26} {barre} {r['importance']:.5f}")

    # --- À quoi ressemble un pronostic ---
    print("\n" + "=" * 66)
    print("  5. Exemple de sortie sur une course")
    print("=" * 66)
    course = test[test["course_id"] == test["course_id"].iloc[len(test) // 2]]
    course = course.sort_values("proba", ascending=False)
    print(f"  {'n°':>3} {'proba':>8} {'cote':>7} {'implicite':>10} "
          f"{'valeur':>8} {'arrivée':>8}")
    for _, r in course.iterrows():
        valeur = r["proba"] * r["mkt_cote"] - 1 if pd.notna(r["mkt_cote"]) else float("nan")
        marque = " ←" if r["ordre_arrivee"] == 1 else ""
        print(f"  {int(r['num_pmu']):>3} {r['proba']:>8.3f} {r['mkt_cote']:>7.1f} "
              f"{r['mkt_proba_implicite']:>10.3f} {valeur:>+8.2f} "
              f"{int(r['ordre_arrivee']):>8}{marque}")
    conf = course["ecart_top2"].iloc[0]
    print(f"\n  écart entre le 1er et le 2e choix : {conf:.3f}"
          f"  → confiance {'élevée' if conf > 0.10 else 'faible'}")

    print("\n" + "=" * 66)
    print("  Lecture")
    print("=" * 66)
    print("""
  Le modèle SANS marché est le seul qui puisse rapporter quelque chose :
  s'il ne bat pas le marché, il ne sert qu'à confirmer le consensus.
  Le modèle AVEC marché donne le plafond de ce qui est prévisible.

  Sur données synthétiques, l'écart entre les deux mesure ce que le public
  sait déjà. Sur données réelles il sera plus grand — le monde est plus
  bruité que ce simulateur, et le public y est plus outillé.
""")


if __name__ == "__main__":
    main()
