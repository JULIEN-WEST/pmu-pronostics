"""
Pronostics du jour.

    python -m pmu.predict entrainer      # ré-entraîne et sauve le modèle
    python -m pmu.predict jour           # calcule les pronostics du jour
    python -m pmu.predict jour --date 2026-08-25

Le résultat est écrit dans la table `pronostic`, d'où l'API et le
publicateur MQTT le relisent. On ne recalcule jamais à la volée dans
l'API : une requête HTTP ne doit pas déclencher trente secondes de
calcul de features.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from . import dataset, db, evaluate as ev, explain as xp, features as ft
from .train import (Decoupage, ModeleParDiscipline, ModelePmu,
                    charger_modele, modele_present, resumer_arbitrage)

log = logging.getLogger("pmu.predict")

DOSSIER_MODELES = Path(os.environ.get("PMU_MODELES", "/data/modeles"))


def _par_discipline_par_defaut() -> bool:
    """
    Découpage par famille de discipline, actif par défaut.
    `PMU_PAR_DISCIPLINE=0` revient au modèle unique — utile pour comparer
    les deux sur ta propre base sans toucher au code.
    """
    return os.environ.get("PMU_PAR_DISCIPLINE", "1").strip() not in ("0", "non", "false")

SQL_TABLE = """
CREATE TABLE IF NOT EXISTS pronostic (
    course_id     bigint   NOT NULL REFERENCES course (course_id) ON DELETE CASCADE,
    num_pmu       smallint NOT NULL,
    proba         numeric(8,5) NOT NULL,
    rang          smallint     NOT NULL,
    ecart_top2    numeric(8,5),
    valeur        numeric(8,4),      -- proba × cote − 1
    cote          numeric(10,2),
    modele        text     NOT NULL, -- 'sans_marche' / 'avec_marche'
    calcule_le    timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (course_id, num_pmu, modele)
);
CREATE INDEX IF NOT EXISTS idx_pronostic_course ON pronostic (course_id, rang);
-- Justification du pronostic : motifs pondérés et faits chiffrés.
-- Ajoutée après coup, d'où l'ALTER : les bases déjà en service ne
-- doivent pas exiger de réinitialisation pour recevoir cette colonne.
ALTER TABLE pronostic ADD COLUMN IF NOT EXISTS details jsonb;
"""


# ---------------------------------------------------------------------

def entrainer(conn, *, avec_marche: bool = False, jusqua: date | None = None,
              par_discipline: bool | None = None) -> dict:
    """Ré-entraîne sur tout l'historique disponible et sauve le modèle."""
    jusqua = jusqua or date.today()
    if par_discipline is None:
        par_discipline = _par_discipline_par_defaut()
    brut = dataset.charger_pour_prediction(conn, jusqua)
    if brut.empty:
        raise RuntimeError("base vide — lancer la collecte avant l'entraînement")

    df = ft.construire(brut, avec_marche=True)
    df = df[df["est_cible"]].copy()
    if len(df) < 5000:
        log.warning("seulement %d partants exploitables — le modèle sera fragile", len(df))

    decoupage = Decoupage.par_proportions(df["heure_depart"], 0.6, 0.2)
    nom = "avec_marche" if avec_marche else "sans_marche"
    fabrique = ModeleParDiscipline if par_discipline else ModelePmu
    modele = fabrique(cible="y_gagnant", avec_marche=avec_marche).entrainer(df, decoupage)
    modele.sauver(DOSSIER_MODELES / nom)

    _, _, m_test = decoupage.masques(df["heure_depart"])
    test = df[m_test].copy()
    pred = modele.predire(test)
    test["proba"] = pred["proba"].reindex(test.index)
    rapport = ev.rapport(test)
    if par_discipline:
        rapport["arbitrage"] = modele.arbitrage

    texte = ev.afficher(rapport)
    if par_discipline:
        texte += ("\n\n── Arbitrage par famille " + "─" * 34 + "\n"
                  + resumer_arbitrage(modele.arbitrage))
    log.info("modèle %s entraîné\n%s", nom, texte)

    (DOSSIER_MODELES / nom).mkdir(parents=True, exist_ok=True)
    (DOSSIER_MODELES / nom / "rapport.txt").write_text(texte, encoding="utf-8")
    return rapport


def pronostiquer(conn, jour: date | None = None, *, modeles=("sans_marche",)) -> pd.DataFrame:
    """
    Calcule et enregistre les pronostics des courses du jour.

    Rappel du piège : on charge l'historique AVEC les courses du jour pour
    que les features glissantes aient de quoi se calculer, puis on ne garde
    que le jour demandé.
    """
    jour = jour or date.today()
    conn.execute(SQL_TABLE)

    brut = dataset.charger_pour_prediction(conn, jour)
    if brut.empty:
        log.warning("aucune donnée jusqu'au %s", jour)
        return pd.DataFrame()

    df = ft.construire(brut, avec_marche=True)
    # Le filtre porte sur est_cible : une performance importée datée
    # d'aujourd'hui (cas rare mais possible) n'est pas une course à
    # pronostiquer, c'est une trace du passé.
    du_jour = df[(df["date_reunion"] == jour) & df["est_cible"]].copy()
    # On garde les non-partants hors pronostic mais on les a laissés dans le
    # cadre pour ne pas casser les cumuls.
    du_jour = du_jour[du_jour["statut"].ne("NON_PARTANT")]
    if du_jour.empty:
        log.warning("aucun partant le %s", jour)
        return pd.DataFrame()

    sorties = []
    for nom in modeles:
        chemin = DOSSIER_MODELES / nom
        if not modele_present(chemin):
            log.warning("modèle %s absent (%s) — ignoré", nom, chemin)
            continue
        # `charger_modele` accepte les deux formes : modèle unique
        # (d'avant l'étape 4) comme modèle scindé par famille.
        modele = charger_modele(chemin)
        pred = modele.predire(du_jour)
        pred["modele"] = nom
        pred["cote"] = du_jour["mkt_cote"].reindex(pred.index)
        pred["valeur"] = pred["proba"] * pred["cote"] - 1.0

        # Justification. Elle ne doit JAMAIS faire échouer un pronostic :
        # mieux vaut une sélection sans explication qu'une journée perdue.
        try:
            motifs = xp.expliquer(modele, du_jour)
        except Exception as exc:                          # pragma: no cover
            log.warning("explication indisponible pour %s : %s", nom, exc)
            motifs = {}
        pred["details"] = [
            json.dumps(motifs.get((int(c), int(n)), {}), ensure_ascii=False)
            for c, n in zip(pred["course_id"], pred["num_pmu"])
        ]
        sorties.append(pred)

    if not sorties:
        raise RuntimeError(
            f"aucun modèle entraîné dans {DOSSIER_MODELES} — "
            "lancer `python -m pmu.predict entrainer`"
        )

    tout = pd.concat(sorties, ignore_index=True)
    lignes = [
        (int(r.course_id), int(r.num_pmu), float(r.proba), int(r.rang),
         float(r.ecart_top2), None if pd.isna(r.valeur) else float(r.valeur),
         None if pd.isna(r.cote) else float(r.cote), r.modele, r.details)
        for r in tout.itertuples()
    ]
    conn.cursor().executemany(
        """
        INSERT INTO pronostic (course_id, num_pmu, proba, rang, ecart_top2,
                               valeur, cote, modele, details)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
        ON CONFLICT (course_id, num_pmu, modele) DO UPDATE SET
            proba = EXCLUDED.proba, rang = EXCLUDED.rang,
            ecart_top2 = EXCLUDED.ecart_top2, valeur = EXCLUDED.valeur,
            cote = EXCLUDED.cote, details = EXCLUDED.details, calcule_le = now()
        """,
        lignes,
    )
    conn.commit()
    log.info("%d pronostics écrits pour le %s (%d courses)",
             len(lignes), jour, tout["course_id"].nunique())
    return tout


# ---------------------------------------------------------------------

def lire_pronostics(conn, jour: date, modele: str = "sans_marche") -> list[dict]:
    """
    Relit les pronostics enregistrés, groupés par course, prêts à sérialiser.
    C'est la source unique de l'API et du publicateur MQTT.
    """
    rows = conn.execute(
        """
        SELECT
            c.course_id, c.num_reunion, c.num_ordre, c.libelle AS course,
            c.discipline, c.distance, c.heure_depart, c.montant_prix,
            c.nombre_partants, c.etat_terrain, c.ordre_arrivee,
            h.libelle_long AS hippodrome, r.hippodrome_code,
            pr.num_pmu, pr.proba, pr.rang, pr.ecart_top2, pr.valeur, pr.cote,
            pr.calcule_le, pr.details,
            ch.nom AS cheval, ch.nom_pere, ch.nom_pere_mere,
            pd.nom_affiche AS driver, pe.nom_affiche AS entraineur, p.musique,
            p.age, p.sexe, p.place_corde, p.deferre,
            p.nombre_courses, p.nombre_victoires, p.gains_carriere,
            p.ordre_arrivee AS arrivee_cheval
        FROM pronostic pr
        JOIN course  c  ON c.course_id = pr.course_id
        JOIN reunion r  ON r.date_reunion = c.date_reunion AND r.num_officiel = c.num_reunion
        LEFT JOIN hippodrome h ON h.code = r.hippodrome_code
        JOIN partant p  ON p.course_id = pr.course_id AND p.num_pmu = pr.num_pmu
        LEFT JOIN cheval   ch ON ch.id_cheval = p.id_cheval
        LEFT JOIN personne pd ON pd.id = p.id_driver
        LEFT JOIN personne pe ON pe.id = p.id_entraineur
        WHERE c.date_reunion = %s AND pr.modele = %s
        ORDER BY c.heure_depart NULLS LAST, c.course_id, pr.rang
        """,
        (jour, modele),
    ).fetchall()

    courses: dict[int, dict] = {}
    for r in rows:
        cid = r["course_id"]
        if cid not in courses:
            courses[cid] = {
                "course_id": cid,
                "reunion": r["num_reunion"],
                "course": r["num_ordre"],
                "code": f"R{r['num_reunion']}C{r['num_ordre']}",
                "libelle": r["course"],
                "hippodrome": r["hippodrome"] or r["hippodrome_code"],
                "discipline": r["discipline"],
                "distance": r["distance"],
                "terrain": r["etat_terrain"],
                "allocation": float(r["montant_prix"]) if r["montant_prix"] else None,
                "partants": r["nombre_partants"],
                "depart": r["heure_depart"].isoformat() if r["heure_depart"] else None,
                "arrivee_connue": r["ordre_arrivee"] is not None,
                "confiance": float(r["ecart_top2"] or 0),
                "calcule_le": r["calcule_le"].isoformat() if r["calcule_le"] else None,
                "selection": [],
            }
        details = r["details"] or {}
        courses[cid]["selection"].append({
            "num": r["num_pmu"],
            "cheval": r["cheval"],
            "driver": r["driver"],
            "entraineur": r["entraineur"],
            "musique": r["musique"],
            "age": r["age"],
            "sexe": r["sexe"],
            "corde": r["place_corde"],
            "deferre": r["deferre"],
            "pere": r["nom_pere"],
            "pere_mere": r["nom_pere_mere"],
            "nb_courses": r["nombre_courses"],
            "nb_victoires": r["nombre_victoires"],
            "gains": float(r["gains_carriere"]) if r["gains_carriere"] is not None else None,
            "proba": round(float(r["proba"]), 4),
            "rang": r["rang"],
            "cote": float(r["cote"]) if r["cote"] is not None else None,
            "valeur": round(float(r["valeur"]), 3) if r["valeur"] is not None else None,
            "arrivee": r["arrivee_cheval"],
            # Le « pourquoi » : motifs pondérés et faits chiffrés.
            "motifs": details.get("motifs", []),
            "faits": details.get("faits", {}),
        })
    return list(courses.values())


# ---------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Pronostics PMU")
    ap.add_argument("mode", choices=["entrainer", "jour"])
    ap.add_argument("--date", type=lambda s: date.fromisoformat(s))
    ap.add_argument("--avec-marche", action="store_true",
                    help="entraîne aussi la variante qui voit la cote")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    )

    with db.connect() as conn:
        if args.mode == "entrainer":
            entrainer(conn, avec_marche=False, jusqua=args.date)
            if args.avec_marche:
                entrainer(conn, avec_marche=True, jusqua=args.date)
        else:
            modeles = ("sans_marche", "avec_marche") if args.avec_marche else ("sans_marche",)
            pronostiquer(conn, args.date, modeles=modeles)


if __name__ == "__main__":
    main()
