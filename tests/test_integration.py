"""
Test d'intégration : schéma → insertion → extraction → features → API.

Il ne remplace pas un essai contre l'API PMU réelle, mais il valide tout
ce qui est sous notre contrôle :

  - le schéma SQL s'applique et les contraintes tiennent ;
  - les upserts sont bien idempotents (rejouer une collecte ne duplique
    rien et n'efface pas une arrivée déjà connue) ;
  - la requête d'extraction rend exactement les colonnes attendues par
    `features.construire()` ;
  - le job de pronostic écrit des lignes relisibles ;
  - l'API sert une charge utile conforme à ce que la vue Lovelace lit.

Ignoré si aucune base n'est joignable — passer PMU_TEST_DSN pour l'activer.

    PMU_TEST_DSN=postgresql://pmu:pmu@localhost:5433/pmu pytest tests/test_integration.py
"""

from __future__ import annotations

import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE / "scripts"))

DSN = os.environ.get("PMU_TEST_DSN")
pytestmark = pytest.mark.skipif(not DSN, reason="PMU_TEST_DSN non défini")

if DSN:
    os.environ["DATABASE_URL"] = DSN

from pmu import dataset, db, features as ft  # noqa: E402


# ---------------------------------------------------------------------
# Amorçage : on injecte l'univers du simulateur dans une vraie base.
# ---------------------------------------------------------------------

def _semer(conn, n_courses: int = 300) -> pd.DataFrame:
    from simulateur import generer

    brut = generer(n_courses=n_courses, n_chevaux=400)
    db.apply_schema(conn, str(RACINE / "sql" / "001_schema.sql"))

    hippos = brut["hippodrome_code"].unique()
    for code in hippos:
        db.upsert_hippodrome(conn, {"code": code, "libelle_court": code,
                                    "libelle_long": f"Hippodrome {code}",
                                    "pays_code": "FRA", "pays_libelle": "France"})

    jour0 = date(2023, 1, 1)
    for cid, grp in brut.groupby("course_id"):
        tete = grp.iloc[0]
        heure = pd.Timestamp(tete["heure_depart"]).to_pydatetime()
        num_r = 1 + (int(cid) % 4)
        num_c = 1 + (int(cid) // 4) % 9
        jour = heure.date()

        db.upsert_reunion(conn, {
            "date_reunion": jour, "num_officiel": num_r,
            "hippodrome_code": tete["hippodrome_code"], "nature": "DIURNE",
            "audience": "PUBLIQUE", "statut": "FIN_COURSE",
            "pays_code": "FRA", "meteo": {"nebulositeLibelle": "Ciel voilé"},
        })
        # numéro d'ordre rendu unique dans la réunion pour éviter les collisions
        num_c = 1 + (int(cid) % 9)
        course_id = db.upsert_course(conn, {
            "date_reunion": jour, "num_reunion": num_r, "num_ordre": num_c,
            "libelle": f"PRIX SYNTHETIQUE {cid}", "libelle_court": f"PS{cid}",
            "discipline": tete["discipline"], "specialite": None,
            "categorie_particularite": None, "categorie_statut": None,
            "conditions": None, "condition_age": None, "condition_sexe": None,
            "distance": int(tete["distance"]), "distance_unit": "METRE",
            "corde": None, "depart_type": None,
            "montant_prix": float(tete["montant_prix"]),
            "nombre_declares_partants": int(tete["nombre_partants"]),
            "nombre_partants": int(tete["nombre_partants"]),
            "etat_terrain": tete["etat_terrain"], "penetrometre": None,
            "heure_depart": heure, "statut": "FIN_COURSE",
            "ordre_arrivee": [[int(r.num_pmu)] for r in
                              grp.sort_values("ordre_arrivee").itertuples()],
            "rapports_definitifs_disponibles": True,
        })

        for r in grp.itertuples():
            p = {
                # Identifiant TEXTE, comme le PMU : nom-mère-père.
                "id_cheval": f"CHEVAL {r.id_cheval}-{r.nom_pere_mere}-{r.nom_pere}",
                "nom_cheval": f"CHEVAL {r.id_cheval}",
                "sexe": r.sexe, "race": "TROTTEUR FRANCAIS", "pays": "FR",
                "nom_pere": r.nom_pere, "nom_mere": None, "nom_pere_mere": r.nom_pere_mere,
                "num_pmu": int(r.num_pmu), "age": int(r.age),
                "place_corde": int(r.place_corde), "handicap_poids": float(r.handicap_poids),
                "handicap_valeur": None, "handicap_distance": None,
                "poids_condition_monte": None, "oeilleres": r.oeilleres,
                "deferre": r.deferre, "supplement": None, "engagement": True,
                "jument_pleine": False, "indicateur_inedit": False, "allure": None,
                "musique": r.musique, "nombre_courses": int(r.nombre_courses),
                "nombre_victoires": int(r.nombre_victoires),
                "nombre_places": int(r.nombre_places),
                "nombre_places_second": None, "nombre_places_troisieme": None,
                "gains_carriere": float(r.gains_carriere),
                "gains_victoires": None, "gains_place": None,
                "gains_annee_en_cours": float(r.gains_annee_en_cours),
                "gains_annee_precedente": None,
                "statut": "PARTANT", "ordre_arrivee": int(r.ordre_arrivee),
                "statut_arrivee": "PLACE" if r.ordre_arrivee <= 3 else "NON_PLACE",
                "temps_officiel_ms": None, "reduction_km_ms": None,
                "distance_cheval_precedent": None, "commentaire_apres_course": None,
                "driver_change": False,
            }
            id_d = db.upsert_personne(conn, f"DRIVER {r.id_driver}", f"DRIVER {r.id_driver}")
            id_e = db.upsert_personne(conn, f"ENTR {r.id_entraineur}", f"ENTR {r.id_entraineur}")
            db.upsert_cheval(conn, p)
            db.upsert_partant(conn, course_id, p, id_d, id_e, None)
            db.insert_cotes(conn, course_id, [{
                "num_pmu": int(r.num_pmu), "releve_le": heure - timedelta(minutes=3),
                "type_pari": "SIMPLE_GAGNANT", "rapport": float(r.cote_finale),
                "favoris": False, "grosse_prise": False, "tendance": 0,
            }, {
                "num_pmu": int(r.num_pmu), "releve_le": heure - timedelta(minutes=60),
                "type_pari": "SIMPLE_GAGNANT", "rapport": float(r.cote_ouverture),
                "favoris": False, "grosse_prise": False, "tendance": 0,
            }])
    conn.commit()
    return brut


@pytest.fixture(scope="module")
def base():
    with db.connect(DSN) as conn:
        conn.execute("DROP SCHEMA IF EXISTS pmu CASCADE")
        conn.commit()
        _semer(conn)
        yield conn


# ---------------------------------------------------------------------

def test_schema_et_volumetrie(base):
    s = dataset.stats(base)
    assert s["courses"] > 250
    assert s["partants"] > 3000
    assert s["chevaux"] > 300
    assert s["releves_cote"] == s["partants"] * 2
    assert s["courses_arrivees"] == s["courses"]


def test_upsert_idempotent(base):
    """Rejouer une collecte ne doit rien dupliquer."""
    avant = dataset.stats(base)
    _semer(base, n_courses=300)
    apres = dataset.stats(base)
    assert apres["courses"] == avant["courses"]
    assert apres["partants"] == avant["partants"]
    assert apres["releves_cote"] == avant["releves_cote"]


def test_arrivee_jamais_effacee(base):
    """
    Une course rejouée AVANT son arrivée ne doit pas effacer l'arrivée déjà
    collectée — c'est ce que garantissent les COALESCE des upserts.
    """
    row = base.execute(
        "SELECT course_id, date_reunion, num_reunion, num_ordre "
        "FROM course WHERE ordre_arrivee IS NOT NULL LIMIT 1"
    ).fetchone()
    db.upsert_course(base, {
        "date_reunion": row["date_reunion"], "num_reunion": row["num_reunion"],
        "num_ordre": row["num_ordre"], "libelle": None, "libelle_court": None,
        "discipline": None, "specialite": None, "categorie_particularite": None,
        "categorie_statut": None, "conditions": None, "condition_age": None,
        "condition_sexe": None, "distance": None, "distance_unit": None,
        "corde": None, "depart_type": None, "montant_prix": None,
        "nombre_declares_partants": None, "nombre_partants": None,
        "etat_terrain": None, "penetrometre": None, "heure_depart": None,
        "statut": None, "ordre_arrivee": None,          # ← on tente d'effacer
        "rapports_definitifs_disponibles": False,
    })
    base.commit()
    apres = base.execute(
        "SELECT ordre_arrivee, distance FROM course WHERE course_id = %s",
        (row["course_id"],)
    ).fetchone()
    assert apres["ordre_arrivee"] is not None
    assert apres["distance"] is not None


def test_extraction_donne_les_colonnes_attendues(base):
    df = dataset.charger(base, date(2023, 1, 1), date(2030, 1, 1))
    assert len(df) > 3000
    requises = {
        "course_id", "heure_depart", "discipline", "distance", "etat_terrain",
        "hippodrome_code", "nombre_partants", "montant_prix", "num_pmu",
        "id_cheval", "id_driver", "id_entraineur", "nom_pere", "nom_pere_mere",
        "age", "sexe", "place_corde", "handicap_poids", "deferre", "oeilleres",
        "musique", "nombre_courses", "nombre_victoires", "nombre_places",
        "gains_carriere", "gains_annee_en_cours", "ordre_arrivee", "statut",
        "cote_finale", "cote_ouverture",
    }
    manquantes = requises - set(df.columns)
    assert not manquantes, f"colonnes absentes de l'extraction : {manquantes}"
    assert df["cote_finale"].notna().all()
    assert df["cote_ouverture"].notna().all()


def test_extraction_ne_renvoie_pas_les_noms_de_colonnes(base):
    """
    Régression d'un bug silencieux et coûteux.

    `pd.read_sql(sql, conn)` sur une connexion psycopg en `dict_row` itère
    les CLÉS de chaque dict au lieu des valeurs : chaque colonne se remplit
    de son propre nom. `statut` vaut `"statut"` sur toutes les lignes,
    `ordre_arrivee` vaut `"ordre_arrivee"`, aucune exception n'est levée,
    `est_exploitable` tombe à zéro et le modèle s'entraîne sur du vide.

    Le symptôme observé était « fenêtres d'entraînement vides » — trois
    couches plus loin que la cause.
    """
    df = dataset.charger(base, date(2023, 1, 1), date(2030, 1, 1))
    for col in df.columns:
        valeurs = df[col].dropna().unique()
        assert not (len(valeurs) == 1 and valeurs[0] == col), (
            f"la colonne {col!r} ne contient que son propre nom — "
            "l'extraction itère les clés au lieu des valeurs"
        )
    assert df["statut"].eq("PARTANT").all()
    assert pd.api.types.is_numeric_dtype(pd.to_numeric(df["ordre_arrivee"]))


def test_cote_finale_est_bien_la_derniere(base):
    """
    La vue doit rendre le relevé le plus proche du départ, pas le premier.
    Le simulateur produit une ouverture plus bruitée que la cote finale ;
    on vérifie que les deux colonnes diffèrent réellement.
    """
    df = dataset.charger(base, date(2023, 1, 1), date(2030, 1, 1))
    assert (df["cote_finale"] != df["cote_ouverture"]).mean() > 0.8


def test_chaine_complete_features(base):
    """L'extraction alimente `construire()` sans retouche."""
    brut = dataset.charger(base, date(2023, 1, 1), date(2030, 1, 1))
    df = ft.construire(brut, avec_marche=True)
    assert df["est_exploitable"].sum() > 3000
    assert df["y_gagnant"].sum() > 250
    cols = ft.colonnes_features(df, avec_marche=True)
    assert len(cols) > 60
    assert df["mkt_proba_implicite"].notna().any()
    # Les probabilités implicites renormalisées somment à 1 par course.
    somme = df.groupby("course_id")["mkt_proba_implicite"].sum()
    assert somme.between(0.99, 1.01).mean() > 0.95


def test_genealogie_resolue_sans_homonymie(base):
    """
    `link_genealogie` ne lie que les noms uniques. Nos pères synthétiques ne
    sont pas des chevaux de la base : rien ne doit être lié à tort.
    """
    db.link_genealogie(base)
    base.commit()
    lies = base.execute(
        "SELECT count(*) AS n FROM cheval WHERE id_pere IS NOT NULL"
    ).fetchone()
    assert lies["n"] == 0


def test_pronostic_et_relecture(base):
    """Entraînement court, pronostic, relecture — la chaîne de service."""
    from pmu import predict

    jour = base.execute("SELECT max(date_reunion) AS d FROM course").fetchone()["d"]
    predict.DOSSIER_MODELES = RACINE / ".tmp_modeles"
    predict.entrainer(base, avec_marche=False, jusqua=jour)
    tout = predict.pronostiquer(base, jour)
    assert len(tout) > 0

    courses = predict.lire_pronostics(base, jour)
    assert courses, "aucune course relue"
    c = courses[0]
    assert {"code", "hippodrome", "selection", "confiance"} <= set(c)
    assert c["selection"], "course sans sélection"
    assert c["selection"][0]["rang"] == 1
    # Les probabilités d'une course somment à 1 : c'est la normalisation
    # intra-course, et c'est ce qui rend le classement lisible.
    total = sum(s["proba"] for s in c["selection"])
    assert 0.97 <= total <= 1.03, f"somme des probabilités = {total}"

    # La justification doit survivre à l'aller-retour PostgreSQL. C'est
    # le seul endroit où le jsonb est réellement écrit puis relu ; un
    # test en mémoire ne prouverait rien sur la sérialisation.
    brut = base.execute(
        "SELECT count(*) AS n FROM pronostic WHERE details IS NOT NULL"
    ).fetchone()
    assert brut["n"] > 0, "aucune justification écrite en base"

    avec_motifs = [s for co in courses for s in co["selection"] if s["motifs"]]
    assert avec_motifs, "aucun motif relu depuis la base"
    m = avec_motifs[0]["motifs"][0]
    assert {"groupe", "titre", "icone", "sens", "poids"} <= set(m)
    assert m["sens"] in ("+", "−")
    assert isinstance(m.get("details"), list)
    # Les champs qui nourrissent la vue : ils doivent être là, quitte à
    # valoir None, sinon la vue affiche « undefined ».
    for champ in ("pere", "musique", "gains", "nb_courses", "entraineur", "faits"):
        assert champ in avec_motifs[0], f"champ {champ} absent de la sélection"


def test_insertion_d_un_participant_reel(base):
    """
    LE test qui manquait.

    Toute la suite s'appuyait sur des fixtures que j'avais inventées, avec
    un `idCheval` entier et des champs texte bien sages. La vraie API
    renvoie un identifiant en CHAÎNE et trois champs en OBJET. Résultat :
    tests au vert, collecte à zéro partant en production, et pas une seule
    erreur pour le signaler.

    Ce test part de la charge utile réellement observée sur
    /programme/22082026/R1/C1/participants et la fait traverser toute la
    chaîne jusqu'à l'insertion.
    """
    from pmu import normalize as nz

    reel = {
        "numPmu": 1,
        "idCheval": "KHAMEPHIS GAME-AKITA-ZARAK",   # chaîne, pas entier
        "nom": "KHAMEPHIS GAME",
        "pays": "France",
        "age": 6, "sexe": "HONGRES", "race": "PUR-SANG", "statut": "PARTANT",
        "placeCorde": 6, "oeilleres": "SANS_OEILLERES",
        "proprietaire": "ECURIE HARAS DU CHATEAU",
        "entraineur": "M.BRASME (S)", "driver": "M. PROTTI",
        "driverChange": True, "indicateurInedit": False,
        "musique": "0p1p0p(25)4p1p1p3p9p0p",
        "nombreCourses": 24, "nombreVictoires": 4, "nombrePlaces": 12,
        "nombrePlacesSecond": 1, "nombrePlacesTroisieme": 4,
        "gainsParticipant": {"gainsCarriere": 3820000, "gainsVictoires": 1500000,
                             "gainsPlace": 900000, "gainsAnneeEnCours": 720000,
                             "gainsAnneePrecedente": 640000},
        "handicapValeur": 32.0, "handicapPoids": 595,
        "poidsConditionMonte": 580, "poidsConditionMonteChange": True,
        "nomPere": "ZARAK", "nomMere": "AKITA", "nomPereMere": "GIANT'S CAUSEWAY",
        "ordreArrivee": 2, "jumentPleine": False, "engagement": False,
        "supplement": 0, "eleveur": "ECURIE HARAS DU CHATEAU", "allure": "GALOP",
        # Les trois objets déguisés en texte
        "robe": {"code": "001", "libelleCourt": "ALEZAN", "libelleLong": "ALEZAN"},
        "commentaireApresCourse": {"texte": "Bien placé, a fini fort.", "source": "PMU"},
        "distanceChevalPrecedent": {"libelleCourt": "1/2 L",
                                    "libelleLong": "une demi-longueur",
                                    "code": 3, "identifiant": "DEMI_LONGUEUR"},
        "dernierRapportDirect": {"typePari": "SIMPLE_GAGNANT", "rapport": 12.0,
                                 "favoris": True, "dateRapport": 1787515200000,
                                 "nombreIndicateurTendance": -2, "grossePrise": False},
    }

    row = base.execute("SELECT course_id FROM course LIMIT 1").fetchone()
    course_id = row["course_id"]

    p = nz.parse_participant(reel)
    assert p["id_cheval"] == "KHAMEPHIS GAME-AKITA-ZARAK"

    id_d = db.upsert_personne(base, p["driver"], nz.norm_person(p["driver"]))
    id_e = db.upsert_personne(base, p["entraineur"], nz.norm_person(p["entraineur"]))
    db.upsert_cheval(base, p)
    # num_pmu 1 existe déjà dans la course : on décale pour ne pas écraser.
    p["num_pmu"] = 99
    db.upsert_partant(base, course_id, p, id_d, id_e, None)
    db.insert_cotes(base, course_id, nz.parse_cotes(reel, datetime.now(timezone.utc)))
    base.commit()

    stocke = base.execute(
        """
        SELECT p.id_cheval, p.commentaire_apres_course, p.distance_cheval_precedent,
               p.gains_carriere, p.ordre_arrivee, c.nom, c.nom_pere_mere
          FROM partant p JOIN cheval c ON c.id_cheval = p.id_cheval
         WHERE p.course_id = %s AND p.num_pmu = 99
        """,
        (course_id,),
    ).fetchone()

    assert stocke is not None, "le partant réel n'a pas été inséré"
    assert stocke["id_cheval"] == "KHAMEPHIS GAME-AKITA-ZARAK"
    assert stocke["commentaire_apres_course"] == "Bien placé, a fini fort."
    assert stocke["distance_cheval_precedent"] == "une demi-longueur"
    assert float(stocke["gains_carriere"]) == 38200.0
    assert stocke["ordre_arrivee"] == 2
    assert stocke["nom_pere_mere"] == "GIANT S CAUSEWAY"   # normalisé


def test_api_sert_la_charge_utile_ha(base):
    """Ce que consommera la vue Lovelace."""
    from fastapi.testclient import TestClient
    from pmu import api

    client = TestClient(api.app)

    sante = client.get("/sante").json()
    assert sante["ok"] is True
    assert sante["courses"] > 250

    resume = client.get("/ha/resume").json()
    assert "programme" in resume and "prochaine" in resume
    # La charge utile doit rester sous la limite d'attributs de HA.
    import json
    assert len(json.dumps(resume, ensure_ascii=False).encode()) < 16000
