-- =====================================================================
--  pmu-pronostics — schéma relationnel
--  PostgreSQL 15+
--
--  Principe directeur : ce schéma stocke des FAITS HORODATÉS, jamais des
--  agrégats. Tout ce qui est « taux de réussite », « forme récente »,
--  « adéquation au terrain » est recalculé à la demande à partir des faits,
--  avec une borne temporelle stricte (cf. src/pmu/features.py).
--  Stocker un agrégat dans une table, c'est se garantir une fuite de données
--  le jour où on le recalcule sans faire attention à la date.
-- =====================================================================

CREATE SCHEMA IF NOT EXISTS pmu;
SET search_path TO pmu, public;


-- ---------------------------------------------------------------------
-- 1. Référentiels
-- ---------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS hippodrome (
    code            text PRIMARY KEY,          -- ex. 'VIN' (Vincennes)
    libelle_court   text,
    libelle_long    text,
    pays_code       text,
    pays_libelle    text
);

-- Drivers, jockeys, entraîneurs, propriétaires, éleveurs.
-- L'API PMU ne donne que des libellés ; on normalise pour pouvoir joindre.
-- nom_norme = upper, sans accents, ponctuation réduite (cf. normalize.norm_person).
CREATE TABLE IF NOT EXISTS personne (
    id          bigserial PRIMARY KEY,
    nom_norme   text NOT NULL UNIQUE,
    nom_affiche text NOT NULL
);

CREATE TABLE IF NOT EXISTS cheval (
    -- ⚠️ TEXTE, pas un entier. Le PMU compose l'identifiant à partir du
    -- nom, de la mère et du père : « KHAMEPHIS GAME-AKITA-ZARAK ».
    -- C'est stable, unique, et lisible — mais ce n'est pas un nombre.
    id_cheval       text PRIMARY KEY,
    nom             text NOT NULL,
    nom_norme       text NOT NULL,
    sexe            text,                      -- MALES / FEMELLES / HONGRES
    race            text,
    pays            text,
    annee_naissance smallint,

    -- Généalogie telle que fournie par le PMU : des NOMS, pas des identifiants.
    nom_pere        text,
    nom_mere        text,
    nom_pere_mere   text,
    -- Résolution vers de vrais identifiants quand IFCE / LeTrot la permettent.
    id_pere         text REFERENCES cheval (id_cheval),
    id_mere         text REFERENCES cheval (id_cheval),
    id_pere_mere    text REFERENCES cheval (id_cheval),

    id_eleveur      bigint REFERENCES personne (id),
    -- Rattachement aux sources externes
    numero_sire     text,                      -- IFCE / infochevaux
    id_letrot       text,
    maj_le          timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_cheval_nom_norme  ON cheval (nom_norme);
CREATE INDEX IF NOT EXISTS idx_cheval_pere       ON cheval (nom_pere);
CREATE INDEX IF NOT EXISTS idx_cheval_pere_mere  ON cheval (nom_pere_mere);


-- ---------------------------------------------------------------------
-- 2. Le programme : réunion → course → partant
-- ---------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS reunion (
    date_reunion    date    NOT NULL,
    num_officiel    integer NOT NULL,          -- le R de R1C3
    hippodrome_code text REFERENCES hippodrome (code),
    nature          text,                      -- DIURNE / NOCTURNE / SEMINOCTURNE
    audience        text,
    statut          text,
    pays_code       text,
    meteo           jsonb,                     -- températures, vent, nébulosité
    PRIMARY KEY (date_reunion, num_officiel)
);

CREATE TABLE IF NOT EXISTS course (
    course_id       bigserial PRIMARY KEY,
    date_reunion    date    NOT NULL,
    num_reunion     integer NOT NULL,
    num_ordre       integer NOT NULL,          -- le C de R1C3
    libelle         text,
    libelle_court   text,

    discipline      text,                      -- ATTELE / MONTE / PLAT / HAIES / STEEPLE / CROSS
    specialite      text,
    categorie_particularite text,
    categorie_statut        text,
    conditions      text,
    condition_age   text,
    condition_sexe  text,

    distance            integer,               -- mètres
    distance_unit       text,
    corde               text,                  -- CORDE_GAUCHE / CORDE_DROITE / NULL (trot)
    depart_type         text,                  -- AUTOSTART / VOLTE (trot)
    montant_prix        numeric(12,2),
    nombre_declares_partants smallint,
    nombre_partants     smallint,              -- après non-partants
    etat_terrain        text,                  -- BON / SOUPLE / COLLANT / LOURD ...
    penetrometre        numeric(6,2),

    heure_depart    timestamptz,               -- ⚠️ borne temporelle de référence
    statut          text,
    ordre_arrivee   jsonb,                     -- [[3],[7],[1,9]] — ex æquo possibles
    rapports_definitifs_disponibles boolean DEFAULT false,

    collecte_le     timestamptz NOT NULL DEFAULT now(),
    UNIQUE (date_reunion, num_reunion, num_ordre),
    FOREIGN KEY (date_reunion, num_reunion) REFERENCES reunion (date_reunion, num_officiel)
);
CREATE INDEX IF NOT EXISTS idx_course_heure       ON course (heure_depart);
CREATE INDEX IF NOT EXISTS idx_course_discipline  ON course (discipline, heure_depart);

CREATE TABLE IF NOT EXISTS partant (
    course_id       bigint  NOT NULL REFERENCES course (course_id) ON DELETE CASCADE,
    num_pmu         smallint NOT NULL,
    id_cheval       text    REFERENCES cheval (id_cheval),

    -- Identité au moment de la course (l'âge change, pas le cheval)
    age             smallint,
    sexe            text,
    race            text,

    id_driver       bigint REFERENCES personne (id),   -- driver (trot) ou jockey (galop)
    id_entraineur   bigint REFERENCES personne (id),
    id_proprietaire bigint REFERENCES personne (id),
    driver_change   boolean,

    -- Conditions de participation
    place_corde     smallint,                  -- galop : stalle ; trot : ligne/recul
    handicap_poids  numeric(6,2),              -- galop, en hg ou kg selon la source
    handicap_valeur numeric(6,2),
    handicap_distance integer,                 -- trot : distance de recul
    poids_condition_monte numeric(6,2),
    oeilleres       text,                      -- SANS_OEILLERES / OEILLERES_CLASSIQUES / AUSTRALIENNES
    deferre         text,                      -- DEFERRE_ANTERIEURS / POSTERIEURS / QUATRE_PIEDS / NULL
    supplement      numeric(12,2),
    engagement      boolean,
    jument_pleine   boolean,
    indicateur_inedit boolean,
    allure          text,

    -- Palmarès DÉCLARÉ par le PMU au moment de la course.
    -- Utilisable comme feature : c'est une info disponible avant le départ.
    musique         text,
    nombre_courses  smallint,
    nombre_victoires smallint,
    nombre_places   smallint,
    nombre_places_second smallint,
    nombre_places_troisieme smallint,
    gains_carriere  numeric(14,2),
    gains_victoires numeric(14,2),
    gains_place     numeric(14,2),
    gains_annee_en_cours numeric(14,2),
    gains_annee_precedente numeric(14,2),

    -- Résultat (rempli après la course)
    statut          text,                      -- PARTANT / NON_PARTANT
    ordre_arrivee   smallint,                  -- NULL si non classé / disqualifié
    statut_arrivee  text,                      -- PLACE / NON_PLACE / DISQUALIFIE / TOMBE ...
    temps_officiel_ms integer,
    reduction_km_ms integer,                   -- trot : ms/km
    distance_cheval_precedent text,
    commentaire_apres_course text,

    PRIMARY KEY (course_id, num_pmu)
);
CREATE INDEX IF NOT EXISTS idx_partant_cheval ON partant (id_cheval);
CREATE INDEX IF NOT EXISTS idx_partant_driver ON partant (id_driver);
CREATE INDEX IF NOT EXISTS idx_partant_entr   ON partant (id_entraineur);


-- ---------------------------------------------------------------------
-- 3. Le marché
-- ---------------------------------------------------------------------

-- Série temporelle des cotes. Une ligne par relevé.
-- C'est la table qui permet de mesurer la dérive de cote — un signal fort,
-- et le seul moyen honnête de savoir ce que « le marché » savait à T-5 min.
CREATE TABLE IF NOT EXISTS cote (
    course_id   bigint   NOT NULL REFERENCES course (course_id) ON DELETE CASCADE,
    num_pmu     smallint NOT NULL,
    releve_le   timestamptz NOT NULL,
    type_pari   text     NOT NULL,             -- SIMPLE_GAGNANT / SIMPLE_PLACE / E_SIMPLE_GAGNANT
    rapport     numeric(10,2),
    favoris     boolean,
    grosse_prise boolean,
    tendance    smallint,
    PRIMARY KEY (course_id, num_pmu, type_pari, releve_le)
);
CREATE INDEX IF NOT EXISTS idx_cote_course ON cote (course_id, releve_le DESC);

CREATE TABLE IF NOT EXISTS rapport_definitif (
    course_id   bigint NOT NULL REFERENCES course (course_id) ON DELETE CASCADE,
    type_pari   text   NOT NULL,
    combinaison text   NOT NULL,               -- '7' ou '7-3-11'
    rapport     numeric(12,2),
    mise_base   numeric(10,2),
    nombre_gagnants numeric(14,2),
    PRIMARY KEY (course_id, type_pari, combinaison)
);


-- ---------------------------------------------------------------------
-- 4. L'historique importé (performances-detaillees)
-- ---------------------------------------------------------------------

-- ⭐ Table stratégique. L'endpoint /performances-detaillees renvoie, pour
-- chaque partant du jour, le détail de ses courses PASSÉES — y compris
-- celles antérieures au début de votre collecte. C'est ce qui permet
-- d'amorcer un historique exploitable en quelques jours au lieu d'années
-- de rattrapage jour par jour.
--
-- Ces lignes sont moins riches qu'un `partant` collecté en direct
-- (pas de cote, pas de gains détaillés) : on les traite comme une source
-- SECONDAIRE, et on préfère toujours la ligne `partant` quand elle existe
-- pour la même course (cf. vue v_historique_cheval).
CREATE TABLE IF NOT EXISTS performance_passee (
    id              bigserial PRIMARY KEY,
    id_cheval       text   NOT NULL REFERENCES cheval (id_cheval),
    date_course     date   NOT NULL,
    hippodrome_lib  text,
    hippodrome_code text,
    nom_prix        text,
    discipline      text,
    specialite      text,
    distance        integer,
    allocation      numeric(12,2),
    nb_participants smallint,
    place           smallint,
    statut_arrivee  text,
    corde           smallint,
    poids_jockey    numeric(6,2),
    nom_jockey      text,
    oeillere        text,
    deferre         text,
    etat_terrain    text,
    temps_premier_ms integer,
    reduction_km_ms integer,
    distance_avec_precedent text,
    -- Lien vers la course interne si on l'a déjà collectée en direct.
    course_id       bigint REFERENCES course (course_id),
    source          text NOT NULL DEFAULT 'pmu_perf_detaillees',
    collecte_le     timestamptz NOT NULL DEFAULT now(),
    UNIQUE (id_cheval, date_course, hippodrome_lib, distance)
);
CREATE INDEX IF NOT EXISTS idx_perf_cheval_date ON performance_passee (id_cheval, date_course DESC);


-- ---------------------------------------------------------------------
-- 5. Journal de collecte
-- ---------------------------------------------------------------------

-- Rend la collecte reprenable : on sait exactement ce qui a été aspiré,
-- ce qui a échoué, et ce qui reste à re-tenter.
CREATE TABLE IF NOT EXISTS collecte_journal (
    id          bigserial PRIMARY KEY,
    ressource   text NOT NULL,                 -- 'programme' / 'participants' / 'perf'
    cle         text NOT NULL,                 -- '23082026' ou '23082026/R1/C3'
    statut      text NOT NULL,                 -- OK / VIDE / ERREUR
    http_code   integer,
    message     text,
    duree_ms    integer,
    fait_le     timestamptz NOT NULL DEFAULT now(),
    UNIQUE (ressource, cle)
);
CREATE INDEX IF NOT EXISTS idx_journal_statut ON collecte_journal (statut, ressource);


-- ---------------------------------------------------------------------
-- 6. Vues d'accès
-- ---------------------------------------------------------------------

-- Historique unifié d'un cheval : les courses collectées en direct
-- (riches) complétées par les performances importées (pauvres mais
-- profondes). La colonne `qualite` dit laquelle on regarde.
CREATE OR REPLACE VIEW v_historique_cheval AS
SELECT
    p.id_cheval,
    c.heure_depart                      AS moment,
    c.date_reunion                      AS date_course,
    c.course_id,
    c.discipline,
    c.distance,
    c.etat_terrain,
    r.hippodrome_code,
    c.nombre_partants                   AS nb_participants,
    p.ordre_arrivee                     AS place,
    p.statut_arrivee,
    p.reduction_km_ms,
    p.id_driver,
    p.id_entraineur,
    p.place_corde                       AS corde,
    p.gains_carriere,
    'direct'::text                      AS qualite
FROM partant p
JOIN course  c ON c.course_id = p.course_id
JOIN reunion r ON r.date_reunion = c.date_reunion AND r.num_officiel = c.num_reunion
WHERE p.statut IS DISTINCT FROM 'NON_PARTANT'

UNION ALL

SELECT
    pp.id_cheval,
    -- Pas d'heure exacte : on borne à minuit, ce qui exclut par construction
    -- la course du jour même. Volontairement conservateur.
    pp.date_course::timestamptz         AS moment,
    pp.date_course,
    pp.course_id,
    pp.discipline,
    pp.distance,
    pp.etat_terrain,
    pp.hippodrome_code,
    pp.nb_participants,
    pp.place,
    pp.statut_arrivee,
    pp.reduction_km_ms,
    NULL::bigint                        AS id_driver,
    NULL::bigint                        AS id_entraineur,
    pp.corde,
    NULL::numeric                       AS gains_carriere,
    'importe'::text                     AS qualite
FROM performance_passee pp
WHERE pp.course_id IS NULL;   -- si on a la course en direct, on ignore le doublon


-- Dernière cote connue avant le départ, par partant.
-- C'est LA probabilité implicite du marché, la référence à battre.
CREATE OR REPLACE VIEW v_cote_finale AS
SELECT DISTINCT ON (co.course_id, co.num_pmu, co.type_pari)
    co.course_id, co.num_pmu, co.type_pari, co.rapport, co.releve_le, co.favoris
FROM cote co
JOIN course c ON c.course_id = co.course_id
WHERE c.heure_depart IS NULL OR co.releve_le <= c.heure_depart
ORDER BY co.course_id, co.num_pmu, co.type_pari, co.releve_le DESC;

-- ---------------------------------------------------------------------
-- 7. Météo par hippodrome
-- ---------------------------------------------------------------------
--
-- ⚠️ Ces deux tables sont créées ICI, dans le schéma de base, et pas
-- seulement par `pmu.meteo`. La raison : `dataset.charger()` fait une
-- jointure externe dessus. Si elles n'existaient qu'après le premier
-- appel au module météo, TOUTE l'extraction échouerait sur
-- « relation meteo does not exist » tant qu'aucune météo n'a été
-- collectée — c'est-à-dire au premier démarrage.

CREATE TABLE IF NOT EXISTS meteo_lieu (
    hippodrome_code text PRIMARY KEY,
    libelle         text,
    latitude        double precision,
    longitude       double precision,
    -- Un géocodage qui échoue est mémorisé lui aussi : sans ça, chaque
    -- collecte retenterait indéfiniment les mêmes libellés introuvables.
    resolu          boolean NOT NULL DEFAULT false,
    tente_le        timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS meteo (
    hippodrome_code text NOT NULL,
    date_course     date NOT NULL,
    temperature     numeric(5,1),
    pluie_jour      numeric(6,2),   -- mm cumulés sur la journée de course
    pluie_24h       numeric(6,2),   -- mm cumulés sur les 24 h précédentes
    vent_max        numeric(5,1),
    humidite        numeric(5,1),
    source          text,           -- 'archive' / 'prevision'
    collecte_le     timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (hippodrome_code, date_course)
);


-- ---------------------------------------------------------------------
-- 8. Avis d'expert (DATAHIPPIQUE, servi par l'API PMU)
-- ---------------------------------------------------------------------
--
-- Un classement COMPLET des partants par un analyste, avec sa cote
-- probable, publié AVANT la course et consultable rétroactivement.
--
-- C'est le seul avis expert gratuit et structuré trouvé pour ce projet :
-- l'API généalogique de l'IFCE est commerciale (500 à 9 000 €/an) et
-- LeTrot ne publie aucune API. Celui-ci était dans l'API qu'on
-- interrogeait déjà.
--
-- ⚠️ C'est un AVIS, pas un fait. Il est corrélé au marché et n'a donc
-- rien à faire dans le modèle `sans_marche`, dont tout l'intérêt est
-- d'être indépendant du consensus.

CREATE TABLE IF NOT EXISTS pronostic_expert (
    course_id       bigint  NOT NULL REFERENCES course (course_id) ON DELETE CASCADE,
    num_pmu         smallint NOT NULL,
    rang_expert     smallint,
    cote_probable   numeric(10,2),   -- décimale : « 3/1 » vaut 4,00
    est_crible      boolean NOT NULL DEFAULT false,
    commentaire_expert text,
    source_expert   text,
    collecte_le     timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (course_id, num_pmu)
);
