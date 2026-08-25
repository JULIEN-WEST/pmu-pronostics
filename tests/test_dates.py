"""
Fuseau horaire des dates de course.

Bug réel, trouvé sur les données de production : les champs de date du
PMU valent MINUIT heure de Paris. Lus en UTC, ils basculent la veille —
22 h 00 en été — et toute la base se décale d'un jour.

Symptôme observé : la collecte annonce « 2026-08-25 — 5 réunions, 40
courses, 470 partants », `/sante` répond `jusqua: 2026-08-24`, et le
calcul des pronostics conclut « aucun partant le 2026-08-25 ».

Valeurs de ce fichier : relevées sur l'API le 25/08/2026.
"""

from __future__ import annotations

from datetime import date

import pytest

from pmu import normalize as nz


# dateReunion de la première réunion du 25/08/2026.
# Le PMU joint lui-même timezoneOffset = 7200000 (2 h) dans sa réponse.
MINUIT_25_AOUT_PARIS = 1787608800000


def test_minuit_paris_ne_bascule_pas_la_veille():
    """Le cas exact qui a décalé toute la base."""
    assert nz.ms_to_date(MINUIT_25_AOUT_PARIS) == date(2026, 8, 25)


def test_l_instant_reste_correct():
    """
    Seule la DATE calendaire était fausse. L'instant, lui, était juste —
    et c'est l'instant (`heure_depart`) qui sert à l'ordre chronologique
    des features. La règle anti-fuite n'a donc jamais été compromise.
    """
    dt = nz.ms_to_dt(MINUIT_25_AOUT_PARIS)
    assert dt.isoformat() == "2026-08-24T22:00:00+00:00"


def test_heure_de_depart_inchangee():
    """Départ de la première course : 2026-08-25 14:18 à Paris."""
    dt = nz.ms_to_dt(1787660280000)
    assert dt.astimezone(nz.FUSEAU_COURSES).strftime("%Y-%m-%d %H:%M") == "2026-08-25 14:18"


@pytest.mark.parametrize("ms, attendu", [
    # Minuit Paris en HIVER (UTC+1) : 23 h 00 UTC la veille.
    (1767222000000, date(2026, 1, 1)),
    # Minuit Paris en ÉTÉ (UTC+2) : 22 h 00 UTC la veille.
    (1787608800000, date(2026, 8, 25)),
])
def test_le_basculement_heure_ete_hiver_est_gere(ms, attendu):
    """
    Un décalage figé à +2 h casserait en hiver, et +1 h casserait en été.
    D'où le recours à une vraie base de fuseaux plutôt qu'une constante.
    """
    assert nz.ms_to_date(ms) == attendu


def test_midi_donne_le_meme_jour_dans_les_deux_lectures():
    """Contrôle de non-régression : un horodatage de plein jour ne bouge pas."""
    midi_utc = 1787659200000          # 2026-08-25 14:00 Paris
    assert nz.ms_to_date(midi_utc) == date(2026, 8, 25)


def test_valeurs_invalides_toujours_tolerees():
    for valeur in (None, "", 0, "abc"):
        assert nz.ms_to_date(valeur) is None
