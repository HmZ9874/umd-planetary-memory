from __future__ import annotations

from benchmarks.umd334_adapter import QueryFissionResult
from benchmarks.umd335_adapter import (
    BackgroundGhostCatalog,
    constellation_budget,
    distill_ghost,
    temporal_aliases,
)


def test_ghost_distillation_keeps_fact_and_drops_request():
    text = (
        "user: Can you recommend a camera? I currently use a Nikon Z6 and "
        "prefer lightweight lenses. assistant: Here are ten cameras."
    )
    ghost, mass = distill_ghost(text)
    assert "Nikon Z6" in ghost
    assert "Here are ten" not in ghost
    assert mass > 0


def test_silent_ghost_does_not_invent_fact_from_generic_request():
    ghost, mass = distill_ghost("user: Can you recommend a movie? assistant: Sure.")
    assert ghost == ""
    assert mass == 0.0


def test_silent_catalog_does_not_activate_on_generic_preference_request():
    result = BackgroundGhostCatalog([
        "user: Can you recommend a movie? assistant: Sure.",
    ]).solve(
        "Can you recommend a movie?", size=1,
        fission=QueryFissionResult(periapsis_scores=(0.2,)),
    )
    assert not result.active


def test_temporal_aliases_are_prefix_relative():
    import datetime as dt

    aliases = temporal_aliases(dt.date(2026, 8, 1), dt.date(2026, 8, 8))
    assert "1 week ago" in aliases
    assert "last saturday" in aliases


def test_ghost_catalog_reanchors_to_original_source():
    texts = [
        "2026/08/01 user: I bought a Nikon Z6 and prefer light lenses.",
        "2026/08/08 user: Can you recommend a camera?",
    ]
    field = QueryFissionResult(periapsis_scores=(0.4, 0.5))
    result = BackgroundGhostCatalog(texts).solve(
        "Can you suggest lightweight lenses for my Nikon camera?",
        size=2, fission=field,
    )
    assert result.active
    assert result.order[0] == 0


def test_silent_source_cannot_enter_mixed_ghost_orbit():
    result = BackgroundGhostCatalog([
        "user: Can you recommend a Nikon camera?",
        "user: I own a Nikon Z6 and prefer lightweight lenses.",
    ]).solve(
        "Can you recommend a Nikon camera?", size=2,
        fission=QueryFissionResult(periapsis_scores=(1.0, 0.1)),
    )
    assert result.order == (1,)


def test_ghost_catalog_is_dormant_for_direct_non_temporal_query():
    result = BackgroundGhostCatalog(["user: I own a Nikon Z6."]).solve(
        "What camera do I own?", size=1, fission=QueryFissionResult(),
    )
    assert not result.active


def test_repeated_fact_resonance_activates_version_constellation():
    result = BackgroundGhostCatalog([
        "user: My personal best for the charity 5K was 31 minutes.",
        "user: My personal best for the charity 5K is now 28 minutes.",
    ]).solve(
        "What was my personal best time in the charity 5K?", size=2,
        fission=QueryFissionResult(periapsis_scores=(0.7, 0.8)),
    )
    assert "resonance" in result.modes
    assert result.constellation_budget == 6


def test_ghost_catalog_is_dormant_for_unmarked_documents():
    result = BackgroundGhostCatalog([
        "I prefer Nikon cameras and bought a lightweight lens last week.",
    ]).solve(
        "Can you recommend a camera?", size=1,
        fission=QueryFissionResult(periapsis_scores=(0.1,)),
    )
    assert not result.active


def test_preference_ghost_cannot_replace_strict_rank_one():
    result = BackgroundGhostCatalog([
        "2026/08/01 user: I prefer Nikon cameras and light lenses.",
        "2026/08/08 user: I own a Canon camera.",
    ]).solve(
        "Can you recommend lightweight lenses for my Nikon camera?",
        size=2, fission=QueryFissionResult(periapsis_scores=(0.2, 0.1)),
    )
    assert result.active
    assert result.promoted_source is None


def test_relative_date_filter_cannot_replace_strict_rank_one():
    result = BackgroundGhostCatalog([
        "2026/08/01 user: I met Rachel and bought a book.",
        "2026/08/08 user: I went to a concert.",
    ]).solve(
        "What did I do with Rachel last Saturday?", size=2,
        fission=QueryFissionResult(periapsis_scores=(0.9, 0.1)),
    )
    assert result.active
    assert result.promoted_source is None


def test_temporal_computation_may_promote_original_source():
    result = BackgroundGhostCatalog([
        "2026/08/01 user: I met Emma at lunch.",
        "2026/08/08 user: I went to a concert.",
    ]).solve(
        "How many days ago did I meet Emma?", size=2,
        fission=QueryFissionResult(periapsis_scores=(0.9, 0.1)),
    )
    assert result.promoted_source == 0


def test_constellation_budget_tracks_explicit_event_count():
    assert constellation_budget(
        "Put the six museum visits in order", ("temporal", "composite"),
    ) == 12


def test_constellation_budget_is_bounded_for_updates():
    assert constellation_budget(
        "What is my current personal best?", ("evolution",),
    ) == 5


def test_umd335_legacy_mode_does_not_enable_adaptive_constellations():
    catalog = BackgroundGhostCatalog([
        "user: My personal best for the charity 5K was 31 minutes.",
        "user: My personal best for the charity 5K is now 28 minutes.",
    ])
    direct = catalog.solve(
        "What was my personal best time in the charity 5K?", size=2,
        fission=QueryFissionResult(periapsis_scores=(0.7, 0.8)),
        adaptive_constellation=False,
    )
    temporal = catalog.solve(
        "How many days ago was my charity 5K?", size=2,
        fission=QueryFissionResult(periapsis_scores=(0.7, 0.8)),
        adaptive_constellation=False,
    )
    assert not direct.active
    assert temporal.constellation_budget == 2
