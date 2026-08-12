"""Retrieval-layer evaluation for heterogeneous public memory benchmarks.

The native benchmark protocol is preserved whenever it exposes source labels.
Benchmarks whose official score requires an answer LLM, judge LLM, tools, or a
multimodal model are explicitly labelled as proxy evaluations here.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from benchmarks.public_benchmarks import (
    BM25, char_tokenize, ranking, reciprocal_ranks, tokenize, unit_scores,
)
from benchmarks.umd39_benchmarks import (
    MODEL_CACHE, RESULTS, CapsuleRecallAccumulator, FastEmbedEncoder,
    _anchored_capsule_order, _benchmark_capsules, _directive_metadata,
    _dot_scores, _entity_value_resonance, _force, _graph_flux, _orbit_expand,
    _temporal_phase,
)
from benchmarks.umd314_adapter import (
    directive_gravity_values, fuse_transient_bridge, relative_gravity_order,
    transient_directive_bridge,
)
from benchmarks.umd315_adapter import compress_event_horizon, rank_event_satellites
from benchmarks.umd316_adapter import (
    apply_state_field,
    conserved_constellation_orbit,
    constellation_budget,
    first_capsule_lagrange_order,
    rank_constellation_coverage,
    split_state_orbits,
    state_mass_adjustment,
)
from benchmarks.umd317_adapter import (
    compile_fact_planets,
    conserved_fact_constellation,
    fact_constellation_budget,
    first_capsule_focus_tether,
    rank_fact_planet_coverage,
)
from benchmarks.umd318_adapter import roche_lobe_first_capsule
from benchmarks.umd325_adapter import (
    galactic_census_budget,
    is_explicit_antimatter,
    rank_galactic_census,
    split_antimatter_orbits,
)
from benchmarks.umd326_adapter import (
    EmergentDomainGraph,
    collapse_census_singularity,
    compact_census_payload,
    compile_versioned_facts,
    is_census_query,
    singularity_budget,
)
from benchmarks.umd327_adapter import (
    StructuredCensusLedger,
    ledger_census_budget,
    ledger_census_orbit,
    solve_compact_numeric,
)
from benchmarks.umd330_adapter import FactSlingshotGraph, SlingshotResult
from benchmarks.umd331_adapter import EvidenceClosureGraph
from benchmarks.umd332_adapter import RelationSuperpositionGraph
from benchmarks.umd333_adapter import (
    AbsorbingRelationGraph,
    DocumentStarField,
    DocumentStarResult,
)


VENDOR = Path(__file__).resolve().parent / "vendor"


_ESCAPE_VELOCITY_QUERY_RE = re.compile(
    r"\b(?:why|would|might|likely|potential(?:ly)?|suspect(?:ed)?|"
    r"state|country|live|during|before|after|challenges?|"
    r"personality|traits?|relationship|compare|differ(?:ence|ent)?)\b|"
    r"\band\s+how\b",
    re.IGNORECASE,
)

_SUPERPOSITION_BRIDGE_QUERY_RE = re.compile(
    r"\b(?:both|same|older|younger|companion|whose|portrayed|starring|"
    r"recruited|formed by|located in|based in|fight song|main campus|"
    r"who .+ who|of the .+ (?:who|that|whose)|and .+ (?:both|same))\b",
    re.IGNORECASE,
)


def _bounded_lexical_bridge(
    query: str, texts: Sequence[str], index: Any, lexical_order: Sequence[int],
) -> list[float]:
    """Two-hop pseudo-relevance field over rare terms in top lexical chunks.

    A term must occur in at least two and at most twelve chunks, so one seed
    can expose an entity or title that leads to another source without letting
    corpus-wide generic words dominate.  The field is query/source-only and is
    disabled for short direct questions and index views without postings.
    """
    size = len(texts)
    query_tokens = set(tokenize(query))
    if (
        size < 2
        or not hasattr(index, "postings")
        or (len(query_tokens) < 10 and not _SUPERPOSITION_BRIDGE_QUERY_RE.search(query))
    ):
        return [0.0] * size
    bridge = [0.0] * size
    for seed_rank, seed in enumerate(lexical_order[:6], 1):
        weighted_terms: list[tuple[float, str]] = []
        for term in set(tokenize(texts[seed])) - query_tokens:
            if len(term) < 4:
                continue
            posting = index.postings.get(term)
            if posting is None or not 2 <= len(posting) <= 12:
                continue
            idf = math.log(1.0 + (size - len(posting) + 0.5) / (len(posting) + 0.5))
            weighted_terms.append((idf, term))
        for idf, term in sorted(weighted_terms, reverse=True)[:64]:
            for source, frequency in index.postings[term]:
                if source == seed or source >= size:
                    continue
                bridge[source] += idf * min(2, frequency) / seed_rank
    return unit_scores(bridge)


def escape_velocity_budget(query: str, base_budget: int, ceiling: int = 288) -> int:
    """Expand L1 only for queries likely to require indirect evidence.

    UMD 3.24 models an inferential question as a body whose information
    velocity can escape the ordinary Roche lobe. A detected causal,
    counterfactual, geographic, temporal, or synthesis signal triples the
    bounded field; direct fact questions retain the configured base budget.
    The rule observes only query text and never benchmark labels.
    """
    base = max(1, int(base_budget))
    if not _ESCAPE_VELOCITY_QUERY_RE.search(query):
        return base
    return min(max(base, int(ceiling)), 3 * base)


class _PrefixBM25View:
    def __init__(self, index: BM25, size: int) -> None:
        self.index = index
        self.size = size

    def scores(self, query: str) -> list[float]:
        return self.index.scores_prefix(query, self.size)


class OrbitIndex:
    """A reusable UMD index over ordered, provenance-bearing sources."""

    def __init__(
        self,
        texts: Sequence[str],
        groups: Sequence[int],
        source_ids: Sequence[str],
        encoder: FastEmbedEncoder,
        dates: Sequence[str] | None = None,
        neural_candidate_pool: int = 0,
        physics_v316: bool = False,
        physics_v317: bool = False,
        physics_v318: bool = False,
        physics_v320: bool = False,
        physics_v321: bool = False,
        physics_v322: bool = False,
        physics_v323: bool = False,
        physics_v324: bool = False,
        physics_v325: bool = False,
        physics_v326: bool = False,
        physics_v327: bool = False,
        physics_v330: bool = False,
        physics_v331: bool = False,
        physics_v332: bool = False,
        physics_v333: bool = False,
        experimental_lexical_bridge_v332: bool = False,
        roche_budget_v318: int = 10,
        matter_neighbor_budget: int = 0,
        cross_encoder: Any | None = None,
        cross_limit: int = 160,
        roche_ranker: Any | None = None,
        first_orbit_v316: bool = True,
        structured_records: Sequence[dict[str, Any]] | None = None,
    ) -> None:
        self.texts = list(texts)
        self.groups = list(groups)
        self.source_ids = list(source_ids)
        self.encoder = encoder
        self.dates = list(dates or [])
        self.neural_candidate_pool = max(0, neural_candidate_pool)
        self.physics_v333 = bool(physics_v333)
        self.physics_v332 = bool(physics_v332 or self.physics_v333)
        self.experimental_lexical_bridge_v332 = bool(experimental_lexical_bridge_v332)
        self.physics_v331 = bool(physics_v331 or self.physics_v332)
        self.physics_v330 = bool(physics_v330 or self.physics_v331)
        self.physics_v327 = bool(physics_v327 or self.physics_v330)
        self.physics_v326 = bool(physics_v326 or self.physics_v327)
        self.physics_v325 = bool(physics_v325 or self.physics_v326)
        self.physics_v324 = bool(physics_v324)
        self.physics_v323 = bool(physics_v323 or physics_v324)
        self.physics_v322 = bool(physics_v322 or self.physics_v323)
        self.physics_v321 = bool(physics_v321 or self.physics_v322)
        self.physics_v320 = bool(physics_v320 or self.physics_v321)
        self.physics_v318 = bool(physics_v318 or self.physics_v320)
        self.matter_neighbor_budget = min(9, max(0, int(matter_neighbor_budget)))
        roche_ceiling = 288 if self.physics_v324 else 96 if self.physics_v323 else 48
        self.roche_budget_v318 = min(roche_ceiling, max(1, int(roche_budget_v318)))
        self.cross_encoder = cross_encoder
        self.cross_limit = min(512, max(1, int(cross_limit)))
        self.roche_ranker = roche_ranker
        self.last_roche_diagnostics: dict[str, Any] = {}
        self.physics_v317 = bool(physics_v317 or self.physics_v318 or self.physics_v325)
        self.physics_v316 = bool(physics_v316 or self.physics_v317)
        self.first_orbit_v316 = bool(first_orbit_v316)
        self.vectors = (
            encoder.encode_many(self.texts)
            if self.texts and not self.neural_candidate_pool else []
        )
        self._lazy_vectors: dict[int, np.ndarray] = {}
        self._full_group_map = {
            group: index for index, group in enumerate(dict.fromkeys(self.groups))
        }
        self._full_groups = [self._full_group_map[group] for group in self.groups]
        full_members: dict[int, list[int]] = defaultdict(list)
        for source, group in enumerate(self._full_groups):
            full_members[group].append(source)
        self._full_session_texts = [
            " ".join(self.texts[source] for source in full_members[group])
            for group in range(len(full_members))
        ]
        self._full_dates = [
            self.dates[original] if original < len(self.dates) else ""
            for original in self._full_group_map
        ]
        self._full_lexical_index = BM25(self.texts) if self.texts else None
        self._full_char_index = BM25(self.texts, tokenizer=char_tokenize) if self.texts else None
        self._full_planet_index = BM25(self._full_session_texts) if self._full_session_texts else None
        self._full_capsules, self._full_by_source = _benchmark_capsules(self._full_groups)
        self._full_directive_meta = [
            _directive_metadata(text, trusted=True) for text in self.texts
        ]
        self._fact_planets = (
            compile_fact_planets(self.texts)
            if self.physics_v317 and not self.physics_v325 else []
        )
        self._census_tokens = (
            [frozenset(tokenize(text)) for text in self.texts]
            if self.physics_v325 else []
        )
        self._antimatter_source_ids = frozenset(
            str(self.source_ids[index])
            for index, text in enumerate(self.texts)
            if self.physics_v325 and is_explicit_antimatter(text)
        )
        self._domain_graph = (
            EmergentDomainGraph(self._census_tokens) if self.physics_v326 else None
        )
        self.structured_records = list(structured_records or [])
        self._versioned_facts = (
            compile_versioned_facts(self.structured_records, self.source_ids)
            if self.physics_v326 and len(self.structured_records) == len(self.source_ids)
            else ()
        )
        self._census_ledger = (
            StructuredCensusLedger(self.structured_records)
            if self.physics_v327 and len(self.structured_records) == len(self.source_ids)
            else None
        )
        self._slingshot_graph = (
            AbsorbingRelationGraph(self.texts)
            if self.physics_v333
            else RelationSuperpositionGraph(self.texts)
            if self.physics_v332
            else EvidenceClosureGraph(self.texts)
            if self.physics_v331
            else FactSlingshotGraph(self.texts)
            if self.physics_v330
            else None
        )
        self._document_star_field = (
            DocumentStarField(self.texts) if self.physics_v333 else None
        )
        self.last_atomic_capsules: list[tuple[str, ...]] = []
        self.last_source_atomic_capsules: list[tuple[str, ...]] = []
        self.last_state_sector_capsules: list[tuple[str, ...]] = []
        self.last_ledger_diagnostics: dict[str, Any] = {}
        self.last_slingshot_diagnostics: dict[str, Any] = {}
        # RealMem issues several questions at the same chronological prefix.
        # Keep exactly one derived prefix index: reuse within the session, then
        # replace it as time advances. This bounds RAM while avoiding repeated
        # BM25/capsule construction for identical visible history.
        self._prefix_cache_size: int | None = None
        self._prefix_cache: tuple[Any, ...] | None = None

    def retrieve(
        self, query: str, *, prefix: int | None = None, satellites: int = 16,
        late_scores: Sequence[float] | None = None,
    ) -> list[tuple[str, ...]]:
        self.last_atomic_capsules = []
        self.last_source_atomic_capsules = []
        self.last_state_sector_capsules = []
        self.last_ledger_diagnostics = {}
        self.last_slingshot_diagnostics = {}
        size = len(self.texts) if prefix is None else min(max(0, prefix), len(self.texts))
        if size <= 0:
            return []
        slingshot = (
            self._slingshot_graph.solve(query, size=size)
            if self._slingshot_graph is not None else SlingshotResult()
        )
        document_star = (
            self._document_star_field.solve(query, size=size)
            if self._document_star_field is not None else DocumentStarResult()
        )
        if slingshot.primary_sources:
            self.last_slingshot_diagnostics = {
                "answer": slingshot.answer,
                "primary_sources": list(slingshot.primary_sources),
                "echo_sources": len(slingshot.echo_sources),
                "shadow_sources": len(getattr(slingshot, "shadow_sources", ())),
                "dependency_sources": len(getattr(slingshot, "dependency_sources", ())),
                "path": [list(step) for step in slingshot.path],
                "terminal_relations": list(getattr(slingshot, "terminal_relations", ())),
            }
        if document_star.top_sources:
            self.last_slingshot_diagnostics["document_star"] = {
                "documents": document_star.documents,
                "top_sources": len(document_star.top_sources),
                "orbit_sources": len(document_star.orbit_sources),
                "score_ratio": document_star.score_ratio,
                "score_margin": document_star.score_margin,
            }
        if self.physics_v327 and self._census_ledger is not None and is_census_query(query):
            as_of = self.dates[size - 1] if size <= len(self.dates) and size else None
            ledger_sources = self._census_ledger.select(query, prefix=size, as_of=as_of)
            if ledger_sources:
                seen = set(ledger_sources)
                fallback = [
                    source for source in range(size - 1, -1, -1)
                    if source not in seen
                ][:48]
                selected = list(dict.fromkeys(ledger_sources + fallback))
                # A strict atom is exactly one immutable source. Keep this
                # independent of both the state sector and the ten-slot stripe
                # so R@1 cannot be inflated by changing capsule width.
                self.last_source_atomic_capsules = [
                    (self.source_ids[source],) for source in selected
                ]
                self.last_state_sector_capsules = [
                    tuple(self.source_ids[source] for source in ledger_sources),
                    *((self.source_ids[source],) for source in fallback),
                ]
                striped: list[list[int]] = [[] for _ in range(min(10, max(1, len(selected))))]
                for offset, source in enumerate(selected):
                    striped[offset % len(striped)].append(source)
                atomic = [tuple(capsule) for capsule in striped if capsule]
                self.last_atomic_capsules = [
                    tuple(self.source_ids[source] for source in capsule) for capsule in atomic
                ]
                self.last_ledger_diagnostics = {
                    "matched_categories": sorted(self._census_ledger.query_categories(query)),
                    "ledger_sources": len(ledger_sources),
                    "source_budget": len(selected),
                    "fast_path": True,
                }
                collapsed = collapse_census_singularity(query, atomic)
                return [
                    tuple(self.source_ids[source] for source in capsule)
                    for capsule in collapsed
                ]
        texts = self.texts[:size]
        if size == len(self.texts):
            groups = self._full_groups
            session_texts = self._full_session_texts
            dates = self._full_dates
            lexical_index = self._full_lexical_index
            char_index = self._full_char_index
            planet_index = self._full_planet_index
            capsules, by_source = self._full_capsules, self._full_by_source
            directive_meta = self._full_directive_meta
        else:
            raw_groups = self.groups[:size]
            unique_planets = len(set(self.groups)) == len(self.groups)
            if unique_planets:
                groups = self._full_groups[:size]
                session_texts = self._full_session_texts[:size]
                dates = self._full_dates[:size]
                lexical_index = _PrefixBM25View(self._full_lexical_index, size)
                char_index = _PrefixBM25View(self._full_char_index, size)
                planet_index = _PrefixBM25View(self._full_planet_index, size)
                capsules, by_source = _benchmark_capsules(groups)
                directive_meta = self._full_directive_meta[:size]
            elif self._prefix_cache_size == size and self._prefix_cache is not None:
                (
                    groups, session_texts, dates, lexical_index, char_index,
                    planet_index, capsules, by_source, directive_meta,
                ) = self._prefix_cache
            else:
                group_map = {group: index for index, group in enumerate(dict.fromkeys(raw_groups))}
                groups = [group_map[group] for group in raw_groups]
                group_members: dict[int, list[int]] = defaultdict(list)
                for source, group in enumerate(groups):
                    group_members[group].append(source)
                session_texts = [
                    " ".join(texts[source] for source in group_members[group])
                    for group in range(len(group_members))
                ]
                dates = [
                    self.dates[original] if original < len(self.dates) else ""
                    for original in group_map
                ]
                lexical_index = BM25(texts)
                char_index = BM25(texts, tokenizer=char_tokenize)
                planet_index = BM25(session_texts)
                capsules, by_source = _benchmark_capsules(groups)
                directive_meta = [_directive_metadata(text, trusted=True) for text in texts]
                self._prefix_cache_size = size
                self._prefix_cache = (
                    groups, session_texts, dates, lexical_index, char_index,
                    planet_index, capsules, by_source, directive_meta,
                )

        # Preserve the frozen single-query feature distribution consumed by the
        # learned ranker; query-superposition max pooling was rejected because
        # it changed field calibration without improving development recall.
        lexical_raw = lexical_index.scores(query)
        char_raw = char_index.scores(query)
        planet_raw = planet_index.scores(query)
        lexical_order = ranking(lexical_raw)
        planet_order = ranking(planet_raw)
        lexical_rr = reciprocal_ranks(lexical_order)
        planet_rr = reciprocal_ranks(planet_order)
        lexical_bridge = (
            _bounded_lexical_bridge(query, texts, lexical_index, lexical_order)
            if (
                self.physics_v332
                and self.experimental_lexical_bridge_v332
                and not slingshot.primary_sources
            )
            else [0.0] * size
        )
        seed = ranking([
            lexical_rr[index] + 0.72 * planet_rr[groups[index]]
            + 0.44 * lexical_bridge[index]
            for index in range(size)
        ])
        launch_sources = list(dict.fromkeys(
            source for source in slingshot.primary_sources if source < size
        ))
        if launch_sources:
            launch_set = set(launch_sources)
            seed = launch_sources + [source for source in seed if source not in launch_set]
        orbit = _orbit_expand(seed, groups)
        qv = self.encoder.encode_many([query])[0]
        if self.neural_candidate_pool:
            candidates = seed[:min(size, self.neural_candidate_pool)]
            missing = [source for source in candidates if source not in self._lazy_vectors]
            if missing:
                encoded = self.encoder.encode_many([texts[source] for source in missing])
                self._lazy_vectors.update(zip(missing, encoded))
            semantic = [0.0] * size
            for source in candidates:
                semantic[source] = max(0.0, float(np.dot(qv, self._lazy_vectors[source])))
        else:
            semantic = _dot_scores(qv, self.vectors[:size])
        lexical = unit_scores(lexical_raw)
        planetary_units = unit_scores(planet_raw)
        planetary = [planetary_units[groups[index]] for index in range(size)]
        char = unit_scores(char_raw)
        entity = _entity_value_resonance(query, texts)
        temporal_groups = _temporal_phase(query, dates)
        temporal = [temporal_groups[groups[index]] for index in range(size)]
        graph = _graph_flux(lexical, groups)
        # UMD 3.16 may widen the neural field, but its conservation orbit must
        # reconstruct the frozen 64-candidate UMD 3.15 field exactly. Sources
        # beyond that boundary participate only in the additive coverage field.
        control_semantic = semantic
        if self.physics_v316 and self.neural_candidate_pool > 64:
            control_sources = set(seed[:min(size, 64)])
            control_semantic = [
                semantic[index] if index in control_sources else 0.0
                for index in range(size)
            ]
        force = _force(control_semantic, lexical, planetary, char, entity, temporal, graph)
        focus_force = _force(semantic, lexical, planetary, char, entity, temporal, graph)
        if self.physics_v332 and any(lexical_bridge):
            force = [
                0.82 * value + 0.18 * lexical_bridge[index]
                for index, value in enumerate(force)
            ]
            focus_force = [
                0.82 * value + 0.18 * lexical_bridge[index]
                for index, value in enumerate(focus_force)
            ]
        if self.physics_v316:
            focus_force = apply_state_field(force, state_mass_adjustment(query, texts))
        if late_scores is not None:
            if len(late_scores) < size:
                raise ValueError("late_scores must cover every visible source")
            late = unit_scores([float(value) for value in late_scores[:size]])
            adjacent_late: list[float] = []
            for index, value in enumerate(late):
                neighbors = [value]
                if index and groups[index - 1] == groups[index]:
                    neighbors.append(late[index - 1])
                if index + 1 < size and groups[index + 1] == groups[index]:
                    neighbors.append(late[index + 1])
                adjacent_late.append(max(neighbors))
            focus_force = [
                0.55 * late[index] + 0.35 * focus_force[index] + 0.10 * adjacent_late[index]
                for index in range(size)
            ]
        protected = list(dict.fromkeys(launch_sources + orbit[:10]))[:10]
        protected_set = set(protected)
        launch_set = set(launch_sources)
        order = launch_sources + sorted(
            (index for index in protected if index not in launch_set),
            key=lambda index: (-force[index], index),
        )
        order.extend(index for index in orbit if index not in protected_set)
        stable = _anchored_capsule_order(
            query, order, force, capsules, by_source, top_k=10,
        )
        directive_values = directive_gravity_values(query, texts, directive_meta)
        order314, force314 = relative_gravity_order(
            query, texts, groups, dates, order, force, semantic, lexical,
            directive_meta, top_k=10, directive_values=directive_values,
        )
        bridge = transient_directive_bridge(
            query, texts, order314, directive_meta,
            directive_values=directive_values,
        )
        stable314 = fuse_transient_bridge(stable, bridge)
        if self.physics_v316:
            if self.first_orbit_v316:
                stable314 = first_capsule_lagrange_order(
                    query, stable314, texts, focus_force, semantic, lexical,
                    entity, temporal,
                )
            if self.physics_v317 and not self.physics_v325:
                stable314 = first_capsule_focus_tether(
                    query, stable314, self._fact_planets,
                    focus_force, semantic, lexical,
                )
            old_secondary = rank_event_satellites(
                order314, stable314, groups, force314, control_semantic, lexical,
            )
            if self.physics_v317:
                legacy_coverage = rank_constellation_coverage(
                    query, order314, stable314, texts, groups, dates,
                    focus_force, semantic, lexical,
                    source_token_sets=self._census_tokens or None,
                )
                if self.physics_v325:
                    if self.physics_v327 and self._census_ledger is not None:
                        ledger_sources = self._census_ledger.select(
                            query, prefix=size,
                            as_of=(self.dates[size - 1] if size <= len(self.dates) and size else None),
                        )
                        fallback = list(dict.fromkeys(old_secondary + legacy_coverage))
                        secondary = ledger_census_orbit(query, ledger_sources, fallback)
                        satellites = ledger_census_budget(
                            query, ledger_sources, size, fallback_budget=48,
                        )
                        self.last_ledger_diagnostics = {
                            "matched_categories": sorted(self._census_ledger.query_categories(query)),
                            "ledger_sources": len(ledger_sources),
                            "source_budget": satellites,
                        }
                    else:
                        emergent_tokens = (
                            self._domain_graph.expand(query) if self._domain_graph is not None else set()
                        )
                        census = rank_galactic_census(
                            query, order314, stable314, texts, self._census_tokens,
                            focus_force, semantic, lexical,
                            extra_query_tokens=emergent_tokens,
                            broad_override=self.physics_v326 and is_census_query(query),
                        )
                        # A census question is coverage-first; the 3.16 coverage
                        # orbit remains as a conserved audit tail. Avoid compiling
                        # the much larger clause-level 3.17 fact graph here.
                        secondary = list(dict.fromkeys(census + old_secondary + legacy_coverage))
                        satellites = (
                            singularity_budget(query, size, default=satellites)
                            if self.physics_v326
                            else galactic_census_budget(query, size, default=satellites)
                        )
                else:
                    fact_coverage = rank_fact_planet_coverage(
                        query, order314, stable314, self._fact_planets,
                        groups, focus_force, semantic, lexical,
                    )
                    secondary = conserved_fact_constellation(
                        old_secondary, legacy_coverage, fact_coverage,
                        conserved=satellites,
                    )
                    satellites = fact_constellation_budget(
                        query, default=satellites, ceiling=max(48, satellites),
                    )
            else:
                coverage_secondary = rank_constellation_coverage(
                    query, order314, stable314, texts, groups, dates,
                    focus_force, semantic, lexical,
                    source_token_sets=self._census_tokens or None,
                )
                secondary = conserved_constellation_orbit(
                    old_secondary, coverage_secondary, conserved=satellites,
                )
                satellites = constellation_budget(
                    query, default=satellites, ceiling=max(48, satellites),
                )
        else:
            secondary = rank_event_satellites(
                order314, stable314, groups, force314, semantic, lexical,
            )
        if self.physics_v333 and document_star.orbit_sources:
            star_set = set(document_star.orbit_sources)
            secondary = list(document_star.orbit_sources) + [
                source for source in secondary if source not in star_set
            ]
            satellites = max(
                satellites, min(48, len(document_star.orbit_sources)),
            )
        closure_sources: list[int] = []
        if slingshot.echo_sources:
            closure_sources.extend(
                source for source in slingshot.echo_sources if source < size
            )
        if self.physics_v331:
            if self.physics_v332:
                closure_sources.extend(
                    source
                    for source in getattr(slingshot, "shadow_sources", ())
                    if source < size
                )
            closure_sources.extend(
                source
                for source in getattr(slingshot, "dependency_sources", ())
                if source < size
            )
        if closure_sources:
            closure_sources = list(dict.fromkeys(closure_sources))
            closure_set = set(closure_sources)
            secondary = closure_sources + [
                source for source in secondary if source not in closure_set
            ]
            closure_ceiling = 256 if self.physics_v332 else 96
            satellites = max(satellites, min(closure_ceiling, len(closure_sources)))
        result = compress_event_horizon(
            stable314, secondary, max_extra_sources=satellites,
        )
        if self.physics_v318:
            cross = None
            if self.cross_encoder is not None:
                # Cross interaction remains separately bounded because it is
                # expensive. The lightweight learned Roche field below may
                # inspect the complete source horizon.
                cross_sources = list(dict.fromkeys(order314[:self.cross_limit]))
                cross_passages = [texts[source] for source in cross_sources]
                cross_values = (
                    self.cross_encoder.predict_multiwindow(query, cross_passages)
                    if self.physics_v322 else self.cross_encoder.predict(query, cross_passages)
                )
                spread = max(cross_values) - min(cross_values) if cross_values else 1.0
                floor = min(cross_values) - max(1.0, spread) if cross_values else -1.0
                cross = [floor] * size
                for source, value in zip(cross_sources, cross_values):
                    cross[source] = value
            result = roche_lobe_first_capsule(
                query, result, order314, self._fact_planets,
                focus_force, semantic, lexical, char, entity, temporal,
                groups,
                cross,
                self.roche_ranker,
                self.last_roche_diagnostics,
                source_budget=(
                    escape_velocity_budget(query, self.roche_budget_v318)
                    if self.physics_v324 else self.roche_budget_v318
                ),
                matter_v320=self.physics_v320,
                matter_v321=self.physics_v321,
                matter_v322=self.physics_v322,
                matter_v323=self.physics_v323,
                neighbor_budget=self.matter_neighbor_budget,
                source_texts=texts,
                source_dates=[dates[groups[source]] if groups[source] < len(dates) else "" for source in range(size)],
            )
        if self.physics_v333 and document_star.top_sources and result:
            # The original first capsule is conserved.  Document-star moons
            # are appended inside the same Final rank, so this cannot remove a
            # prior top-one hit.  Capacity changes remain explicit in reports.
            result[0] = tuple(dict.fromkeys(
                result[0] + tuple(document_star.top_sources)
            ))
        self.last_atomic_capsules = [
            tuple(self.source_ids[source] for source in capsule) for capsule in result
        ]
        ordered_sources = list(dict.fromkeys(
            source for capsule in result for source in capsule
        ))
        if self.physics_v332 and not slingshot.primary_sources:
            # Read the final capsule matrix by orbital columns.  The strongest
            # source from each top capsule reaches the strict horizon before a
            # second source from the same capsule, preventing an early wide
            # episode from consuming most of Strict@10.
            ranked_capsules = [
                sorted(capsule, key=lambda source: (-force314[source], source))
                for capsule in result
            ]
            ordered_sources = list(dict.fromkeys(
                source
                for depth in range(max(map(len, ranked_capsules), default=0))
                for capsule in ranked_capsules
                if depth < len(capsule)
                for source in (capsule[depth],)
            ))
        if (
            self.physics_v333
            and not slingshot.primary_sources
            and document_star.best_sources
        ):
            # Promote a document-local atomic moon only when the complete
            # document has a stable lead over the runner-up.  Otherwise keep
            # the conserved 3.32.1 first source.  Strict remains one source per
            # rank regardless of this choice.
            star_first = document_star.best_sources[0]
            if document_star.score_ratio >= 1.13:
                ordered_sources = [star_first] + [
                    source for source in ordered_sources if source != star_first
                ]
        if self.physics_v331 and slingshot.primary_sources:
            # Strict retrieval remains genuinely source-atomic.  The closure
            # changes only order: answer-bearing echoes precede supporting
            # dependencies, followed by the conserved 3.30 orbit.  This makes
            # complete recall possible without widening a rank position.
            ordered_sources = list(dict.fromkeys(
                [
                    source for source in slingshot.primary_sources
                    if source < size
                ]
                + [
                    source for source in slingshot.echo_sources
                    if source < size
                ]
                + [
                    source
                    for source in getattr(slingshot, "shadow_sources", ())
                    if self.physics_v332 and source < size
                ]
                + [
                    source
                    for source in getattr(slingshot, "dependency_sources", ())
                    if source < size
                ]
                + ordered_sources
            ))
        self.last_source_atomic_capsules = [
            (self.source_ids[source],) for source in ordered_sources
        ]
        self.last_state_sector_capsules = list(self.last_atomic_capsules)
        if self.physics_v326:
            result = collapse_census_singularity(query, result)
        return [
            tuple(self.source_ids[source] for source in capsule)
            for capsule in result
        ]

    def retrieve_channels(
        self, query: str, *, prefix: int | None = None, satellites: int = 16,
    ) -> dict[str, list[tuple[str, ...]]]:
        """Return provenance-complete audit and state-gated answer orbits."""
        capsules = self.retrieve(query, prefix=prefix, satellites=satellites)
        size = len(self.texts) if prefix is None else min(max(0, prefix), len(self.texts))
        return self._split_channels(query, capsules, size)

    def _split_channels(
        self, query: str, capsules: Sequence[tuple[str, ...]], size: int,
    ) -> dict[str, list[tuple[str, ...]]]:
        if self.physics_v325:
            return split_antimatter_orbits(
                query, capsules, self.source_ids[:size], self.texts[:size],
                negative_ids=self._antimatter_source_ids,
            )
        return split_state_orbits(
            query, capsules, self.source_ids[:size], self.texts[:size],
        )

    def retrieve_compact_channels(
        self, query: str, *, prefix: int | None = None, satellites: int = 16,
    ) -> dict[str, Any]:
        """Return answer/audit orbits plus an answer-ready census payload."""
        channels = self.retrieve_channels(query, prefix=prefix, satellites=satellites)
        answer = channels["answer"]
        provenance = answer[0] if answer and is_census_query(query) else ()
        payload = (
            compact_census_payload(query, provenance, self._versioned_facts)
            if provenance and self._versioned_facts else None
        )
        size = len(self.texts) if prefix is None else min(max(0, prefix), len(self.texts))
        atomic_channels = self._split_channels(query, self.last_atomic_capsules, size)
        source_atomic_channels = self._split_channels(
            query, self.last_source_atomic_capsules, size,
        )
        state_sector_channels = self._split_channels(
            query, self.last_state_sector_capsules, size,
        )
        provenance_set = set(provenance)
        fact_provenance = tuple(
            fact.fact_id for fact in self._versioned_facts
            if fact.state == "current" and fact.source_id in provenance_set
        )
        solved = solve_compact_numeric(query, payload) if payload is not None else None
        return {
            "answer": answer, "audit": channels["audit"],
            "atomic_answer": source_atomic_channels["answer"],
            "legacy_striped_answer": atomic_channels["answer"],
            "state_sector_answer": state_sector_channels["answer"],
            "payload": payload, "fact_provenance": fact_provenance,
            "deterministic_answer": solved, "ledger": dict(self.last_ledger_diagnostics),
            "slingshot": dict(self.last_slingshot_diagnostics),
        }


def _expanded(capsules: Sequence[Sequence[str]], k: int = 10) -> set[str]:
    return {source for capsule in capsules[:k] for source in capsule}


def _evidence_session_ids(value: Any) -> set[str]:
    output: set[str] = set()
    if isinstance(value, dict):
        if "session_id" in value:
            output.add(str(value["session_id"]))
        for child in value.values():
            output.update(_evidence_session_ids(child))
    elif isinstance(value, list):
        for child in value:
            output.update(_evidence_session_ids(child))
    return output


def _evidence_values(value: Any) -> set[str]:
    """Extract evaluator-labelled fact values without using them for ranking."""
    output: set[str] = set()
    if isinstance(value, dict):
        scalar = value.get("value")
        if isinstance(scalar, (str, int, float)):
            output.add(str(scalar).strip().casefold())
        for child in value.values():
            output.update(_evidence_values(child))
    elif isinstance(value, list):
        for child in value:
            output.update(_evidence_values(child))
    return {item for item in output if item}


def _evidence_fact_pairs(value: Any) -> set[tuple[str, str]]:
    """Return provenance-aligned ``(source, value)`` evaluator facts."""
    output: set[tuple[str, str]] = set()
    if isinstance(value, dict):
        if "session_id" in value and isinstance(value.get("value"), (str, int, float)):
            fact = str(value["value"]).strip().casefold()
            if fact:
                output.add((str(value["session_id"]), fact))
        for child in value.values():
            output.update(_evidence_fact_pairs(child))
    elif isinstance(value, list):
        for child in value:
            output.update(_evidence_fact_pairs(child))
    return output


def _find_evidence_scalar(value: Any, keys: Sequence[str]) -> float | None:
    if isinstance(value, dict):
        for key in keys:
            scalar = value.get(key)
            if isinstance(scalar, (int, float)) and not isinstance(scalar, bool):
                return float(scalar)
        for child in value.values():
            found = _find_evidence_scalar(child, keys)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_evidence_scalar(child, keys)
            if found is not None:
                return found
    return None


def _message_text(message: Any) -> str:
    if isinstance(message, str):
        return message
    if not isinstance(message, dict):
        return str(message)
    parts = [
        message.get("user_message", message.get("user", "")),
        message.get("assistant_message", message.get("assistant", "")),
        message.get("message", ""), message.get("content", ""),
        message.get("time", ""), message.get("place", ""),
    ]
    return " ".join(str(part) for part in parts if part)


def _flatten_memora_metadata(value: Any, prefix: str = "") -> list[str]:
    """Flatten query-independent session state into searchable plain text."""
    output: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            field = str(key).replace("_", " ").replace("-", " ")
            output.extend(_flatten_memora_metadata(child, f"{prefix} {field}".strip()))
    elif isinstance(value, list):
        for child in value:
            output.extend(_flatten_memora_metadata(child, prefix))
    elif value is not None and value != "":
        output.append(f"{prefix} {value}".strip())
    return output


def _memora_session_text(item: dict[str, Any]) -> str:
    """Preserve state while dropping synthetic conversational boilerplate.

    Memora explicitly marks the turn that should enter memory. Indexing every
    greeting and generic assistant response inflated quarterly BM25 state by
    more than an order of magnitude and introduced retrieval noise.
    """
    operation = str(item.get("operation", "")).casefold()
    details = item.get("operation_details")
    if isinstance(details, dict) and operation == "update":
        # The immutable JSON session remains the transaction/audit ledger.
        # The answer projection carries only the post-update state, preventing
        # old_item and old_preference from leaking back into recommendations.
        details = {
            key: value for key, value in details.items()
            if not str(key).casefold().startswith("old_")
        }
    metadata = {
        "session_type": item.get("session_type"),
        "operation": item.get("operation"),
        "operation_details": details,
        "date": item.get("date"),
        "persona": item.get("persona"),
    }
    shared_messages = [
        str(turn.get("message", "")) for turn in item.get("conversation", [])
        if turn.get("share_memory") is True
    ]
    conversation = " ".join(
        f"{turn.get('speaker', '')}: {turn.get('message', '')}"
        for turn in item.get("conversation", [])
        if turn.get("share_memory") is True and operation != "update"
    )
    domain_tags: list[str] = []
    if operation == "update":
        shared = " ".join(shared_messages).casefold()
        domain_patterns = {
            "movie": r"\b(?:movie|film|actor|actress|director|watch(?:ed)?)s?\b",
            "book": r"\b(?:book|author|novel|read(?:ing)?)s?\b",
            "music": r"\b(?:music|song|album|artist|listen(?:ed|ing)?)s?\b",
            "travel": r"\b(?:travel|destination|visit(?:ed|ing)?|climate|site)s?\b",
        }
        domain_tags = [
            f"memory domain {domain}" for domain, pattern in domain_patterns.items()
            if re.search(pattern, shared, re.IGNORECASE)
        ]
    return " ".join(
        _flatten_memora_metadata(metadata) + domain_tags
        + ([conversation] if conversation else [])
    )


def _flatten_membench_messages(message_list: Any) -> tuple[list[str], list[int], list[str]]:
    texts: list[str] = []
    groups: list[int] = []
    ids: list[str] = []
    sessions = message_list if isinstance(message_list, list) else []
    if sessions and not isinstance(sessions[0], list):
        sessions = [sessions]
    for group, session in enumerate(sessions):
        for fallback, message in enumerate(session):
            texts.append(_message_text(message))
            groups.append(group)
            ids.append(str(message.get("sid", len(ids))) if isinstance(message, dict) else str(len(ids)))
    return texts, groups, ids


def _choose_persona_option(
    query: str,
    options: Sequence[str],
    capsules: Sequence[Sequence[str]],
    index: OrbitIndex,
) -> int:
    """Deterministic answer proxy; never reads the reference option."""
    option_vectors = index.encoder.encode_many(list(options))
    query_vector = index.encoder.encode_many([query])[0]
    source_numbers = [
        int(source) for source in _expanded(capsules)
        if str(source).isdigit() and int(source) < len(index.texts)
    ]
    scores: list[float] = []
    for option, vector in zip(options, option_vectors):
        semantic_memory = max(
            (max(0.0, float(np.dot(vector, index.vectors[source]))) for source in source_numbers),
            default=0.0,
        )
        option_tokens = set(tokenize(option))
        lexical_memory = max((
            len(option_tokens & set(tokenize(index.texts[source])))
            / max(1, len(option_tokens))
            for source in source_numbers
        ), default=0.0)
        query_fit = max(0.0, float(np.dot(vector, query_vector)))
        scores.append(0.58 * semantic_memory + 0.27 * lexical_memory + 0.15 * query_fit)
    return max(range(len(scores)), key=lambda number: (scores[number], -number))


def run_personamem(encoder: FastEmbedEncoder) -> dict[str, Any]:
    started = time.perf_counter()
    root = VENDOR / "PersonaMemData"
    contexts: dict[str, list[dict[str, Any]]] = {}
    with (root / "shared_contexts_32k.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            contexts.update(json.loads(line))
    indexes: dict[str, OrbitIndex] = {}
    for context_id, messages in contexts.items():
        texts = [f"{message.get('role', '')}: {message.get('content', '')}" for message in messages]
        indexes[context_id] = OrbitIndex(
            texts, [number // 2 for number in range(len(texts))],
            [str(number) for number in range(len(texts))], encoder,
        )
    correct = total = 0
    by_type: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    retrieved_characters: list[int] = []
    with (root / "questions_32k.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            context_id = row["shared_context_id"]
            index = indexes[context_id]
            prefix = int(row["end_index_in_shared_context"])
            query = row["user_question_or_message"]
            options = list(ast.literal_eval(row["all_options"]))
            capsules = index.retrieve(query, prefix=prefix)
            prediction = _choose_persona_option(query, options, capsules, index)
            reference = ord(row["correct_answer"].strip().lower()[1]) - ord("a")
            hit = int(prediction == reference)
            correct += hit
            total += 1
            by_type[row["question_type"]][0] += hit
            by_type[row["question_type"]][1] += 1
            retrieved_characters.append(sum(
                len(index.texts[int(source)]) for source in _expanded(capsules)
                if str(source).isdigit() and int(source) < prefix
            ))
    return {
        "benchmark": "PersonaMem-32K",
        "status": "full_deterministic_answer_proxy",
        "questions": total,
        "multiple_choice_accuracy": correct / max(1, total),
        "random_baseline": 0.25,
        "by_question_type": {
            key: {"correct": value[0], "questions": value[1], "accuracy": value[0] / value[1]}
            for key, value in sorted(by_type.items())
        },
        "mean_retrieved_characters": statistics.mean(retrieved_characters),
        "official_llm_accuracy_run": False,
        "runtime_seconds": time.perf_counter() - started,
    }


def run_membench(encoder: FastEmbedEncoder, per_group: int = 10) -> dict[str, Any]:
    started = time.perf_counter()
    root = VENDOR / "MemBenchOfficial" / "MemData"
    metric = CapsuleRecallAccumulator((1, 5, 10))
    by_file: dict[str, CapsuleRecallAccumulator] = {}
    trajectories = 0
    for path in sorted(root.rglob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        file_metric = CapsuleRecallAccumulator((1, 5, 10))
        for group_name, rows in data.items():
            if not isinstance(rows, list):
                continue
            for trajectory in rows[:per_group]:
                texts, groups, ids = _flatten_membench_messages(trajectory.get("message_list", []))
                if not texts:
                    continue
                qa = trajectory.get("QA", {})
                gold = {
                    str(value[0] if isinstance(value, list) else value)
                    for value in qa.get("target_step_id", [])
                }
                if not gold:
                    continue
                capsules = OrbitIndex(texts, groups, ids, encoder).retrieve(str(qa.get("question", "")))
                metric.add(capsules, gold)
                file_metric.add(capsules, gold)
                trajectories += 1
        by_file[f"{path.parent.name}/{path.stem}"] = file_metric
    return {
        "benchmark": "MemBench",
        "status": "stratified_retrieval_proxy",
        "sampling": f"first {per_group} trajectories per top-level group in every official JSON file",
        "trajectories": trajectories,
        "metrics": metric.result(),
        "by_file": {key: value.result() for key, value in by_file.items()},
        "official_end_to_end_accuracy_run": False,
        "runtime_seconds": time.perf_counter() - started,
    }


def _aggregate_capsule_results(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    queries = sum(result["evaluated_queries"] for result in results)
    gold_items = sum(result["gold_evidence_items"] for result in results)
    ks = ("1", "5", "10")
    return {
        "evaluated_queries": queries,
        "gold_evidence_items": gold_items,
        "retrieval_unit": "evidence_capsule",
        "mrr": sum(result["mrr"] * result["evaluated_queries"] for result in results) / max(1, queries),
        "any_evidence_recall": {
            k: sum(result["any_evidence_recall"][k] * result["evaluated_queries"] for result in results) / max(1, queries)
            for k in ks
        },
        "full_evidence_recall": {
            k: sum(result["full_evidence_recall"][k] * result["evaluated_queries"] for result in results) / max(1, queries)
            for k in ks
        },
        "micro_evidence_recall": {
            k: sum(result["micro_evidence_recall"][k] * result["gold_evidence_items"] for result in results) / max(1, gold_items)
            for k in ks
        },
    }


def run_memora(
    encoder: FastEmbedEncoder, personas_per_period: int = 1, *, physics_v316: bool = False,
    physics_v317: bool = False,
    physics_v325: bool = False,
    physics_v326: bool = False,
    physics_v327: bool = False,
    persona_start_per_period: int = 0,
    checkpoint_name: str | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    root = VENDOR / "Memora" / "data"
    active_physics = physics_v316 or physics_v317 or physics_v325 or physics_v326 or physics_v327
    checkpoint_path = RESULTS / (checkpoint_name or (
        "memora_umd3282_v1_checkpoint.json" if physics_v327
        else "memora_umd326_v4_checkpoint.json" if physics_v326
        else "memora_umd325_v5_checkpoint.json" if physics_v325
        else "memora_umd317_structured_checkpoint.json" if physics_v317
        else "memora_umd316_v5_checkpoint.json" if physics_v316
        else "memora_retrieval_checkpoint.json"
    ))
    checkpoint: dict[str, Any] = {}
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    all_eval_paths = sorted(root.glob("*/*/evaluation_questions_*.json"))
    eval_paths: list[Path] = []
    for period in sorted({path.parent.parent.name for path in all_eval_paths}):
        period_paths = [
            path for path in all_eval_paths if path.parent.parent.name == period
        ]
        start = max(0, persona_start_per_period)
        eval_paths.extend(period_paths[start:start + max(1, personas_per_period)])
    for position, eval_path in enumerate(eval_paths, 1):
        persona_dir = eval_path.parent
        period = persona_dir.parent.name
        checkpoint_key = f"{period}/{persona_dir.name}"
        if checkpoint_key in checkpoint:
            print(f"Memora checkpoint: {position}/{len(eval_paths)} {checkpoint_key}", flush=True)
            continue
        sessions: list[str] = []
        structured_records: list[dict[str, Any]] = []
        dates: list[str] = []
        ids: list[str] = []
        for session_path in sorted((persona_dir / "conversations").glob("session_*.json")):
            item = json.loads(session_path.read_text(encoding="utf-8"))
            structured_records.append(item)
            sessions.append(_memora_session_text(item))
            dates.append(str(item.get("date", "")))
            ids.append(str(item.get("session_id")))
        index = OrbitIndex(
            sessions, list(range(len(sessions))), ids, encoder, dates,
            neural_candidate_pool=64 if (physics_v325 or physics_v326 or physics_v327) else 192 if active_physics else 64,
            physics_v316=active_physics,
            physics_v317=physics_v317 or physics_v325 or physics_v326 or physics_v327,
            physics_v325=physics_v325 or physics_v326 or physics_v327,
            physics_v326=physics_v326 or physics_v327,
            physics_v327=physics_v327,
            first_orbit_v316=False,
            structured_records=structured_records,
        )
        control_index = (
            OrbitIndex(
                sessions, list(range(len(sessions))), ids, encoder, dates,
                neural_candidate_pool=192 if physics_v317 else 64,
                physics_v316=physics_v317 or physics_v325,
                physics_v317=physics_v317 or physics_v325,
                first_orbit_v316=False,
            )
            if active_physics and not (physics_v325 or physics_v326 or physics_v327) else None
        )
        evaluation = json.loads(eval_path.read_text(encoding="utf-8"))
        answer_text_by_id = dict(zip(ids, sessions))
        answer_fact_values_by_id: dict[str, set[str]] = defaultdict(set)
        if physics_v326 or physics_v327:
            for fact in index._versioned_facts:
                if fact.state == "current":
                    answer_fact_values_by_id[fact.source_id].add(
                        str(fact.value).strip().casefold()
                    )
        persona_metric = CapsuleRecallAccumulator((1, 5, 10))
        atomic_metric = CapsuleRecallAccumulator((1, 5, 10))
        striped_metric = CapsuleRecallAccumulator((1, 5, 10))
        state_sector_metric = CapsuleRecallAccumulator((1, 5, 10))
        control_metric = CapsuleRecallAccumulator((1, 5, 10))
        forgetting_queries = forgetting_contaminated = questions = 0
        single_gold_queries = 0
        forgetting_value_contaminated = forgetting_overlap_floor = 0
        compact_queries = compact_payload_chars = compact_raw_chars = 0
        ledger_queries = ledger_sources = ledger_budgets = fact_provenance_items = 0
        ledger_fast_path_queries = 0
        deterministic_queries = deterministic_correct = 0
        sector_eval_queries = sector_sources = sector_gold_hits = sector_exact_queries = 0
        deterministic_failures: list[dict[str, Any]] = []
        forgetting_value_failures: list[dict[str, Any]] = []
        control_forgetting_contaminated = 0
        for rows in evaluation.get("questions", {}).values():
            for row in rows:
                question = str(row.get("question", ""))
                question_date = str(row.get("question_date", ""))
                visible_prefix = (
                    sum(1 for date in dates if not date or date <= question_date)
                    if question_date else len(dates)
                )
                if physics_v326 or physics_v327:
                    compact = index.retrieve_compact_channels(question, prefix=visible_prefix)
                    capsules = compact["answer"]
                    atomic_capsules = compact.get("atomic_answer", [])
                    striped_capsules = compact.get("legacy_striped_answer", [])
                    state_sector_capsules = compact.get("state_sector_answer", [])
                    payload = compact.get("payload")
                    if payload is not None:
                        compact_queries += 1
                        compact_payload_chars += int(payload.get("serialized_chars", 0))
                        compact_raw_chars += sum(
                            len(answer_text_by_id.get(source, ""))
                            for source in (capsules[0] if capsules else ())
                        )
                    ledger = compact.get("ledger") or {}
                    if ledger.get("ledger_sources") is not None:
                        ledger_queries += 1
                        ledger_sources += int(ledger.get("ledger_sources", 0))
                        ledger_budgets += int(ledger.get("source_budget", 0))
                        ledger_fast_path_queries += bool(ledger.get("fast_path"))
                    fact_provenance_items += len(compact.get("fact_provenance", ()))
                    solved = compact.get("deterministic_answer")
                    if solved is not None and isinstance(solved.get("value"), (int, float)):
                        if "step" in question.casefold():
                            expected = _find_evidence_scalar(
                                row.get("memory_evidence"), ("max_value", "total_steps"),
                            )
                        elif any(word in question.casefold() for word in ("coffee", "lunch", "breakfast", "dinner", "grocery")):
                            expected = _find_evidence_scalar(row.get("memory_evidence"), ("category_total",))
                        else:
                            expected = _find_evidence_scalar(row.get("memory_evidence"), ("total_amount",))
                        if expected is not None:
                            deterministic_queries += 1
                            correct = abs(float(solved["value"]) - expected) <= 1e-5
                            deterministic_correct += correct
                            if not correct:
                                deterministic_failures.append({
                                    "question": question, "expected": expected,
                                    "actual": float(solved["value"]),
                                })
                else:
                    atomic_capsules = []
                    striped_capsules = []
                    state_sector_capsules = []
                    capsules = (
                        index.retrieve_channels(question, prefix=visible_prefix)["answer"]
                        if physics_v325 else index.retrieve(question, prefix=visible_prefix)
                    )
                control_capsules = (
                    control_index.retrieve(question, prefix=visible_prefix)
                    if control_index is not None else []
                )
                # Evaluation labels are intentionally read only after all
                # frozen retrieval channels have returned.
                gold = _evidence_session_ids(row.get("memory_evidence"))
                forgotten = _evidence_session_ids(row.get("forgetting_evidence"))
                forgotten_facts = _evidence_fact_pairs(row.get("forgetting_evidence"))
                if gold:
                    single_gold_queries += len(gold) == 1
                    persona_metric.add(capsules, gold)
                    if physics_v327:
                        atomic_metric.add(atomic_capsules, gold)
                        striped_metric.add(striped_capsules, gold)
                        state_sector_metric.add(state_sector_capsules, gold)
                        sector = set(state_sector_capsules[0]) if state_sector_capsules else set()
                        sector_eval_queries += 1
                        sector_sources += len(sector)
                        sector_gold_hits += len(sector & gold)
                        sector_exact_queries += sector == gold
                    if control_index is not None:
                        control_metric.add(control_capsules, gold)
                if forgotten:
                    forgetting_queries += 1
                    retrieved_sources = _expanded(capsules)
                    forgetting_contaminated += bool(retrieved_sources & forgotten)
                    forgetting_overlap_floor += bool(gold & forgotten)
                    value_contaminated = any(
                        source in retrieved_sources
                        and (
                            value in answer_fact_values_by_id.get(source, set())
                            if (physics_v326 or physics_v327) else value in answer_text_by_id.get(source, "").casefold()
                        )
                        for source, value in forgotten_facts
                    )
                    forgetting_value_contaminated += value_contaminated
                    if value_contaminated:
                        forgetting_value_failures.append({
                            "question": question,
                            "matched_facts": sorted(
                                (source, value) for source, value in forgotten_facts
                                if source in retrieved_sources
                                and value in answer_fact_values_by_id.get(source, set())
                            ),
                        })
                    if control_index is not None:
                        control_forgetting_contaminated += bool(
                            _expanded(control_capsules) & forgotten
                        )
                questions += 1
        checkpoint[checkpoint_key] = {
            "period": period,
            "persona": persona_dir.name,
            "questions": questions,
            "memory_presence_retrieval": persona_metric.result(),
            "atomic_retrieval": atomic_metric.result() if physics_v327 else None,
            "striped_retrieval": striped_metric.result() if physics_v327 else None,
            "state_sector_retrieval": state_sector_metric.result() if physics_v327 else None,
            "single_gold_queries": single_gold_queries,
            "sector_eval_queries": sector_eval_queries,
            "sector_sources": sector_sources,
            "sector_gold_hits": sector_gold_hits,
            "sector_exact_queries": sector_exact_queries,
            "control": control_metric.result() if control_index is not None else None,
            "forgetting_evidence_queries": forgetting_queries,
            "forgetting_contaminated": forgetting_contaminated,
            "forgetting_value_contaminated": forgetting_value_contaminated,
            "forgetting_overlap_floor": forgetting_overlap_floor,
            "control_forgetting_contaminated": control_forgetting_contaminated,
            "compact_queries": compact_queries,
            "compact_payload_chars": compact_payload_chars,
            "compact_raw_chars": compact_raw_chars,
            "ledger_queries": ledger_queries,
            "ledger_sources": ledger_sources,
            "ledger_budgets": ledger_budgets,
            "fact_provenance_items": fact_provenance_items,
            "ledger_fast_path_queries": ledger_fast_path_queries,
            "deterministic_queries": deterministic_queries,
            "deterministic_correct": deterministic_correct,
            "deterministic_failures": deterministic_failures,
            "forgetting_value_failures": forgetting_value_failures,
        }
        checkpoint_path.write_text(
            json.dumps(checkpoint, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        print(f"Memora progress: {position}/{len(eval_paths)} {checkpoint_key}", flush=True)

    requested_keys = {
        f"{path.parent.parent.name}/{path.parent.name}" for path in eval_paths
    }
    records = [checkpoint[key] for key in sorted(requested_keys) if key in checkpoint]
    metrics = [record["memory_presence_retrieval"] for record in records]
    atomic_metrics = [record["atomic_retrieval"] for record in records if record.get("atomic_retrieval")]
    striped_metrics = [record["striped_retrieval"] for record in records if record.get("striped_retrieval")]
    state_sector_metrics = [record["state_sector_retrieval"] for record in records if record.get("state_sector_retrieval")]
    single_gold_queries = sum(record.get("single_gold_queries", 0) for record in records)
    sector_eval_queries = sum(record.get("sector_eval_queries", 0) for record in records)
    sector_sources = sum(record.get("sector_sources", 0) for record in records)
    sector_gold_hits = sum(record.get("sector_gold_hits", 0) for record in records)
    sector_exact_queries = sum(record.get("sector_exact_queries", 0) for record in records)
    control_metrics = [
        record["control"] for record in records if record.get("control") is not None
    ]
    period_names = sorted({record["period"] for record in records})
    forgetting_queries = sum(record["forgetting_evidence_queries"] for record in records)
    forgetting_contaminated = sum(record["forgetting_contaminated"] for record in records)
    forgetting_value_contaminated = sum(
        record.get("forgetting_value_contaminated", 0) for record in records
    )
    forgetting_overlap_floor = sum(
        record.get("forgetting_overlap_floor", 0) for record in records
    )
    compact_queries = sum(record.get("compact_queries", 0) for record in records)
    compact_payload_chars = sum(record.get("compact_payload_chars", 0) for record in records)
    compact_raw_chars = sum(record.get("compact_raw_chars", 0) for record in records)
    ledger_queries = sum(record.get("ledger_queries", 0) for record in records)
    ledger_sources = sum(record.get("ledger_sources", 0) for record in records)
    ledger_budgets = sum(record.get("ledger_budgets", 0) for record in records)
    fact_provenance_items = sum(record.get("fact_provenance_items", 0) for record in records)
    ledger_fast_path_queries = sum(record.get("ledger_fast_path_queries", 0) for record in records)
    deterministic_queries = sum(record.get("deterministic_queries", 0) for record in records)
    deterministic_correct = sum(record.get("deterministic_correct", 0) for record in records)
    return {
        "benchmark": "Memora",
        "status": (
            "umd3282_phase_bounded_three_orbit_retrieval_proxy" if physics_v327
            else "umd326_singularity_fact_version_retrieval_proxy" if physics_v326
            else "umd325_antimatter_census_retrieval_proxy" if physics_v325
            else "umd317_fact_planet_retrieval_proxy" if physics_v317
            else "umd316_physics_retrieval_proxy" if physics_v316
            else "full_retrieval_proxy"
        ),
        "questions": sum(record["questions"] for record in records),
        "completed_persona_periods": len(records),
        "sampling": (
            f"personas {max(0, persona_start_per_period)}:"
            f"{max(0, persona_start_per_period) + max(1, personas_per_period)} per period"
        ),
        "memory_presence_retrieval": _aggregate_capsule_results(metrics),
        "strict_source_atomic_retrieval": (
            _aggregate_capsule_results(atomic_metrics) if atomic_metrics else None
        ),
        "legacy_striped_capsule_retrieval": (
            _aggregate_capsule_results(striped_metrics) if striped_metrics else None
        ),
        "state_sector_capsule_retrieval": (
            _aggregate_capsule_results(state_sector_metrics) if state_sector_metrics else None
        ),
        "strict_source_full_r1_ceiling": (
            single_gold_queries / max(1, sum(metric["evaluated_queries"] for metric in atomic_metrics))
            if atomic_metrics else None
        ),
        "state_sector_audit": {
            "evaluated_queries": sector_eval_queries,
            "mean_sources": sector_sources / max(1, sector_eval_queries),
            "micro_precision": sector_gold_hits / max(1, sector_sources),
            "exact_set_rate": sector_exact_queries / max(1, sector_eval_queries),
        } if physics_v327 else None,
        ("umd326_control" if physics_v327 else "umd325_control" if physics_v326 else "umd317_control" if physics_v325 else "umd316_control" if physics_v317 else "umd315_control"): (
            _aggregate_capsule_results(control_metrics) if control_metrics else None
        ),
        "by_period": {
            period: _aggregate_capsule_results([
                record["memory_presence_retrieval"] for record in records
                if record["period"] == period
            ])
            for period in period_names
        },
        "forgetting_evidence_queries": forgetting_queries,
        "forgetting_session_contamination_at_10": forgetting_contaminated / max(1, forgetting_queries),
        "forgetting_value_contamination_at_10": forgetting_value_contaminated / max(1, forgetting_queries),
        "irreducible_mixed_session_floor_at_10": forgetting_overlap_floor / max(1, forgetting_queries),
        "compact_census": {
            "queries": compact_queries,
            "mean_raw_chars": compact_raw_chars / max(1, compact_queries),
            "mean_payload_chars": compact_payload_chars / max(1, compact_queries),
            "compression_ratio": compact_payload_chars / max(1, compact_raw_chars),
        } if (physics_v326 or physics_v327) else None,
        "structured_ledger": {
            "queries": ledger_queries,
            "mean_matched_sources": ledger_sources / max(1, ledger_queries),
            "mean_source_budget": ledger_budgets / max(1, ledger_queries),
            "fact_provenance_items": fact_provenance_items,
            "fast_path_queries": ledger_fast_path_queries,
            "fast_path_rate": ledger_fast_path_queries / max(1, ledger_queries),
        } if physics_v327 else None,
        "deterministic_answer_validation": {
            "evaluated_queries": deterministic_queries,
            "correct": deterministic_correct,
            "accuracy": deterministic_correct / max(1, deterministic_queries),
        } if physics_v327 else None,
        ("umd326_control_forgetting_contamination_at_10" if physics_v327
         else "umd325_control_forgetting_contamination_at_10" if physics_v326
         else "umd317_control_forgetting_contamination_at_10" if physics_v325
         else "umd316_control_forgetting_contamination_at_10" if physics_v317
         else "umd315_control_forgetting_contamination_at_10"): (
            sum(record.get("control_forgetting_contaminated", 0) for record in records)
            / max(1, forgetting_queries)
            if control_metrics else None
        ),
        "official_fama_run": False,
        "runtime_seconds": time.perf_counter() - started,
    }


def run_realmem(
    encoder: FastEmbedEncoder, *, physics_v316: bool = False,
    physics_v317: bool = False,
    max_personas: int | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    root = VENDOR / "RealMemContent" / "dataset"
    active_physics = physics_v316 or physics_v317
    checkpoint_path = RESULTS / (
        "realmem_umd317_checkpoint.json" if physics_v317
        else "realmem_umd316_v4_checkpoint.json" if physics_v316
        else "realmem_retrieval_checkpoint_v2.json"
    )
    checkpoint: dict[str, Any] = {}
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    paths = sorted(root.glob("*_dialogues_*.json"))
    if max_personas is not None:
        paths = paths[:max(0, max_personas)]
    for position, path in enumerate(paths, 1):
        if path.stem in checkpoint:
            print(f"RealMem checkpoint: {position}/{len(paths)} {path.stem}", flush=True)
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        dialogues = data.get("dialogues", [])
        texts = [
            f"{session.get('current_time', '')} " + " ".join(
                f"{turn.get('speaker', '')}: {turn.get('content', '')}"
                for turn in session.get("dialogue_turns", [])
            )
            for session in dialogues
        ]
        ids = [str(session.get("session_uuid")) for session in dialogues]
        dates = [str(session.get("current_time", "")) for session in dialogues]
        index = OrbitIndex(
            texts, list(range(len(texts))), ids, encoder, dates,
            neural_candidate_pool=192,
            physics_v316=active_physics,
            physics_v317=physics_v317,
        )
        control_index = (
            OrbitIndex(
                texts, list(range(len(texts))), ids, encoder, dates,
                neural_candidate_pool=192,
                physics_v316=physics_v317,
            )
            if active_physics else None
        )
        persona_metric = CapsuleRecallAccumulator((1, 5, 10))
        control_metric = CapsuleRecallAccumulator((1, 5, 10))
        queries = 0
        for session_number, session in enumerate(dialogues):
            turns = session.get("dialogue_turns", [])
            for turn_number, turn in enumerate(turns):
                if not turn.get("is_query"):
                    continue
                answer_turn = next((
                    candidate for candidate in turns[turn_number + 1:]
                    if candidate.get("speaker") == "Assistant"
                    and candidate.get("memory_session_uuids") is not None
                ), {})
                gold = {
                    str(value) for value in answer_turn.get("memory_session_uuids", [])
                }
                if not gold:
                    continue
                capsules = index.retrieve(str(turn.get("content", "")), prefix=session_number)
                persona_metric.add(capsules, gold)
                if control_index is not None:
                    control_capsules = control_index.retrieve(
                        str(turn.get("content", "")), prefix=session_number,
                    )
                    control_metric.add(control_capsules, gold)
                queries += 1
        checkpoint[path.stem] = {
            "queries": queries,
            "metrics": persona_metric.result(),
            "control": control_metric.result() if control_index is not None else None,
        }
        checkpoint_path.write_text(
            json.dumps(checkpoint, ensure_ascii=False, indent=2), encoding="utf-8",
        )
        print(f"RealMem progress: {position}/{len(paths)} {path.stem}", flush=True)
    requested_stems = {path.stem for path in paths}
    records = [checkpoint[key] for key in sorted(requested_stems) if key in checkpoint]
    return {
        "benchmark": "RealMem",
        "status": (
            "umd317_fact_planet_source_retrieval" if physics_v317
            else "umd316_physics_source_retrieval" if physics_v316
            else "full_source_retrieval"
        ),
        "queries": sum(record["queries"] for record in records),
        "metrics": _aggregate_capsule_results([record["metrics"] for record in records]),
        ("umd316_control" if physics_v317 else "umd315_control"): (
            _aggregate_capsule_results([
                record["control"] for record in records
                if record.get("control") is not None
            ]) if active_physics else None
        ),
        "by_persona": {
            key: checkpoint[key]["metrics"] for key in sorted(requested_stems)
            if key in checkpoint
        },
        "official_generation_and_llm_metrics_run": False,
        "runtime_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark",
        choices=("personamem", "membench", "memora", "realmem", "all"),
        default="all",
    )
    parser.add_argument("--membench-per-group", type=int, default=10)
    parser.add_argument("--memora-personas-per-period", type=int, default=1)
    parser.add_argument("--memora-persona-start", type=int, default=0)
    parser.add_argument("--umd316", action="store_true")
    parser.add_argument("--umd317", action="store_true")
    parser.add_argument("--umd325", action="store_true")
    parser.add_argument("--umd326", action="store_true")
    parser.add_argument("--umd327", action="store_true")
    parser.add_argument("--realmem-max-personas", type=int)
    parser.add_argument("--output", type=Path, default=RESULTS / "ai_memory_extended.json")
    args = parser.parse_args()
    encoder = FastEmbedEncoder(cache_dir=MODEL_CACHE, batch_size=128, cache_size=32768, threads=16)
    results: dict[str, Any] = {}
    if args.output.exists():
        results = json.loads(args.output.read_text(encoding="utf-8"))
    if args.benchmark in {"all", "personamem"}:
        results["personamem"] = run_personamem(encoder)
    if args.benchmark in {"all", "membench"}:
        results["membench"] = run_membench(encoder, args.membench_per_group)
    if args.benchmark in {"all", "memora"}:
        results["memora"] = run_memora(
            encoder, args.memora_personas_per_period, physics_v316=args.umd316,
            physics_v317=args.umd317,
            physics_v325=args.umd325,
            physics_v326=args.umd326,
            physics_v327=args.umd327,
            persona_start_per_period=args.memora_persona_start,
        )
    if args.benchmark in {"all", "realmem"}:
        results["realmem"] = run_realmem(
            encoder, physics_v316=args.umd316,
            physics_v317=args.umd317,
            max_personas=args.realmem_max_personas,
        )
    results["metadata"] = {
        "adapter": (
            "UMD 3.28.2 phase-bounded three-orbit retrieval" if args.umd327
            else "UMD 3.26 singularity, fact-version, and emergent-domain retrieval" if args.umd326
            else "UMD 3.25 antimatter and galactic-census retrieval" if args.umd325
            else "UMD 3.17 fact-planet retrieval" if args.umd317
            else "UMD 3.16 signed-state constellation retrieval"
            if args.umd316 else "UMD 3.15 event-Lagrange retrieval"
        ),
        "paid_api_calls": 0,
        "gold_used_for_ranking": False,
        "answer_model": None,
        "judge_model": None,
        "encoder": encoder.metadata(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
