"""UMD 3.15 bounded event satellites for production gravity retrieval.

UMD 3.14 builds stable Lagrange bridges.  This layer uses the remaining
query-relevant candidate orbit as transient event satellites: it appends
provenance and bounded excerpts to existing bridges without deleting a stable
source or mutating durable hierarchy membership.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime

try:
    from .umd35_core import utcnow
    from .umd313_gravity import GravityHit
    from .umd314_lagrange import UMD314Config, UMD314LagrangeMemory, _AdaptiveCandidate
except ImportError:
    from umd35_core import utcnow
    from umd313_gravity import GravityHit
    from umd314_lagrange import UMD314Config, UMD314LagrangeMemory, _AdaptiveCandidate


@dataclass
class UMD315Config(UMD314Config):
    event_satellite_sources: int = 16
    event_candidate_pool: int = 64
    event_min_excerpt_chars: int = 80
    event_force_weight: float = 0.46
    event_semantic_weight: float = 0.30
    event_lexical_weight: float = 0.14
    event_rank_weight: float = 0.06
    event_new_planet_weight: float = 0.04


class UMD315EventLagrangeMemory(UMD314LagrangeMemory):
    """Attach bounded secondary-orbit evidence to stable UMD 3.14 bridges."""

    EVENT_FORMULA = (
        "H=.46*field+.30*semantic+.14*lexical+.06*reciprocal_rank+.04*new_planet"
    )

    def __init__(self, durable, *, config: UMD315Config | None = None, **kwargs) -> None:
        self.event_config = config or UMD315Config()
        super().__init__(durable, config=self.event_config, **kwargs)

    def _event_score(
        self, candidate: _AdaptiveCandidate, rank: int, stable_planets: set[str],
    ) -> float:
        semantic = max((item.semantic_resonance for item in candidate.elements), default=0.0)
        lexical = max((item.lexical_resonance for item in candidate.elements), default=0.0)
        new_planet = bool(
            candidate.node.planet_id and candidate.node.planet_id not in stable_planets
        )
        return (
            self.event_config.event_force_weight * candidate.best_potential
            + self.event_config.event_semantic_weight * semantic
            + self.event_config.event_lexical_weight * lexical
            + self.event_config.event_rank_weight / (1.0 + rank)
            + self.event_config.event_new_planet_weight * float(new_planet)
        )

    def retrieve_gravity(
        self, query: str, *, scope: str | None = None, top_k: int = 10,
        budget_chars: int | None = None, now: datetime | None = None,
    ) -> list[GravityHit]:
        current = now or utcnow()
        stable = super().retrieve_gravity(
            query, scope=scope, top_k=top_k, budget_chars=budget_chars, now=current,
        )
        if not stable or self.event_config.event_satellite_sources <= 0:
            return stable

        _, candidates, _, _ = self._adaptive_candidates(query, scope, current)
        stable_sources = {source for hit in stable for source in hit.source_ids}
        stable_planets = {
            candidate.node.planet_id
            for candidate in candidates
            if candidate.node.id in stable_sources and candidate.node.planet_id
        }
        pool = [
            candidate for candidate in candidates
            if candidate.node.id not in stable_sources
        ][: self.event_config.event_candidate_pool]
        rank_by_id = {candidate.node.id: rank for rank, candidate in enumerate(candidates)}
        ranked = sorted(
            pool,
            key=lambda candidate: (
                -self._event_score(
                    candidate, rank_by_id[candidate.node.id], stable_planets,
                ),
                rank_by_id[candidate.node.id],
                candidate.node.id,
            ),
        )[: self.event_config.event_satellite_sources]

        budget = budget_chars or self.durable.engine.config.default_budget_chars
        remaining = max(0, budget - sum(len(hit.text) for hit in stable))
        output = list(stable)
        for offset, candidate in enumerate(ranked):
            if remaining < self.event_config.event_min_excerpt_chars:
                break
            target = offset % len(output)
            hit = output[target]
            prefix = f"\n[event-satellite:{candidate.node.id}] "
            room = remaining - len(prefix)
            if room <= 0:
                break
            excerpt = candidate.text[:room]
            addition = prefix + excerpt
            source_ids = tuple(dict.fromkeys((*hit.source_ids, candidate.node.id)))
            payload = "event_lagrange\0" + "\0".join(source_ids)
            explanation = dict(hit.explanation)
            attached = list(explanation.get("event_satellite_sources", ()))
            attached.append(candidate.node.id)
            explanation.update({
                "event_satellite_sources": tuple(attached),
                "event_satellite_formula": self.EVENT_FORMULA,
                "event_satellite_limit": self.event_config.event_satellite_sources,
                "stable_sources_preserved": True,
                "persistent_reparenting": False,
                "immutable_provenance": True,
                "selection_algorithm": "bounded_event_lagrange",
            })
            elements = dict(hit.elements)
            elements["event_satellite_score"] = self._event_score(
                candidate, rank_by_id[candidate.node.id], stable_planets,
            )
            output[target] = GravityHit(
                "evt_" + hashlib.sha256(payload.encode()).hexdigest()[:20],
                source_ids,
                hit.text + addition,
                max(hit.force, elements["event_satellite_score"]),
                elements,
                explanation,
            )
            remaining -= len(addition)
        return output

    def snapshot(self) -> dict[str, object]:
        base = super().snapshot()
        base.update({
            "version": "UMD 3.15",
            "event_satellite_sources": self.event_config.event_satellite_sources,
            "event_candidate_pool": self.event_config.event_candidate_pool,
            "event_horizon_formula": self.EVENT_FORMULA,
            "stable_sources_preserved": True,
            "persistent_orbits_mutated_by_query": False,
        })
        return base
