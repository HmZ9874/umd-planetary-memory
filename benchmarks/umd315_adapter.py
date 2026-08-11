"""Gold-blind UMD 3.15 event-horizon compression for long dialogue retrieval.

The adapter treats the mature top-ten capsules as stable planets.  Sources in
the secondary atomic orbit are attached as transient event satellites without
removing any existing provenance.  Ranking uses the query-derived UMD orbit;
answers, evidence IDs and benchmark question categories are never inputs.
"""

from __future__ import annotations

import math
import re
from typing import Sequence


INFERENCE_OR_SET_RE = re.compile(
    r"\b(?:would|likely|could|personality|political|religious|identity|"
    r"relationship status|career path|activities|events|books|things|ways|"
    r"interests?|participat(?:e|ed)|what does .+ do|where has .+|"
    r"has .+ (?:done|read|visited|camped))\b",
    re.IGNORECASE,
)
TEMPORAL_POINT_RE = re.compile(
    r"\b(?:when|what date|how long|how many (?:days|weeks|months|years)|how long ago)\b",
    re.IGNORECASE,
)
CAUSAL_PLAN_RE = re.compile(
    r"\b(?:why|how did|plans?|relationship|changed?|developed?|realize[ds]?)\b",
    re.IGNORECASE,
)
def event_satellite_budget(query: str, *, ceiling: int = 20) -> int:
    """Predict evidence-set width from query form without benchmark labels."""
    if ceiling <= 0:
        return 0
    if INFERENCE_OR_SET_RE.search(query):
        return min(20, ceiling)
    if TEMPORAL_POINT_RE.search(query):
        return min(6, ceiling)
    if CAUSAL_PLAN_RE.search(query):
        return min(12, ceiling)
    return min(8, ceiling)


def rank_event_satellites(
    atomic_orbit: Sequence[int],
    stable_capsules: Sequence[tuple[int, ...]],
    groups: Sequence[int],
    force: Sequence[float],
    semantic: Sequence[float],
    lexical: Sequence[float],
    *,
    pool_size: int = 64,
) -> list[int]:
    """Re-rank the secondary orbit for event relevance and planet diversity."""
    stable_sources = {source for capsule in stable_capsules for source in capsule}
    stable_groups = {groups[source] for source in stable_sources}
    pool = [source for source in atomic_orbit if source not in stable_sources][:pool_size]
    rank_by_source = {source: rank for rank, source in enumerate(atomic_orbit)}
    return sorted(
        pool,
        key=lambda source: (
            -(
                0.46 * force[source]
                + 0.30 * semantic[source]
                + 0.14 * lexical[source]
                + 0.06 / (1.0 + rank_by_source[source])
                + 0.04 * float(groups[source] not in stable_groups)
            ),
            rank_by_source[source],
            source,
        ),
    )


def event_field_entropy_budget(
    satellite_orbit: Sequence[int],
    stable_capsules: Sequence[tuple[int, ...]],
    groups: Sequence[int],
    force: Sequence[float],
    semantic: Sequence[float],
    lexical: Sequence[float],
    *,
    floor: int = 4,
    ceiling: int = 8,
    temperature: float = 0.08,
    mass_target: float = 0.70,
) -> int:
    """Allocate satellites from normalized field entropy, without labels."""
    if ceiling <= 0 or not satellite_orbit:
        return 0
    upper = min(ceiling, len(satellite_orbit))
    lower = min(max(1, floor), upper)
    stable_groups = {
        groups[source] for capsule in stable_capsules for source in capsule
    }
    values = [
        0.46 * force[source]
        + 0.30 * semantic[source]
        + 0.14 * lexical[source]
        + 0.04 * float(groups[source] not in stable_groups)
        for source in satellite_orbit[:upper]
    ]
    peak = max(values)
    weights = [math.exp((value - peak) / max(1e-6, temperature)) for value in values]
    total = sum(weights)
    cumulative = 0.0
    for count, weight in enumerate(weights, 1):
        cumulative += weight
        if count >= lower and cumulative / max(1e-9, total) >= mass_target:
            return count
    return upper


def compress_event_horizon(
    stable_capsules: Sequence[tuple[int, ...]],
    atomic_orbit: Sequence[int],
    *,
    max_extra_sources: int = 20,
) -> list[tuple[int, ...]]:
    """Compress a secondary atomic orbit into ten provenance-preserving capsules.

    Satellites are striped across capsules so their first possible rank follows
    the atomic orbit.  Every stable capsule remains a subset of its output
    capsule, giving prefix-level recall monotonicity at the ten-capsule boundary.
    """
    output = [tuple(capsule) for capsule in stable_capsules]
    if not output or max_extra_sources <= 0:
        return output

    stable_sources = {source for capsule in output for source in capsule}
    satellites: list[int] = []
    for source in atomic_orbit:
        if source in stable_sources or source in satellites:
            continue
        satellites.append(source)
        if len(satellites) >= max_extra_sources:
            break

    width = min(10, len(output))
    for offset, source in enumerate(satellites):
        capsule_index = offset % width
        output[capsule_index] = tuple(dict.fromkeys(output[capsule_index] + (source,)))
    return output
