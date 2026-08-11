from __future__ import annotations

import random
import unittest

from benchmarks.public_benchmarks import tokenize, unit_scores
from benchmarks.umd314_adapter import (
    directive_gravity_values,
    relative_gravity_order,
    transient_directive_bridge,
)
from benchmarks.umd316_adapter import (
    _content_tokens,
    _entity_set,
    query_physics,
    rank_constellation_coverage,
)
from benchmarks.umd325_adapter import split_antimatter_orbits


def _reference_coverage(
    query, atomic_orbit, stable_capsules, texts, groups, dates,
    force, semantic, lexical, *, pool_size=192,
):
    """Frozen pre-3.29 set-algebra implementation used as an oracle."""
    stable = {source for capsule in stable_capsules for source in capsule}
    pool = [source for source in atomic_orbit if source not in stable][:pool_size]
    if not pool:
        return []
    if not query_physics(query).broad_set:
        return sorted(
            pool,
            key=lambda i: (-(0.46 * force[i] + 0.34 * semantic[i] + 0.20 * lexical[i]), i),
        )
    feature_sources = stable | set(pool)
    tokens = {source: _content_tokens(texts[source]) for source in feature_sources}
    entities = {source: _entity_set(texts[source]) for source in feature_sources}
    force_u = unit_scores([force[i] for i in pool])
    semantic_u = unit_scores([semantic[i] for i in pool])
    lexical_u = unit_scores([lexical[i] for i in pool])
    base = {
        source: 0.44 * force_u[pos] + 0.36 * semantic_u[pos] + 0.20 * lexical_u[pos]
        for pos, source in enumerate(pool)
    }
    selected: list[int] = []
    selected_set: set[int] = set()
    covered_tokens = set().union(*(tokens[i] for i in stable)) if stable else set()
    covered_entities = set().union(*(entities[i] for i in stable)) if stable else set()
    covered_groups = {groups[i] for i in stable}
    covered_dates = {
        dates[groups[i]] for i in stable
        if groups[i] < len(dates) and dates[groups[i]]
    }
    query_tokens = _content_tokens(query)
    while len(selected) < len(pool):
        best = None
        best_score = float("-inf")
        for source in pool:
            if source in selected_set:
                continue
            source_tokens = tokens[source]
            group = groups[source]
            date = dates[group] if group < len(dates) else ""
            score = (
                base[source]
                + 0.10 * (len(source_tokens & query_tokens) / max(1, len(query_tokens)))
                + 0.09 * bool(entities[source] - covered_entities)
                + 0.07 * (group not in covered_groups)
                + 0.04 * bool(date and date not in covered_dates)
                + 0.08 * (len(source_tokens - covered_tokens) / max(1, len(source_tokens)))
                - 0.13 * (
                    len(source_tokens & covered_tokens)
                    / max(1, len(source_tokens | covered_tokens))
                )
            )
            if score > best_score or (
                score == best_score and (best is None or source < best)
            ):
                best, best_score = source, score
        if best is None:
            break
        selected.append(best)
        selected_set.add(best)
        covered_tokens.update(tokens[best])
        covered_entities.update(entities[best])
        covered_groups.add(groups[best])
        group = groups[best]
        if group < len(dates) and dates[group]:
            covered_dates.add(dates[group])
    return selected


class UMD329PerformanceTests(unittest.TestCase):
    def test_incremental_constellation_is_exactly_reference_equivalent(self) -> None:
        rng = random.Random(329)
        vocabulary = [
            "alpha", "beta", "gamma", "report", "camp", "task", "travel",
            "music", "book", "Maya", "Orion", "Atlas", "Nova",
        ]
        for _ in range(25):
            size = rng.randint(5, 30)
            texts = [
                " ".join(rng.choices(vocabulary, k=rng.randint(4, 20)))
                for _ in range(size)
            ]
            groups = [rng.randrange(max(1, size // 3)) for _ in range(size)]
            dates = [f"2025-{index + 1:02d}" for index in range(max(groups) + 1)]
            score_fields = [[rng.random() for _ in range(size)] for _ in range(3)]
            orbit = list(range(size))
            rng.shuffle(orbit)
            args = (
                "List all current tasks and travel books", orbit, [(0,), (1,)],
                texts, groups, dates, *score_fields,
            )
            expected = _reference_coverage(*args)
            self.assertEqual(rank_constellation_coverage(*args), expected)
            self.assertEqual(
                rank_constellation_coverage(
                    *args,
                    source_token_sets=[frozenset(tokenize(text)) for text in texts],
                ),
                expected,
            )

    def test_batched_directive_field_preserves_order_and_bridge(self) -> None:
        query = "How should I implement this private workflow?"
        texts = ["Use code blocks.", "Keep private data secret.", "Unrelated note."]
        metadata = [("instruction", "format"), ("instruction", "privacy"), None]
        order = [0, 1, 2]
        force = semantic = lexical = [0.8, 0.7, 0.1]
        values = directive_gravity_values(query, texts, metadata)
        default_order = relative_gravity_order(
            query, texts, [0, 1, 2], ["", "", ""], order,
            force, semantic, lexical, metadata, top_k=10,
        )
        cached_order = relative_gravity_order(
            query, texts, [0, 1, 2], ["", "", ""], order,
            force, semantic, lexical, metadata, top_k=10,
            directive_values=values,
        )
        self.assertEqual(cached_order, default_order)
        self.assertEqual(
            transient_directive_bridge(query, texts, order, metadata),
            transient_directive_bridge(
                query, texts, order, metadata, directive_values=values,
            ),
        )

    def test_cached_antimatter_ids_match_full_scan(self) -> None:
        query = "What tasks remain now?"
        capsules = [("old", "new"), ("other",)]
        ids = ["old", "new", "other"]
        texts = ["Operation delete old task", "Operation add new task", "note"]
        expected = split_antimatter_orbits(query, capsules, ids, texts)
        cached = split_antimatter_orbits(
            query, capsules, ids, texts, negative_ids=frozenset({"old"}),
        )
        self.assertEqual(cached, expected)


if __name__ == "__main__":
    unittest.main()
