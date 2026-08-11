"""UMD 3.10 model-neutral evidence retrieval benchmark.

The adapter uses only raw questions and conversation text for ranking. Gold
evidence IDs are read after ranking, exclusively for metric calculation. It
does not call an answer LLM or an LLM judge, so its results must not be
presented as official end-to-end QA scores.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pyarrow.parquet as pq

from benchmarks.public_benchmarks import (
    BM25,
    RecallAccumulator,
    _locomo_sessions,
    _turn_text,
    char_tokenize,
    peak_working_set_mib,
    planetary_query_weights,
    normalize_locomo_evidence_ids,
    ranking,
    reciprocal_ranks,
    tokenize,
    unit_scores,
)
from umd35_neural import FastEmbedEncoder
from umd310_cognitive import UMD310CognitiveMemory
from benchmarks.umd314_adapter import (
    fuse_transient_bridge,
    relative_gravity_order,
    transient_directive_bridge,
)
from benchmarks.umd315_adapter import (
    compress_event_horizon,
    event_field_entropy_budget,
    event_satellite_budget,
    rank_event_satellites,
)


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
MODEL_CACHE = ROOT.parent / ".model-cache"
BEAM_TYPES = (
    "abstention",
    "contradiction_resolution",
    "event_ordering",
    "information_extraction",
    "instruction_following",
    "knowledge_update",
    "multi_session_reasoning",
    "preference_following",
    "summarization",
    "temporal_reasoning",
)
TIME_WORDS = {
    "when", "before", "after", "latest", "current", "previous", "first",
    "last", "recent", "order", "timeline", "weeks", "months", "date",
}
IDENTIFIER_RE = re.compile(r"\b(?:[A-Za-z]+[-_.:/]?\d+(?:\.\d+)*|\d+(?:\.\d+)*)\b")
ENTITY_RE = re.compile(r"\b[A-Z][A-Za-z0-9_.-]{2,}\b")


def _dot_scores(query_vector: np.ndarray, vectors: Sequence[np.ndarray]) -> list[float]:
    return [max(0.0, float(np.dot(query_vector, vector))) for vector in vectors]


def _entity_value_resonance(query: str, texts: Sequence[str]) -> list[float]:
    query_items = set(IDENTIFIER_RE.findall(query)) | set(ENTITY_RE.findall(query))
    if not query_items:
        return [0.0] * len(texts)
    values = []
    for text in texts:
        text_items = set(IDENTIFIER_RE.findall(text)) | set(ENTITY_RE.findall(text))
        values.append(len(query_items & text_items) / max(1, len(query_items)))
    return values


def _temporal_phase(query: str, dates: Sequence[str]) -> list[float]:
    words = set(tokenize(query))
    if not (words & TIME_WORDS or re.search(r"\b(?:19|20)\d{2}\b", query)):
        return [0.0] * len(dates)
    date_index = BM25([date or "unknown" for date in dates])
    raw = unit_scores(date_index.scores(query))
    if any(raw):
        return raw
    # A temporal query without a literal date gets a weak, monotonic recency
    # phase. This is deliberately small so it cannot dominate semantic facts.
    denominator = max(1, len(dates) - 1)
    return [index / denominator for index in range(len(dates))]


def _graph_flux(values: Sequence[float], groups: Sequence[int]) -> list[float]:
    flux = [0.0] * len(values)
    for index, value in enumerate(values):
        neighbors = [value]
        if index and groups[index - 1] == groups[index]:
            neighbors.append(values[index - 1])
        if index + 1 < len(values) and groups[index + 1] == groups[index]:
            neighbors.append(values[index + 1])
        flux[index] = max(neighbors)
    return unit_scores(flux)


def _orbit_expand(seed_order: Sequence[int], groups: Sequence[int]) -> list[int]:
    output: list[int] = []
    seen: set[int] = set()
    for seed in seed_order:
        for candidate in (seed, seed - 1, seed + 1):
            if (
                0 <= candidate < len(groups)
                and candidate not in seen
                and groups[candidate] == groups[seed]
            ):
                output.append(candidate)
                seen.add(candidate)
    output.extend(index for index in range(len(groups)) if index not in seen)
    return output


def _force(
    semantic: Sequence[float], lexical: Sequence[float], planetary: Sequence[float],
    char: Sequence[float], entity: Sequence[float], temporal: Sequence[float],
    graph: Sequence[float],
) -> list[float]:
    """Frozen UMD 3.9 raw-text force; weights sum to one.

    Authority and learned utility are uniform in public raw datasets and thus
    omitted. The production engine still includes both fields.
    """
    return [
        0.34 * semantic[i]
        + 0.22 * lexical[i]
        + 0.16 * planetary[i]
        + 0.08 * char[i]
        + 0.08 * entity[i]
        + 0.06 * temporal[i]
        + 0.06 * graph[i]
        for i in range(len(semantic))
    ]


def _directive_metadata(text: str, *, trusted: bool) -> tuple[str, str] | None:
    """Extract a trusted cognitive satellite without using benchmark labels."""
    if not trusted:
        return None
    role = UMD310CognitiveMemory._directive_role(text)
    if role is None:
        return None
    return role, UMD310CognitiveMemory._directive_family(text)


def _token_set(text: str) -> set[str]:
    return set(tokenize(text))


def _cognitive_order(
    query: str,
    texts: Sequence[str],
    groups: Sequence[int],
    dates: Sequence[str],
    base_order: Sequence[int],
    force: Sequence[float],
    semantic: Sequence[float],
    lexical: Sequence[float],
    directive_meta: Sequence[tuple[str, str] | None],
    *,
    top_k: int,
) -> list[int]:
    """UMD 3.10 label-blind cognitive routing over a UMD 3.9 orbit.

    The first UMD 3.9 anchor remains fixed. Task/preference queries reserve
    three satellite slots for the latest trusted directive in each family.
    Episodic queries greedily cover new planets, time phases and entities.
    """
    if not base_order:
        return []
    mode = UMD310CognitiveMemory._query_mode(query)
    latest: dict[tuple[str, str], int] = {}
    for index, metadata in enumerate(directive_meta):
        if metadata is not None:
            latest[metadata] = index

    requested_roles: set[str] = set()
    if mode.instructions:
        requested_roles.add("instruction")
    if mode.preferences:
        requested_roles.add("preference")
    if not requested_roles and not mode.episodic:
        return list(base_order)
    directive_candidates = [
        index for (role, _), index in latest.items() if role in requested_roles
    ]
    denominator = max(1, len(texts) - 1)
    directive_candidates.sort(key=lambda index: (
        -(0.55 * semantic[index] + 0.25 * lexical[index] + 0.20 * index / denominator),
        -index,
    ))

    selected: list[int] = [base_order[0]]
    selected_set = {base_order[0]}
    for index in directive_candidates[:3]:
        if index not in selected_set:
            selected.append(index)
            selected_set.add(index)

    # Keep computation bounded while admitting the best item from every
    # planet for multi-session/temporal questions.
    pool = list(base_order[:max(64, top_k * 4)])
    if mode.episodic:
        group_best: dict[int, int] = {}
        for index in base_order:
            group = groups[index]
            current = group_best.get(group)
            if current is None or force[index] > force[current]:
                group_best[group] = index
        pool.extend(group_best.values())
    pool.extend(directive_candidates[:3])
    pool = list(dict.fromkeys(pool))

    if mode.episodic:
        selected_groups = {groups[index] for index in selected}
        selected_dates = {
            dates[groups[index]] for index in selected
            if groups[index] < len(dates) and dates[groups[index]]
        }
        selected_entities: set[str] = set()
        selected_tokens: set[str] = set()
        for index in selected:
            selected_entities.update(IDENTIFIER_RE.findall(texts[index]))
            selected_entities.update(ENTITY_RE.findall(texts[index]))
            selected_tokens.update(_token_set(texts[index]))
        while len(selected) < min(top_k, len(texts)):
            best_index = None
            best_score = float("-inf")
            for index in pool:
                if index in selected_set:
                    continue
                group = groups[index]
                date = dates[group] if group < len(dates) else ""
                entities = set(IDENTIFIER_RE.findall(texts[index])) | set(ENTITY_RE.findall(texts[index]))
                tokens = _token_set(texts[index])
                redundancy = len(tokens & selected_tokens) / max(1, len(tokens | selected_tokens))
                score = (
                    force[index]
                    + 0.16 * (group not in selected_groups)
                    + 0.10 * bool(date and date not in selected_dates)
                    + 0.08 * bool(entities - selected_entities)
                    - 0.12 * redundancy
                )
                if score > best_score or (score == best_score and (best_index is None or index < best_index)):
                    best_index, best_score = index, score
            if best_index is None:
                break
            selected.append(best_index)
            selected_set.add(best_index)
            selected_groups.add(groups[best_index])
            group = groups[best_index]
            if group < len(dates) and dates[group]:
                selected_dates.add(dates[group])
            selected_entities.update(IDENTIFIER_RE.findall(texts[best_index]))
            selected_entities.update(ENTITY_RE.findall(texts[best_index]))
            selected_tokens.update(_token_set(texts[best_index]))
    else:
        remaining = [index for index in pool if index not in selected_set]
        remaining.sort(key=lambda index: (-force[index], base_order.index(index)))
        selected.extend(remaining[:max(0, top_k - len(selected))])
        selected_set.update(selected)

    selected.extend(index for index in base_order if index not in selected_set)
    selected_set.update(selected)
    selected.extend(index for index in range(len(texts)) if index not in selected_set)
    return selected


def _metrics_by_type(accumulators: dict[str, RecallAccumulator]) -> dict[str, dict]:
    return {key: value.result() for key, value in sorted(accumulators.items())}


class CapsuleRecallAccumulator:
    """Recall at K capsules after expanding immutable source provenance."""

    def __init__(self, ks: Sequence[int]) -> None:
        self.ks = tuple(ks)
        self.queries = self.gold_items = 0
        self.rr_sum = 0.0
        self.any_hits = {k: 0 for k in self.ks}
        self.full_hits = {k: 0 for k in self.ks}
        self.micro_hits = {k: 0 for k in self.ks}

    def add(self, capsules: Sequence[Sequence[str]], gold: Iterable[str]) -> None:
        gold_set = set(gold)
        if not gold_set:
            return
        self.queries += 1
        self.gold_items += len(gold_set)
        first = next(
            (index for index, capsule in enumerate(capsules, 1) if gold_set & set(capsule)),
            None,
        )
        if first:
            self.rr_sum += 1.0 / first
        for k in self.ks:
            expanded = {source for capsule in capsules[:k] for source in capsule}
            overlap = expanded & gold_set
            self.any_hits[k] += bool(overlap)
            self.full_hits[k] += gold_set <= expanded
            self.micro_hits[k] += len(overlap)

    def result(self) -> dict[str, Any]:
        denominator = max(1, self.queries)
        gold_denominator = max(1, self.gold_items)
        return {
            "evaluated_queries": self.queries,
            "gold_evidence_items": self.gold_items,
            "retrieval_unit": "evidence_capsule",
            "mrr": self.rr_sum / denominator,
            "any_evidence_recall": {str(k): self.any_hits[k] / denominator for k in self.ks},
            "full_evidence_recall": {str(k): self.full_hits[k] / denominator for k in self.ks},
            "micro_evidence_recall": {str(k): self.micro_hits[k] / gold_denominator for k in self.ks},
        }


def _benchmark_capsules(groups: Sequence[int], max_sources: int = 4) -> tuple[
    list[tuple[int, ...]], dict[int, list[int]]
]:
    """Build query-independent atomic, adjacent and bounded episode capsules."""
    capsules: list[tuple[int, ...]] = [(index,) for index in range(len(groups))]
    group_members: dict[int, list[int]] = defaultdict(list)
    for index, group in enumerate(groups):
        group_members[group].append(index)
    for members in group_members.values():
        capsules.extend(tuple(members[index:index + 2]) for index in range(len(members) - 1))
        capsules.extend(
            tuple(members[index:index + max_sources])
            for index in range(0, len(members), max_sources - 1)
            if len(members[index:index + max_sources]) >= 2
        )
    capsules = list(dict.fromkeys(capsules))
    by_source: dict[int, list[int]] = defaultdict(list)
    for capsule_index, capsule in enumerate(capsules):
        for source in capsule:
            by_source[source].append(capsule_index)
    return capsules, by_source


def _anchored_capsule_order(
    query: str, base_order: Sequence[int], force: Sequence[float],
    capsules: Sequence[tuple[int, ...]], by_source: dict[int, list[int]], *, top_k: int,
) -> list[tuple[int, ...]]:
    """Expand each atomic top-K anchor without ever losing that source."""
    mode = UMD310CognitiveMemory._query_mode(query)
    selected: list[tuple[int, ...]] = []
    selected_capsules: set[int] = set()
    used_sources: set[int] = set()
    for anchor in base_order[:top_k]:
        best_index = None
        best_value = float("-inf")
        for capsule_index in by_source.get(anchor, ()):
            if capsule_index in selected_capsules:
                continue
            capsule = capsules[capsule_index]
            # Ordinary exact questions need tight dialogue pairs; broad
            # episodic questions may use four-source event capsules.
            if not mode.episodic and len(capsule) > 2:
                continue
            mean_force = sum(force[source] for source in capsule) / len(capsule)
            new_sources = set(capsule) - used_sources
            overlap = len(set(capsule) & used_sources) / len(capsule)
            density = len(new_sources) / 4.0
            value = (
                0.58 * force[anchor] + 0.22 * mean_force + 0.20 * density
                - 0.18 * overlap
            )
            if value > best_value or (
                value == best_value and (best_index is None or capsule < capsules[best_index])
            ):
                best_index, best_value = capsule_index, value
        if best_index is None:
            capsule = (anchor,)
        else:
            capsule = capsules[best_index]
            selected_capsules.add(best_index)
        selected.append(capsule)
        used_sources.update(capsule)
    return selected


def run_locomo(
    path: Path, encoder: FastEmbedEncoder, *, start: int = 0, stop: int | None = None,
    event_extra_sources: int = 16, adaptive_event_width: bool = False,
    entropy_event_width: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    all_dataset = json.loads(path.read_text(encoding="utf-8"))
    dataset = all_dataset[max(0, start):stop]
    ks = (1, 5, 10, 20, 50, 100)
    metric = RecallAccumulator(ks)
    metric310 = RecallAccumulator(ks)
    metric312 = CapsuleRecallAccumulator(ks)
    metric314 = CapsuleRecallAccumulator(ks)
    metric315 = CapsuleRecallAccumulator(ks)
    event_curve_counts = tuple(range(4, max(4, event_extra_sources) + 1))
    metric315_curve = {
        count: CapsuleRecallAccumulator(ks) for count in event_curve_counts
    }
    official_metric = RecallAccumulator(ks)
    official_metric310 = RecallAccumulator(ks)
    official_metric312 = CapsuleRecallAccumulator(ks)
    official_metric314 = CapsuleRecallAccumulator(ks)
    official_metric315 = CapsuleRecallAccumulator(ks)
    by_type: dict[str, RecallAccumulator] = defaultdict(lambda: RecallAccumulator(ks))
    by_type310: dict[str, RecallAccumulator] = defaultdict(lambda: RecallAccumulator(ks))
    by_type312: dict[str, CapsuleRecallAccumulator] = defaultdict(lambda: CapsuleRecallAccumulator(ks))
    by_type314: dict[str, CapsuleRecallAccumulator] = defaultdict(lambda: CapsuleRecallAccumulator(ks))
    by_type315: dict[str, CapsuleRecallAccumulator] = defaultdict(lambda: CapsuleRecallAccumulator(ks))
    context_chars: list[int] = []
    context_chars310: list[int] = []
    context_chars312: list[int] = []
    context_chars314: list[int] = []
    context_chars315: list[int] = []
    context_chars315_curve: dict[int, list[int]] = {
        count: [] for count in event_curve_counts
    }
    total = no_evidence = official_qa_total = 0
    event_budget_histogram: Counter[int] = Counter()

    for sample in dataset:
        _, sessions, dates = _locomo_sessions(sample)
        turns = [turn for session in sessions for turn in session]
        turn_ids = [str(turn["dia_id"]) for turn in turns]
        texts = [
            f'{turn.get("speaker", "")}: {turn.get("text", "")} {turn.get("blip_caption", "")}'
            for turn in turns
        ]
        groups: list[int] = []
        session_turns: list[list[int]] = []
        cursor = 0
        for session_index, session in enumerate(sessions):
            indices = list(range(cursor, cursor + len(session)))
            session_turns.append(indices)
            groups.extend([session_index] * len(session))
            cursor += len(session)
        session_texts = [
            f"{dates[i]} " + " ".join(texts[j] for j in session_turns[i])
            for i in range(len(sessions))
        ]
        lexical_index = BM25(texts)
        char_index = BM25(texts, tokenizer=char_tokenize)
        planet_index = BM25(session_texts)
        vectors = encoder.encode_many(texts)
        # LoCoMo is a peer-to-peer dialogue without user/assistant trust roles;
        # both named speakers are first-party conversational evidence.
        directive_meta = [_directive_metadata(text, trusted=True) for text in texts]
        capsules, capsules_by_source = _benchmark_capsules(groups)
        qa_vectors = encoder.encode_many([str(qa["question"]) for qa in sample["qa"]])

        for qa_number, qa in enumerate(sample["qa"]):
            total += 1
            category = str(qa.get("category", "unknown"))
            if category in {"1", "2", "3", "4"}:
                official_qa_total += 1
            gold = normalize_locomo_evidence_ids(qa.get("evidence"))
            if not gold:
                no_evidence += 1
                continue
            query = qa["question"]
            lexical_raw = lexical_index.scores(query)
            planet_raw = planet_index.scores(query)
            lexical_order = ranking(lexical_raw)
            planet_order = ranking(planet_raw)
            lexical_rr = reciprocal_ranks(lexical_order)
            planet_rr = reciprocal_ranks(planet_order)
            seed = ranking([
                lexical_rr[i] + 0.72 * planet_rr[groups[i]]
                for i in range(len(texts))
            ])
            orbit = _orbit_expand(seed, groups)
            protected = orbit[:10]
            qv = qa_vectors[qa_number]
            semantic = _dot_scores(qv, vectors)
            lexical = unit_scores(lexical_raw)
            planetary = [unit_scores(planet_raw)[groups[i]] for i in range(len(texts))]
            char = unit_scores(char_index.scores(query))
            entity = _entity_value_resonance(query, texts)
            temporal_sessions = _temporal_phase(query, dates)
            temporal = [temporal_sessions[groups[i]] for i in range(len(texts))]
            graph = _graph_flux(lexical, groups)
            force = _force(semantic, lexical, planetary, char, entity, temporal, graph)
            head = sorted(protected, key=lambda i: (-force[i], i))
            protected_set = set(protected)
            order = head + [i for i in orbit if i not in protected_set]
            ranked_ids = [turn_ids[i] for i in order]
            order310 = _cognitive_order(
                query, texts, groups, dates, order, force, semantic, lexical,
                directive_meta, top_k=10,
            )
            ranked_ids310 = [turn_ids[i] for i in order310]
            capsule_order = _anchored_capsule_order(
                query, order, force, capsules, capsules_by_source, top_k=10,
            )
            order314, force314 = relative_gravity_order(
                query, texts, groups, dates, order, force, semantic, lexical,
                directive_meta, top_k=10,
            )
            bridge314 = transient_directive_bridge(
                query, texts, order314, directive_meta,
            )
            capsule_order314 = fuse_transient_bridge(capsule_order, bridge314)
            satellite_orbit315 = rank_event_satellites(
                order314, capsule_order314, groups, force314, semantic, lexical,
            )
            if entropy_event_width:
                event_budget315 = event_field_entropy_budget(
                    satellite_orbit315, capsule_order314, groups,
                    force314, semantic, lexical,
                    floor=4, ceiling=event_extra_sources,
                )
            elif adaptive_event_width:
                event_budget315 = event_satellite_budget(
                    query, ceiling=event_extra_sources,
                )
            else:
                event_budget315 = event_extra_sources
            event_budget_histogram[event_budget315] += 1
            capsule_order315 = compress_event_horizon(
                capsule_order314, satellite_orbit315,
                max_extra_sources=event_budget315,
            )
            capsule_orders315_curve = {
                count: compress_event_horizon(
                    capsule_order314, satellite_orbit315, max_extra_sources=count,
                )
                for count in event_curve_counts
            }
            ranked_capsules = [tuple(turn_ids[i] for i in capsule) for capsule in capsule_order]
            ranked_capsules314 = [
                tuple(turn_ids[i] for i in capsule) for capsule in capsule_order314
            ]
            ranked_capsules315 = [
                tuple(turn_ids[i] for i in capsule) for capsule in capsule_order315
            ]
            metric.add(ranked_ids, gold)
            metric310.add(ranked_ids310, gold)
            metric312.add(ranked_capsules, gold)
            metric314.add(ranked_capsules314, gold)
            metric315.add(ranked_capsules315, gold)
            for count, curve_order in capsule_orders315_curve.items():
                metric315_curve[count].add(
                    [tuple(turn_ids[i] for i in capsule) for capsule in curve_order],
                    gold,
                )
            by_type[category].add(ranked_ids, gold)
            by_type310[category].add(ranked_ids310, gold)
            by_type312[category].add(ranked_capsules, gold)
            by_type314[category].add(ranked_capsules314, gold)
            by_type315[category].add(ranked_capsules315, gold)
            if category in {"1", "2", "3", "4"}:
                official_metric.add(ranked_ids, gold)
                official_metric310.add(ranked_ids310, gold)
                official_metric312.add(ranked_capsules, gold)
                official_metric314.add(ranked_capsules314, gold)
                official_metric315.add(ranked_capsules315, gold)
            context_chars.append(sum(len(texts[i]) for i in order[:10]))
            context_chars310.append(sum(len(texts[i]) for i in order310[:10]))
            context_chars312.append(sum(
                len(texts[i]) for i in {source for capsule in capsule_order for source in capsule}
            ))
            context_chars314.append(sum(
                len(texts[i]) for i in {
                    source for capsule in capsule_order314 for source in capsule
                }
            ))
            context_chars315.append(sum(
                len(texts[i]) for i in {
                    source for capsule in capsule_order315 for source in capsule
                }
            ))
            for count, curve_order in capsule_orders315_curve.items():
                context_chars315_curve[count].append(sum(
                    len(texts[i]) for i in {
                        source for capsule in curve_order for source in capsule
                    }
                ))

    return {
        "benchmark": "LoCoMo",
        "data_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "conversations": len(dataset),
        "conversation_slice": [max(0, start), stop if stop is not None else len(all_dataset)],
        "qa_total": total,
        "qa_with_evidence": total - no_evidence,
        "unit": "dialogue_turn",
        "umd315_config": {
            "event_extra_sources_ceiling": event_extra_sources,
            "adaptive_evidence_width": adaptive_event_width,
            "entropy_evidence_width": entropy_event_width,
            "event_budget_histogram": {
                str(count): event_budget_histogram[count]
                for count in sorted(event_budget_histogram)
            },
            "secondary_orbit_rerank": "0.46 force + 0.30 semantic + 0.14 lexical + 0.06 reciprocal rank + 0.04 new planet",
        },
        "umd39_cosmic_orbit": metric.result(),
        "umd310_cognitive_orbit": metric310.result(),
        "umd312_constellation_capsules": metric312.result(),
        "umd314_relative_gravity_bridges": metric314.result(),
        "umd315_event_lagrange_bridges": metric315.result(),
        "umd315_event_source_curve": {
            str(count): {
                "metrics": metric315_curve[count].result(),
                "mean_retrieved_characters": statistics.mean(
                    context_chars315_curve[count]
                ),
            }
            for count in event_curve_counts
        },
        "official_default_scope_categories_1_to_4": {
            "qa_total": official_qa_total,
            "retrieval_questions_with_evidence": official_metric.queries,
            "umd39_cosmic_orbit": official_metric.result(),
            "umd310_cognitive_orbit": official_metric310.result(),
            "umd312_constellation_capsules": official_metric312.result(),
            "umd314_relative_gravity_bridges": official_metric314.result(),
            "umd315_event_lagrange_bridges": official_metric315.result(),
        },
        "by_question_category": _metrics_by_type(by_type),
        "umd310_by_question_category": _metrics_by_type(by_type310),
        "umd312_by_question_category": _metrics_by_type(by_type312),
        "umd314_by_question_category": _metrics_by_type(by_type314),
        "umd315_by_question_category": _metrics_by_type(by_type315),
        "mean_retrieved_characters_at_10": statistics.mean(context_chars),
        "umd310_mean_retrieved_characters_at_10": statistics.mean(context_chars310),
        "umd312_mean_retrieved_characters_at_10_capsules": statistics.mean(context_chars312),
        "umd314_mean_retrieved_characters_at_10_bridges": statistics.mean(context_chars314),
        "umd315_mean_retrieved_characters_at_10_bridges": statistics.mean(context_chars315),
        "runtime_seconds": time.perf_counter() - started,
    }


def _long_force_order(
    item: dict[str, Any], encoder: FastEmbedEncoder, query_vector: np.ndarray,
) -> tuple[
    list[int], list[int], list[str], list[float], list[float], list[float],
    list[tuple[str, str] | None],
]:
    sessions = item["haystack_sessions"]
    dates = item["haystack_dates"]
    session_texts = [
        f"{dates[i]} " + " ".join(_turn_text(turn) for turn in session)
        for i, session in enumerate(sessions)
    ]
    session_index = BM25(session_texts)
    turn_texts: list[str] = []
    groups: list[int] = []
    session_turn_indices: list[list[int]] = [[] for _ in sessions]
    for session_number, session in enumerate(sessions):
        for turn in session:
            session_turn_indices[session_number].append(len(turn_texts))
            turn_texts.append(_turn_text(turn))
            groups.append(session_number)

    query = item["question"]
    aggregate_raw = session_index.scores(query)
    turn_raw = BM25(turn_texts).scores(query)
    best_turn = [0.0] * len(sessions)
    best_turn_index = [indices[0] for indices in session_turn_indices]
    for turn_index, score in enumerate(turn_raw):
        session_number = groups[turn_index]
        if score > best_turn[session_number]:
            best_turn[session_number] = score
            best_turn_index[session_number] = turn_index
    aggregate_order = ranking(aggregate_raw)
    moon_order = ranking(best_turn)
    aggregate_rr = reciprocal_ranks(aggregate_order)
    moon_rr = reciprocal_ranks(moon_order)
    base_order = ranking([
        aggregate_rr[i] + 0.82 * moon_rr[i] for i in range(len(sessions))
    ])

    # Preserve the validated first anchor and top-10 candidate orbit. Neural
    # work is bounded to nine query-specific representative turns.
    candidates = base_order[1:10]
    representative_texts = [turn_texts[best_turn_index[i]] for i in candidates]
    representative_vectors = encoder.encode_many(representative_texts)
    semantic_values = _dot_scores(query_vector, representative_vectors)
    semantic = [0.0] * len(sessions)
    for session_index_value, score in zip(candidates, semantic_values):
        semantic[session_index_value] = score
    lexical = unit_scores(best_turn)
    planetary = unit_scores(aggregate_raw)
    char_scores = BM25(session_texts, tokenizer=char_tokenize).scores(query)
    char = unit_scores(char_scores)
    entity = _entity_value_resonance(query, session_texts)
    temporal = _temporal_phase(query, dates)
    graph = [0.0] * len(sessions)
    for index in range(len(sessions)):
        neighbors = [planetary[index]]
        if index:
            neighbors.append(planetary[index - 1])
        if index + 1 < len(sessions):
            neighbors.append(planetary[index + 1])
        graph[index] = max(neighbors)
    force = _force(semantic, lexical, planetary, char, entity, temporal, graph)
    middle = sorted(candidates, key=lambda i: (-force[i], i))
    order39 = base_order[:1] + middle + base_order[10:]
    directive_meta: list[tuple[str, str] | None] = []
    for session in sessions:
        latest_metadata = None
        for turn in session:
            role = str(turn.get("role", turn.get("speaker", ""))).lower()
            metadata = _directive_metadata(_turn_text(turn), trusted=role in {"user", "human"})
            if metadata is not None:
                latest_metadata = metadata
        directive_meta.append(latest_metadata)
    order310 = _cognitive_order(
        query, session_texts, list(range(len(sessions))), dates, order39, force,
        semantic, lexical, directive_meta, top_k=10,
    )
    return order39, order310, session_texts, force, semantic, lexical, directive_meta


def run_longmemeval(
    path: Path, encoder: FastEmbedEncoder, *, event_extra_sources: int = 16,
) -> dict[str, Any]:
    started = time.perf_counter()
    dataset = json.loads(path.read_text(encoding="utf-8"))
    ks = (1, 5, 10, 20)
    metric = RecallAccumulator(ks)
    metric310 = RecallAccumulator(ks)
    metric312 = CapsuleRecallAccumulator(ks)
    metric314 = CapsuleRecallAccumulator(ks)
    metric315 = CapsuleRecallAccumulator(ks)
    all_metric = RecallAccumulator(ks)
    all_metric310 = RecallAccumulator(ks)
    all_metric312 = CapsuleRecallAccumulator(ks)
    all_metric314 = CapsuleRecallAccumulator(ks)
    all_metric315 = CapsuleRecallAccumulator(ks)
    by_type: dict[str, RecallAccumulator] = defaultdict(lambda: RecallAccumulator(ks))
    by_type310: dict[str, RecallAccumulator] = defaultdict(lambda: RecallAccumulator(ks))
    by_type312: dict[str, CapsuleRecallAccumulator] = defaultdict(lambda: CapsuleRecallAccumulator(ks))
    by_type314: dict[str, CapsuleRecallAccumulator] = defaultdict(lambda: CapsuleRecallAccumulator(ks))
    by_type315: dict[str, CapsuleRecallAccumulator] = defaultdict(lambda: CapsuleRecallAccumulator(ks))
    context_chars: list[int] = []
    context_chars310: list[int] = []
    context_chars312: list[int] = []
    context_chars314: list[int] = []
    context_chars315: list[int] = []
    abstention = missing = 0
    query_vectors = encoder.encode_many([str(item["question"]) for item in dataset])

    for item_number, item in enumerate(dataset, 1):
        is_abstention = item["question_id"].endswith("_abs")
        abstention += int(is_abstention)
        gold = [str(value) for value in item.get("answer_session_ids") or []]
        if not gold:
            missing += 1
            continue
        order, order310, session_texts, force, semantic, lexical, directive_meta = _long_force_order(
            item, encoder, query_vectors[item_number - 1]
        )
        ranked_ids = [str(item["haystack_session_ids"][i]) for i in order]
        ranked_ids310 = [str(item["haystack_session_ids"][i]) for i in order310]
        # Session IDs are the atomic sources in LongMemEval. Their stored order
        # is fixed before the question and serves as a bounded episode chain.
        capsules, capsules_by_source = _benchmark_capsules([0] * len(session_texts))
        capsule_order = _anchored_capsule_order(
            item["question"], order, force, capsules, capsules_by_source, top_k=10,
        )
        ranked_capsules = [
            tuple(str(item["haystack_session_ids"][i]) for i in capsule)
            for capsule in capsule_order
        ]
        session_groups = list(range(len(session_texts)))
        order314, force314 = relative_gravity_order(
            item["question"], session_texts, session_groups, item["haystack_dates"],
            order, force, semantic, lexical, directive_meta, top_k=10,
        )
        bridge314 = transient_directive_bridge(
            item["question"], session_texts, order314, directive_meta,
        )
        capsule_order314 = fuse_transient_bridge(capsule_order, bridge314)
        ranked_capsules314 = [
            tuple(str(item["haystack_session_ids"][i]) for i in capsule)
            for capsule in capsule_order314
        ]
        # LongMemEval sessions are the atomic provenance units.  UMD 3.15 keeps
        # every UMD 3.14 capsule intact and stripes a bounded, query-ranked
        # secondary session orbit across the same ten capsules.
        session_groups = list(range(len(session_texts)))
        satellite_orbit315 = rank_event_satellites(
            order314, capsule_order314, session_groups,
            force314, semantic, lexical,
        )
        capsule_order315 = compress_event_horizon(
            capsule_order314, satellite_orbit315,
            max_extra_sources=event_extra_sources,
        )
        ranked_capsules315 = [
            tuple(str(item["haystack_session_ids"][i]) for i in capsule)
            for capsule in capsule_order315
        ]
        all_metric.add(ranked_ids, gold)
        all_metric310.add(ranked_ids310, gold)
        all_metric312.add(ranked_capsules, gold)
        all_metric314.add(ranked_capsules314, gold)
        all_metric315.add(ranked_capsules315, gold)
        if not is_abstention:
            metric.add(ranked_ids, gold)
            metric310.add(ranked_ids310, gold)
            metric312.add(ranked_capsules, gold)
            metric314.add(ranked_capsules314, gold)
            metric315.add(ranked_capsules315, gold)
            by_type[str(item.get("question_type", "unknown"))].add(ranked_ids, gold)
            by_type310[str(item.get("question_type", "unknown"))].add(ranked_ids310, gold)
            by_type312[str(item.get("question_type", "unknown"))].add(ranked_capsules, gold)
            by_type314[str(item.get("question_type", "unknown"))].add(ranked_capsules314, gold)
            by_type315[str(item.get("question_type", "unknown"))].add(ranked_capsules315, gold)
        context_chars.append(sum(len(session_texts[i]) for i in order[:10]))
        context_chars310.append(sum(len(session_texts[i]) for i in order310[:10]))
        context_chars312.append(sum(
            len(session_texts[i]) for i in {source for capsule in capsule_order for source in capsule}
        ))
        context_chars314.append(sum(
            len(session_texts[i]) for i in {
                source for capsule in capsule_order314 for source in capsule
            }
        ))
        context_chars315.append(sum(
            len(session_texts[i]) for i in {
                source for capsule in capsule_order315 for source in capsule
            }
        ))
        if item_number % 25 == 0:
            print(f"LongMemEval UMD310 progress: {item_number}/{len(dataset)}", flush=True)

    return {
        "benchmark": "LongMemEval_S_cleaned",
        "data_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "questions": len(dataset),
        "answerable_questions": len(dataset) - abstention,
        "abstention_questions": abstention,
        "questions_without_evidence_labels": missing,
        "unit": "session",
        "umd315_config": {
            "event_extra_sources": event_extra_sources,
            "secondary_orbit_rerank": "0.46 force + 0.30 semantic + 0.14 lexical + 0.06 reciprocal rank + 0.04 new planet",
        },
        "umd39_cosmic_orbit": metric.result(),
        "umd310_cognitive_orbit": metric310.result(),
        "umd312_constellation_capsules": metric312.result(),
        "umd314_relative_gravity_bridges": metric314.result(),
        "umd315_event_lagrange_bridges": metric315.result(),
        "diagnostic_including_abstention_with_labels": all_metric.result(),
        "umd310_diagnostic_including_abstention_with_labels": all_metric310.result(),
        "umd312_diagnostic_including_abstention_with_labels": all_metric312.result(),
        "umd314_diagnostic_including_abstention_with_labels": all_metric314.result(),
        "umd315_diagnostic_including_abstention_with_labels": all_metric315.result(),
        "by_question_type": _metrics_by_type(by_type),
        "umd310_by_question_type": _metrics_by_type(by_type310),
        "umd312_by_question_type": _metrics_by_type(by_type312),
        "umd314_by_question_type": _metrics_by_type(by_type314),
        "umd315_by_question_type": _metrics_by_type(by_type315),
        "mean_retrieved_characters_at_10": statistics.mean(context_chars),
        "umd310_mean_retrieved_characters_at_10": statistics.mean(context_chars310),
        "umd312_mean_retrieved_characters_at_10_capsules": statistics.mean(context_chars312),
        "umd314_mean_retrieved_characters_at_10_bridges": statistics.mean(context_chars314),
        "umd315_mean_retrieved_characters_at_10_bridges": statistics.mean(context_chars315),
        "runtime_seconds": time.perf_counter() - started,
    }


def _flatten_source_ids(value: Any) -> set[str]:
    output: set[str] = set()
    if isinstance(value, dict):
        for child in value.values():
            output.update(_flatten_source_ids(child))
    elif isinstance(value, (list, tuple, set)):
        for child in value:
            output.update(_flatten_source_ids(child))
    elif value is not None:
        output.add(str(value))
    return output


def _beam_batches(chat: Any) -> list[list[dict[str, Any]]]:
    if not isinstance(chat, list):
        return []
    if chat and isinstance(chat[0], list):
        return chat
    if chat and isinstance(chat[0], dict) and "turns" in chat[0]:
        return [batch.get("turns", []) for batch in chat]
    return []


def run_beam(
    path: Path, encoder: FastEmbedEncoder, *, start: int = 0, stop: int | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    all_rows = pq.read_table(path).to_pylist()
    rows = all_rows[max(0, start):stop]
    ks = (1, 5, 10, 20, 50, 200)
    flat = RecallAccumulator(ks)
    metric = RecallAccumulator(ks)
    metric310 = RecallAccumulator(ks)
    metric312 = CapsuleRecallAccumulator(ks)
    metric314 = CapsuleRecallAccumulator(ks)
    flat_by_type: dict[str, RecallAccumulator] = defaultdict(lambda: RecallAccumulator(ks))
    by_type: dict[str, RecallAccumulator] = defaultdict(lambda: RecallAccumulator(ks))
    by_type310: dict[str, RecallAccumulator] = defaultdict(lambda: RecallAccumulator(ks))
    by_type312: dict[str, CapsuleRecallAccumulator] = defaultdict(lambda: CapsuleRecallAccumulator(ks))
    by_type314: dict[str, CapsuleRecallAccumulator] = defaultdict(lambda: CapsuleRecallAccumulator(ks))
    total = abstention = missing = 0
    turn_counts: list[int] = []
    context_chars: list[int] = []
    context_chars310: list[int] = []
    context_chars312: list[int] = []
    context_chars314: list[int] = []

    for row_number, row in enumerate(rows, 1):
        batches = _beam_batches(row["chat"])
        turns = [turn for batch in batches for turn in batch if isinstance(turn, dict)]
        if not turns:
            continue
        texts = [f'{turn.get("role", "")}: {turn.get("content", "")}' for turn in turns]
        ids = [str(turn.get("id", turn.get("index", index))) for index, turn in enumerate(turns)]
        groups: list[int] = []
        session_indices: list[list[int]] = []
        cursor = 0
        for batch_index, batch in enumerate(batches):
            batch_turns = [turn for turn in batch if isinstance(turn, dict)]
            indices = list(range(cursor, cursor + len(batch_turns)))
            session_indices.append(indices)
            groups.extend([batch_index] * len(batch_turns))
            cursor += len(batch_turns)
        dates = [
            next((str(turn.get("time_anchor")) for turn in batch if turn.get("time_anchor")), "")
            for batch in batches
        ]
        session_texts = [
            f"{dates[i]} " + " ".join(texts[j] for j in indices)
            for i, indices in enumerate(session_indices)
        ]
        lexical_index = BM25(texts)
        char_index = BM25(texts, tokenizer=char_tokenize)
        planet_index = BM25(session_texts)
        vectors = encoder.encode_many(texts)
        directive_meta = [
            _directive_metadata(
                str(turn.get("content", "")),
                trusted=str(turn.get("role", "")).lower() in {"user", "human"},
            )
            for turn in turns
        ]
        capsules, capsules_by_source = _benchmark_capsules(groups)
        turn_counts.append(len(turns))

        questions = ast.literal_eval(row["probing_questions"])
        ordered_questions = [
            question_data
            for question_type in BEAM_TYPES
            for question_data in questions.get(question_type, [])
        ]
        question_vectors = encoder.encode_many([
            str(question_data.get("question") or question_data.get("question_text") or "")
            for question_data in ordered_questions
        ])
        question_vector_number = 0
        for question_type in BEAM_TYPES:
            for question_data in questions.get(question_type, []):
                qv = question_vectors[question_vector_number]
                question_vector_number += 1
                total += 1
                if question_type == "abstention":
                    abstention += 1
                    continue
                gold = _flatten_source_ids(question_data.get("source_chat_ids"))
                if not gold:
                    missing += 1
                    continue
                query = str(question_data.get("question") or question_data.get("question_text") or "")
                lexical_raw = lexical_index.scores(query)
                lexical_order = ranking(lexical_raw)
                flat_ids = [ids[i] for i in lexical_order]
                flat.add(flat_ids, gold)
                flat_by_type[question_type].add(flat_ids, gold)

                planet_raw = planet_index.scores(query)
                planet_order = ranking(planet_raw)
                lexical_rr = reciprocal_ranks(lexical_order)
                planet_rr = reciprocal_ranks(planet_order)
                seed = ranking([
                    lexical_rr[i] + 0.72 * planet_rr[groups[i]]
                    for i in range(len(texts))
                ])
                orbit = _orbit_expand(seed, groups)
                protected = orbit[:20]
                semantic = _dot_scores(qv, vectors)
                lexical = unit_scores(lexical_raw)
                planet_unit = unit_scores(planet_raw)
                planetary = [planet_unit[groups[i]] for i in range(len(texts))]
                char = unit_scores(char_index.scores(query))
                entity = _entity_value_resonance(query, texts)
                temporal_sessions = _temporal_phase(query, dates)
                temporal = [temporal_sessions[groups[i]] for i in range(len(texts))]
                graph = _graph_flux(lexical, groups)
                force = _force(semantic, lexical, planetary, char, entity, temporal, graph)
                head = sorted(protected, key=lambda i: (-force[i], i))
                protected_set = set(protected)
                order = head + [i for i in orbit if i not in protected_set]
                ranked_ids = [ids[i] for i in order]
                order310 = _cognitive_order(
                    query, texts, groups, dates, order, force, semantic, lexical,
                    directive_meta, top_k=20,
                )
                ranked_ids310 = [ids[i] for i in order310]
                capsule_order = _anchored_capsule_order(
                    query, order, force, capsules, capsules_by_source, top_k=20,
                )
                order314, force314 = relative_gravity_order(
                    query, texts, groups, dates, order, force, semantic, lexical,
                    directive_meta, top_k=20,
                )
                bridge314 = transient_directive_bridge(
                    query, texts, order314, directive_meta,
                )
                capsule_order314 = fuse_transient_bridge(capsule_order, bridge314)
                ranked_capsules = [tuple(ids[i] for i in capsule) for capsule in capsule_order]
                ranked_capsules314 = [
                    tuple(ids[i] for i in capsule) for capsule in capsule_order314
                ]
                metric.add(ranked_ids, gold)
                metric310.add(ranked_ids310, gold)
                metric312.add(ranked_capsules, gold)
                metric314.add(ranked_capsules314, gold)
                by_type[question_type].add(ranked_ids, gold)
                by_type310[question_type].add(ranked_ids310, gold)
                by_type312[question_type].add(ranked_capsules, gold)
                by_type314[question_type].add(ranked_capsules314, gold)
                context_chars.append(sum(len(texts[i]) for i in order[:20]))
                context_chars310.append(sum(len(texts[i]) for i in order310[:20]))
                context_chars312.append(sum(
                    len(texts[i]) for i in {
                        source for capsule in capsule_order[:20] for source in capsule
                    }
                ))
                context_chars314.append(sum(
                    len(texts[i]) for i in {
                        source for capsule in capsule_order314[:20] for source in capsule
                    }
                ))
        print(f"BEAM 100K progress: {row_number}/{len(rows)}", flush=True)

    return {
        "benchmark": "BEAM_100K",
        "data_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "conversations": len(rows),
        "conversation_slice": [max(0, start), stop if stop is not None else len(all_rows)],
        "questions": total,
        "abstention_questions_without_positive_evidence": abstention,
        "retrieval_questions": metric.queries,
        "questions_missing_machine_readable_evidence": missing,
        "unit": "chat_turn",
        "flat_bm25": flat.result(),
        "umd39_cosmic_orbit": metric.result(),
        "umd310_cognitive_orbit": metric310.result(),
        "umd312_constellation_capsules": metric312.result(),
        "umd314_relative_gravity_bridges": metric314.result(),
        "flat_by_question_type": _metrics_by_type(flat_by_type),
        "umd39_by_question_type": _metrics_by_type(by_type),
        "umd310_by_question_type": _metrics_by_type(by_type310),
        "umd312_by_question_type": _metrics_by_type(by_type312),
        "umd314_by_question_type": _metrics_by_type(by_type314),
        "turns_per_conversation": {
            "min": min(turn_counts), "mean": statistics.mean(turn_counts), "max": max(turn_counts),
        },
        "mean_retrieved_characters_at_20": statistics.mean(context_chars),
        "umd310_mean_retrieved_characters_at_20": statistics.mean(context_chars310),
        "umd312_mean_retrieved_characters_at_20_capsules": statistics.mean(context_chars312),
        "umd314_mean_retrieved_characters_at_20_bridges": statistics.mean(context_chars314),
        "runtime_seconds": time.perf_counter() - started,
    }


def _load_previous() -> dict[str, Any]:
    path = RESULTS / "public_benchmarks.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _report(results: dict[str, Any]) -> str:
    lines = [
        "# UMD 3.15 LoCoMo / LongMemEval / BEAM Retrieval Results", "",
        "This is a gold-label-blind evidence-retrieval run. It is not an end-to-end LLM answer/judge score.", "",
        "## Frozen formula", "",
        "`F39 = 0.34*semantic + 0.22*lexical + 0.16*planet + 0.08*char + 0.08*entity/value + 0.06*time + 0.06*graph`", "",
        "`Fc = 0.30*semantic + 0.18*lexical + 0.24*directive + 0.12*episode + 0.08*time + 0.08*authority`; "
        "episodic coverage adds `+0.16 new planet +0.10 new time +0.08 new entity -0.12 redundancy`.", "",
        "`R12 = 0.26*semantic + 0.18*lexical + 0.14*entity + 0.12*time + 0.10*contradiction + 0.10*directive + 0.10*provenance`; "
        "each capsule contains at most four query-independent source IDs.", "",
        "`B14(q) = C1 union {latest applicable trusted directive}; Ck'=Ck for k>1`; "
        "the transient bridge never displaces a stable source.", "",
        "`H15 = 0.46*force + 0.30*semantic + 0.14*lexical + 0.06*reciprocal-rank + 0.04*new-planet`; "
        "sixteen secondary event satellites are striped across ten stable capsules.", "",
        "The UMD 3.9 first anchor is protected; UMD 3.10 adds trusted latest-directive routing and episodic coverage.", "",
    ]
    for key in ("locomo", "longmemeval", "beam"):
        if key not in results:
            continue
        item = results[key]
        metric39 = item["umd39_cosmic_orbit"]
        metric310 = item["umd310_cognitive_orbit"]
        metric312 = item["umd312_constellation_capsules"]
        metric314 = item["umd314_relative_gravity_bridges"]
        metric315 = item.get("umd315_event_lagrange_bridges")
        lines.extend([
            f"## {item['benchmark']}", "",
            f"- Evaluated retrieval questions: {metric310['evaluated_queries']}",
            f"- Runtime: {item['runtime_seconds']:.2f} s", "",
            "| Adapter | MRR | Any R@1 | Any R@5 | Any R@10 | Full R@10 | Micro R@10 |",
            "|---|---:|---:|---:|---:|---:|---:|",
            f"| UMD 3.9 | {metric39['mrr']:.4f} | {metric39['any_evidence_recall']['1']:.4f} | {metric39['any_evidence_recall']['5']:.4f} | {metric39['any_evidence_recall']['10']:.4f} | {metric39['full_evidence_recall']['10']:.4f} | {metric39['micro_evidence_recall']['10']:.4f} |",
            f"| UMD 3.10 | {metric310['mrr']:.4f} | {metric310['any_evidence_recall']['1']:.4f} | {metric310['any_evidence_recall']['5']:.4f} | {metric310['any_evidence_recall']['10']:.4f} | {metric310['full_evidence_recall']['10']:.4f} | {metric310['micro_evidence_recall']['10']:.4f} |",
            f"| UMD 3.12 capsules* | {metric312['mrr']:.4f} | {metric312['any_evidence_recall']['1']:.4f} | {metric312['any_evidence_recall']['5']:.4f} | {metric312['any_evidence_recall']['10']:.4f} | {metric312['full_evidence_recall']['10']:.4f} | {metric312['micro_evidence_recall']['10']:.4f} |",
            f"| UMD 3.14 bridges* | {metric314['mrr']:.4f} | {metric314['any_evidence_recall']['1']:.4f} | {metric314['any_evidence_recall']['5']:.4f} | {metric314['any_evidence_recall']['10']:.4f} | {metric314['full_evidence_recall']['10']:.4f} | {metric314['micro_evidence_recall']['10']:.4f} |",
        ])
        if metric315 is not None:
            lines.append(
                f"| UMD 3.15 event bridges* | {metric315['mrr']:.4f} | {metric315['any_evidence_recall']['1']:.4f} | {metric315['any_evidence_recall']['5']:.4f} | {metric315['any_evidence_recall']['10']:.4f} | {metric315['full_evidence_recall']['10']:.4f} | {metric315['micro_evidence_recall']['10']:.4f} |"
            )
        lines.extend([
            "",
            "`*` Expanded adapters preserve immutable provenance; UMD 3.15 adds bounded event satellites without displacing stable sources.", "",
        ])
        if key == "locomo" and "official_default_scope_categories_1_to_4" in item:
            scope = item["official_default_scope_categories_1_to_4"]
            scoped39 = scope["umd39_cosmic_orbit"]
            scoped310 = scope["umd310_cognitive_orbit"]
            scoped312 = scope["umd312_constellation_capsules"]
            scoped314 = scope["umd314_relative_gravity_bridges"]
            scoped315 = scope.get("umd315_event_lagrange_bridges")
            lines.extend([
                f"Official runner default scope (categories 1-4): {scope['qa_total']} total questions, "
                f"{scope['retrieval_questions_with_evidence']} with evidence; UMD 3.9/3.10/3.12 MRR="
                f"{scoped39['mrr']:.4f}/{scoped310['mrr']:.4f}/{scoped312['mrr']:.4f}, Any R@10="
                f"{scoped39['any_evidence_recall']['10']:.4f}/{scoped310['any_evidence_recall']['10']:.4f}/"
                f"{scoped312['any_evidence_recall']['10']:.4f}, "
                f"Full R@10={scoped39['full_evidence_recall']['10']:.4f}/"
                f"{scoped310['full_evidence_recall']['10']:.4f}/"
                f"{scoped312['full_evidence_recall']['10']:.4f}.", "",
                f"UMD 3.14 official-scope MRR={scoped314['mrr']:.4f}, Any R@10="
                f"{scoped314['any_evidence_recall']['10']:.4f}, Full R@10="
                f"{scoped314['full_evidence_recall']['10']:.4f}.", "",
            ])
            if scoped315 is not None:
                lines.extend([
                    f"UMD 3.15 official-scope MRR={scoped315['mrr']:.4f}, Any R@10="
                    f"{scoped315['any_evidence_recall']['10']:.4f}, Full R@10="
                    f"{scoped315['full_evidence_recall']['10']:.4f}.", "",
                ])
    lines.extend([
        "## Verdict", "",
        "UMD 3.12 through 3.15 use multi-source provenance-bearing retrieval units. "
        "Atomic controls, expanded-source scores, and retrieved character cost must be reported side by side.", "",
        "## Limits", "",
        "- Answer accuracy, rubric pass rate, abstention correctness, and BEAM Kendall tau require an answer model and judge model.",
        "- Public raw-text adapters omit production authority and learned-utility signals because those fields are uniform or absent.",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=("all", "locomo", "longmemeval", "beam"), default="all")
    parser.add_argument("--locomo", type=Path, default=DATA / "locomo10.json")
    parser.add_argument("--locomo-start", type=int, default=0)
    parser.add_argument("--locomo-stop", type=int)
    parser.add_argument("--locomo-event-extra", type=int, default=16)
    parser.add_argument("--locomo-event-adaptive", action="store_true")
    parser.add_argument("--locomo-event-entropy", action="store_true")
    parser.add_argument("--longmemeval", type=Path, default=DATA / "longmemeval_s_cleaned.json")
    parser.add_argument("--longmemeval-event-extra", type=int, default=16)
    parser.add_argument(
        "--beam", type=Path,
        default=DATA / "beam_hf" / "data" / "100K-00000-of-00001.parquet",
    )
    parser.add_argument("--beam-start", type=int, default=0)
    parser.add_argument("--beam-stop", type=int)
    parser.add_argument("--output-stem", default="umd315_benchmarks")
    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS / f"{args.output_stem}.json"
    results = {}
    if output_path.exists() and args.benchmark != "all":
        results = json.loads(output_path.read_text(encoding="utf-8"))
        results = {key: value for key, value in results.items() if key in {"locomo", "longmemeval", "beam"}}

    started = time.perf_counter()
    encoder = FastEmbedEncoder(cache_dir=MODEL_CACHE, batch_size=64, cache_size=8192, threads=4)
    if args.benchmark in {"all", "locomo"}:
        results["locomo"] = run_locomo(
            args.locomo, encoder, start=args.locomo_start, stop=args.locomo_stop,
            event_extra_sources=args.locomo_event_extra,
            adaptive_event_width=args.locomo_event_adaptive,
            entropy_event_width=args.locomo_event_entropy,
        )
    if args.benchmark in {"all", "longmemeval"}:
        results["longmemeval"] = run_longmemeval(
            args.longmemeval, encoder,
            event_extra_sources=args.longmemeval_event_extra,
        )
    if args.benchmark in {"all", "beam"}:
        results["beam"] = run_beam(
            args.beam, encoder, start=args.beam_start, stop=args.beam_stop,
        )
    results["run_metadata"] = {
        "adapter": "UMD 3.15 event-Lagrange bridges (with UMD 3.9, 3.10, 3.12 and 3.14 controls)",
        "formula": "stable capsule orbit union bounded neural event satellites",
        "gold_or_answer_used_for_ranking": False,
        "paid_api_calls": 0,
        "answer_model": None,
        "judge_model": None,
        "encoder": encoder.metadata(),
        "total_runtime_seconds": time.perf_counter() - started,
        "peak_working_set_mib": peak_working_set_mib(),
        "previous_retrieval_baselines": _load_previous(),
    }
    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    (RESULTS / f"{args.output_stem.upper()}.md").write_text(_report(results), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
