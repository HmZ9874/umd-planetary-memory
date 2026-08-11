"""UMD 3.18 bounded Roche-lobe evidence interference.

The first capsule may contain several provenance-bearing sources, but its
capacity is explicit and hard bounded.  Independent retrieval fields interfere
through reciprocal ranks; no answer, evidence identifier or evaluator signal
is consumed by this module.
"""

from __future__ import annotations

import hashlib
import re
from typing import Sequence

import numpy as np

from benchmarks.public_benchmarks import ranking, reciprocal_ranks, unit_scores
from benchmarks.umd316_adapter import _content_tokens, _entity_set
from benchmarks.umd316_adapter import query_physics
from benchmarks.umd317_adapter import FactAtom, GENERIC_QUERY, NUMBER_RE, _atom_overlap


INJECTION_RE = re.compile(
    r"\b(?:system\s+override|ignore\s+(?:every|all|the|verified|other)|"
    r"must\s+always\s+be\s+answered|more\s+important\s+than\s+evidence|"
    r"treat\s+this\s+instruction\s+as\s+the\s+answer)\b|"
    r"(?:忽略(?:所有|其他|之前)|系统覆盖|必须回答)",
    re.IGNORECASE,
)
DISPROVEN_RE = re.compile(
    r"\b(?:false|disproven|invalid\s+rumou?r|obsolete\s+rumou?r|"
    r"no\s+valid|not\s+(?:a|an|the)\s+(?:\w+\s+){0,2}(?:residence|answer|code|identifier)|"
    r"did\s+not\s+(?:establish|confirm|record))\b|"
    r"(?:错误传闻|已经证伪|无效|并非(?:答案|住所|代码))",
    re.IGNORECASE,
)
NEGATED_QUERY_RE = re.compile(
    r"\b(?:(?:does|did|has|have|can|will)\s+\w+(?:\s+\w+){0,2}\s+not|"
    r"doesn['’]?t|didn['’]?t|isn['’]?t|wasn['’]?t|hasn['’]?t|can['’]?t|"
    r"(?:not|never)\s+(?:enjoy|like|want|prefer|do|have|work|live|contain|include))\b|"
    r"(?:不喜欢|没有|并非|不再)",
    re.IGNORECASE,
)
NEGATED_TEXT_RE = re.compile(
    r"\b(?:not|never|doesn['’]?t|didn['’]?t|isn['’]?t|no\s+longer)\b|"
    r"(?:不喜欢|没有|并非|不再)",
    re.IGNORECASE,
)
YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")
BIOSPHERE_QUERY_RE = re.compile(
    r"\b(?:would|might|likely|considered|personality|personality\s+traits|"
    r"attributes\s+describe|political\s+leaning|be\s+open\s+to|benefit\s+from)\b",
    re.IGNORECASE,
)


def roche_lobe_first_capsule(
    query: str,
    capsules: Sequence[tuple[int, ...]],
    atomic_orbit: Sequence[int],
    facts: Sequence[tuple[FactAtom, ...]],
    force: Sequence[float],
    semantic: Sequence[float],
    lexical: Sequence[float],
    char: Sequence[float],
    entity: Sequence[float],
    temporal: Sequence[float],
    groups: Sequence[int] | None = None,
    cross: Sequence[float] | None = None,
    learned_ranker: object | None = None,
    diagnostics: dict | None = None,
    *,
    source_budget: int = 10,
    candidate_limit: int = 512,
    matter_v320: bool = False,
    matter_v321: bool = False,
    matter_v322: bool = False,
    matter_v323: bool = False,
    neighbor_budget: int = 0,
    source_texts: Sequence[str] | None = None,
    source_dates: Sequence[str] | None = None,
) -> list[tuple[int, ...]]:
    """Add high-interference sources to L1 under a strict source budget.

    The heuristic law only fills unused capacity.  A learned field may exchange
    L1 sources, but displaced provenance is re-homed in later capsules.  The
    budget is over unique original sources, not compressed facts or summaries.
    """
    matter_v323 = bool(matter_v323)
    matter_v322 = bool(matter_v322 or matter_v323)
    matter_v321 = bool(matter_v321 or matter_v322)
    matter_v320 = bool(matter_v320 or matter_v321)
    if not capsules or source_budget <= 0:
        return list(capsules)
    first = list(dict.fromkeys(capsules[0]))
    if len(first) >= source_budget and learned_ranker is None:
        return list(capsules)
    candidates = list(dict.fromkeys(atomic_orbit[:max(source_budget, candidate_limit)]))
    if not candidates:
        return list(capsules)

    fields = [force, semantic, lexical, char, entity, temporal]
    if cross is not None:
        fields.append(cross)
    rr_fields: list[list[float]] = []
    for field in fields:
        order = ranking([field[source] for source in candidates])
        local_rr = reciprocal_ranks(order)
        rr_fields.append([local_rr[pos] for pos in range(len(candidates))])

    force_unit = unit_scores([force[source] for source in candidates])
    semantic_unit = unit_scores([semantic[source] for source in candidates])
    lexical_unit = unit_scores([lexical[source] for source in candidates])
    char_unit = unit_scores([char[source] for source in candidates])
    cross_unit = (
        unit_scores([cross[source] for source in candidates])
        if cross is not None else [0.0] * len(candidates)
    )
    query_tokens = _content_tokens(query) - GENERIC_QUERY
    query_entities = _entity_set(query)
    query_numbers = {value.casefold() for value in NUMBER_RE.findall(query)}
    orbit_rank = {source: rank for rank, source in enumerate(atomic_orbit)}

    scores: dict[int, float] = {}
    feature_rows: list[list[float]] = []
    mode = query_physics(query)
    first_word = query.strip().split(maxsplit=1)[0].casefold() if query.strip() else ""
    question_words = ("what", "when", "where", "who", "why", "how", "which", "would")
    for pos, source in enumerate(candidates):
        fact_peak = max(
            (_atom_overlap(atom, query_tokens, query_entities, query_numbers) for atom in facts[source]),
            default=0.0,
        )
        interference = (
            0.28 * rr_fields[0][pos] + 0.24 * rr_fields[1][pos]
            + 0.15 * rr_fields[2][pos] + 0.09 * rr_fields[3][pos]
            + 0.13 * rr_fields[4][pos] + 0.11 * rr_fields[5][pos]
        )
        amplitude = (
            0.25 * force_unit[pos] + 0.22 * semantic_unit[pos]
            + 0.13 * lexical_unit[pos] + 0.08 * char_unit[pos]
            + 0.20 * fact_peak
            + 0.12 / (1.0 + orbit_rank.get(source, len(atomic_orbit)))
        )
        scores[source] = 0.58 * interference + 0.42 * amplitude
        if cross is not None:
            # A cross encoder observes the joint question-evidence system and
            # therefore acts as a two-body potential rather than another
            # independent embedding field.
            scores[source] = 0.46 * scores[source] + 0.54 * rr_fields[6][pos]
        feature_rows.append([
            force_unit[pos], semantic_unit[pos], lexical_unit[pos], char_unit[pos],
            float(entity[source]), float(temporal[source]),
            rr_fields[0][pos], rr_fields[1][pos], rr_fields[2][pos],
            rr_fields[3][pos], rr_fields[4][pos], rr_fields[5][pos],
            fact_peak,
            1.0 / (1.0 + orbit_rank.get(source, len(atomic_orbit))),
            scores[source],
            float(mode.broad_set), float(mode.current_state), float(mode.historical),
            *(float(first_word == word) for word in question_words),
            min(1.0, len(set().union(*(atom.tokens for atom in facts[source]))) / 80.0),
        ])
        if matter_v320:
            neighbors = [source]
            if groups is not None:
                if source > 0 and groups[source - 1] == groups[source]:
                    neighbors.append(source - 1)
                if source + 1 < len(groups) and groups[source + 1] == groups[source]:
                    neighbors.append(source + 1)
            neighbor_fact = max((
                _atom_overlap(atom, query_tokens, query_entities, query_numbers)
                for neighbor in neighbors for atom in facts[neighbor]
            ), default=0.0)
            # New matter: companion dark mass, chronon coupling and field
            # coherence. These are still query/source-only observables.
            feature_rows[-1].extend([
                max(float(force[neighbor]) for neighbor in neighbors),
                max(float(semantic[neighbor]) for neighbor in neighbors),
                max(float(lexical[neighbor]) for neighbor in neighbors),
                max(float(char[neighbor]) for neighbor in neighbors),
                neighbor_fact,
                float(temporal[source]) * float(semantic[source]),
                min(semantic_unit[pos], lexical_unit[pos]),
            ])
        if matter_v321:
            # Learned two-body meson observables. The cross encoder sees the
            # question and source jointly, while LambdaRank decides when that
            # expensive interaction is trustworthy.
            cross_rr = rr_fields[6][pos] if cross is not None else 0.0
            feature_rows[-1].extend([
                cross_unit[pos], cross_rr,
                min(cross_unit[pos], semantic_unit[pos]),
            ])

    if diagnostics is not None:
        diagnostics.clear()
        diagnostics.update({
            "sources": list(candidates),
            "features": [list(row) for row in feature_rows],
            "heuristic_scores": [scores[source] for source in candidates],
        })
    if learned_ranker is not None and feature_rows:
        probabilities = learned_ranker.predict_proba(np.asarray(feature_rows, dtype=np.float32))[:, 1]
        learned_unit = unit_scores([float(value) for value in probabilities])
        for pos, source in enumerate(candidates):
            scores[source] = 0.88 * learned_unit[pos] + 0.12 * scores[source]

    unsafe_reasons: dict[int, list[str]] = {}
    if matter_v322 and source_texts is not None:
        negative_query = bool(NEGATED_QUERY_RE.search(query))
        biosphere_query = bool(BIOSPHERE_QUERY_RE.search(query))
        causal_query = bool(re.match(r"\s*why\b", query, re.IGNORECASE))
        requires_literal_polarity = negative_query and not (biosphere_query or causal_query)
        asks_assertion_status = bool(re.search(r"\b(?:rumou?r|false|invalid|claim)\b", query, re.I))
        query_years = set(YEAR_RE.findall(query))
        entity_latest_date: dict[str, str] = {}
        if source_dates is not None:
            for source in candidates:
                date = source_dates[source] if source < len(source_dates) else ""
                for entity_name in _entity_set(source_texts[source]) & query_entities:
                    entity_latest_date[entity_name] = max(entity_latest_date.get(entity_name, ""), date)
        for source in candidates:
            text = source_texts[source]
            reasons: list[str] = []
            if INJECTION_RE.search(text):
                reasons.append("instruction_injection")
            if not asks_assertion_status and DISPROVEN_RE.search(text):
                reasons.append("explicitly_disproven")
            query_entities = _entity_set(query)
            source_entities = _entity_set(text)
            same_entity = bool(query_entities & source_entities)
            if requires_literal_polarity and same_entity and not NEGATED_TEXT_RE.search(text):
                reasons.append("polarity_mismatch")
            source_years = set(YEAR_RE.findall(text))
            source_date = (
                source_dates[source] if source_dates is not None and source < len(source_dates) else ""
            )
            date_years = set(YEAR_RE.findall(source_date))
            if (
                query_years and same_entity and source_years
                and query_years.isdisjoint(source_years)
            ):
                reasons.append("historical_year_mismatch")
            if (
                mode.current_state and not biosphere_query and same_entity and source_date
                and any(source_date < entity_latest_date.get(name, source_date) for name in query_entities & source_entities)
            ):
                reasons.append("stale_version")
            if reasons:
                unsafe_reasons[source] = reasons

    conserved_shadow: list[int] = []
    if learned_ranker is not None and hasattr(learned_ranker, "conservation_scores"):
        shadow_score = learned_ranker.conservation_scores(
            np.asarray(feature_rows, dtype=np.float32),
        )
        shadow_order = sorted(
            range(len(candidates)),
            key=lambda pos: (-float(shadow_score[pos]), pos),
        )
        conserved_shadow = [candidates[pos] for pos in shadow_order[:source_budget]]

    output = list(capsules)
    ranked = sorted(
        candidates,
        key=lambda source: (
            bool(matter_v322 and source in unsafe_reasons),
            -scores[source],
            orbit_rank.get(source, len(atomic_orbit)),
            (
                hashlib.blake2s(source_texts[source].encode("utf-8"), digest_size=6).hexdigest()
                if matter_v322 and source_texts is not None else str(source)
            ),
        ),
    )
    diversified_ranked: list[int] | None = None
    if matter_v323 and BIOSPHERE_QUERY_RE.search(query) and source_texts is not None:
        # Biosphere orbit: inferential questions need a longitudinal sample of
        # the subject's behaviour. Keep two thirds of the learned gravity field
        # and reserve the remainder for the best matching subject utterance in
        # distinct sessions. No benchmark category or gold label is observed.
        safe = [source for source in ranked if source not in unsafe_reasons]
        query_folded = query.casefold()
        subject_sources = [
            source for source in safe
            if source_texts[source].split(":", 1)[0].strip().casefold() in query_folded
        ]
        group_best: dict[int, int] = {}
        for source in subject_sources:
            group = groups[source] if groups is not None else source
            incumbent = group_best.get(group)
            if incumbent is None or scores[source] > scores[incumbent]:
                group_best[group] = source
        biosphere = sorted(
            group_best.values(),
            key=lambda source: (-scores[source], orbit_rank.get(source, len(atomic_orbit))),
        )
        biosphere.extend(source for source in subject_sources if source not in set(biosphere))
        learned_quota = max(1, int(round(source_budget * 2.0 / 3.0)))
        diversified_ranked = list(safe[:learned_quota])
        admitted = set(diversified_ranked)
        for source in biosphere:
            if source not in admitted:
                diversified_ranked.append(source)
                admitted.add(source)
            if len(diversified_ranked) >= source_budget:
                break
        diversified_ranked.extend(source for source in safe if source not in admitted)
        diversified_ranked.extend(source for source in ranked if source in unsafe_reasons)
    if learned_ranker is None:
        additions = [source for source in ranked if source not in first][
            :max(0, source_budget - len(first))
        ]
        output[0] = tuple(first + additions)
        return output

    # Learned gravity may exchange every L1 source, but conservation forbids
    # deletion: displaced legacy sources are re-homed in later capsules. Thus
    # the top-ten provenance union is monotonic while all ten L1 slots become
    # available to the trained evidence field.
    if matter_v320 and groups is not None and neighbor_budget > 0:
        # Broad collection questions need independent facts more than adjacent
        # dialogue context, so their companion matter decouples completely.
        requested_companions = 0 if mode.broad_set else neighbor_budget
        companion_slots = min(max(0, requested_companions), max(0, source_budget - 1))
        selected = list(ranked[:source_budget - companion_slots])
        selected_set = set(selected)
        if companion_slots:
            proposals: list[tuple[float, int, int]] = []
            for seed_rank, seed in enumerate(ranked[:min(24, len(ranked))]):
                for neighbor in (seed - 1, seed + 1):
                    if (
                        0 <= neighbor < len(groups)
                        and groups[neighbor] == groups[seed]
                        and neighbor not in selected_set
                    ):
                        binding = scores[seed] + 0.42 * scores.get(neighbor, 0.0)
                        proposals.append((-binding, seed_rank, neighbor))
            for _, _, neighbor in sorted(proposals):
                if len(selected) >= source_budget:
                    break
                if neighbor not in selected_set:
                    selected.append(neighbor)
                    selected_set.add(neighbor)
        for source in ranked:
            if len(selected) >= source_budget:
                break
            if source not in selected_set:
                selected.append(source)
                selected_set.add(source)
    else:
        if matter_v322:
            # Pauli exclusion: near-duplicate sources cannot occupy most of L1
            # merely by repeating the same lexical mass. Deferred sources are
            # used only if the diverse safe field cannot fill the hard budget.
            selected = []
            deferred: list[int] = []
            selected_tokens: list[set[str]] = []
            selection_order = diversified_ranked if diversified_ranked is not None else ranked
            safe_ranked = [source for source in selection_order if source not in unsafe_reasons]
            unsafe_ranked = [source for source in ranked if source in unsafe_reasons]
            token_cache = {
                source: set().union(*(atom.tokens for atom in facts[source]))
                for source in safe_ranked[:192]
            }
            crowded: set[int] = set()
            probe_sources = list(token_cache)
            for source in probe_sources:
                tokens = token_cache[source]
                peers = 0
                for other in probe_sources:
                    if other == source:
                        continue
                    prior = token_cache[other]
                    if len(tokens & prior) / max(1, len(tokens | prior)) >= 0.82:
                        peers += 1
                        if peers >= 7:
                            crowded.add(source)
                            break
            for source in safe_ranked:
                tokens = token_cache.get(
                    source, set().union(*(atom.tokens for atom in facts[source])),
                )
                duplicate = source in crowded and any(
                    len(tokens & prior) / max(1, len(tokens | prior)) >= 0.82
                    for prior in selected_tokens
                )
                if duplicate and not mode.broad_set:
                    deferred.append(source)
                    continue
                selected.append(source)
                selected_tokens.append(tokens)
                if len(selected) >= source_budget:
                    break
            for source in deferred:
                if len(selected) >= source_budget:
                    break
                selected.append(source)
            if not matter_v323:
                for source in unsafe_ranked:
                    if len(selected) >= source_budget:
                        break
                    selected.append(source)
        else:
            selected = ranked[:source_budget]
    selected_set = set(selected)
    displaced = [source for source in first if source not in selected_set]
    output[0] = tuple(selected)
    width = len(output)
    if width > 1:
        current_union = {source for capsule in output for source in capsule}
        shadow_budget = int(getattr(learned_ranker, "conservation_budget", 0))
        for source in conserved_shadow:
            if shadow_budget <= 0:
                break
            if source not in current_union and source not in displaced:
                displaced.append(source)
                current_union.add(source)
                shadow_budget -= 1
        for offset, source in enumerate(displaced):
            home = 1 + offset % (width - 1)
            output[home] = tuple(dict.fromkeys(output[home] + (source,)))
    if diagnostics is not None and matter_v322:
        diagnostics["unsafe_reasons"] = {
            int(source): list(reasons) for source, reasons in unsafe_reasons.items()
        }
        diagnostics["ranked_sources"] = list(ranked)
        diagnostics["selected_sources"] = list(selected)
        diagnostics["selected_scores"] = [float(scores[source]) for source in selected]
        diagnostics["source_budget"] = int(source_budget)
        diagnostics["biosphere_triggered"] = bool(
            matter_v323 and BIOSPHERE_QUERY_RE.search(query)
        )
    return output
