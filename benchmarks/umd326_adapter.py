"""UMD 3.26 singularity capsules, fact versions, and emergent domains."""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date as calendar_date
from typing import Any, Iterable, Sequence

from benchmarks.public_benchmarks import tokenize
from benchmarks.umd316_adapter import query_physics
from benchmarks.umd325_adapter import AGGREGATE_CENSUS_RE, galactic_census_budget


CENSUS_INTENT_RE = re.compile(
    r"\b(?:where should I (?:travel|go)|what should I (?:read|watch|listen to)|"
    r"help me choose|pick (?:a|some)|ideas? for (?:a|my)|"
    r"recommend(?:ation|ations|ed)?|suggest(?:ion|ions|ed)?)\b",
    re.IGNORECASE,
)

COLLECTION_CATEGORIES = {
    "movie": {"genres", "actors", "directors", "already_watched_list"},
    "music": {"genres", "artists", "decades", "already_listened_list"},
    "book": {"genres", "authors", "already_read_list", "topics"},
    "travel": {"regions", "destination_types", "already_visited_list", "climates"},
}


def is_census_query(query: str) -> bool:
    mode = query_physics(query)
    return bool(mode.broad_set or AGGREGATE_CENSUS_RE.search(query) or CENSUS_INTENT_RE.search(query))


def singularity_budget(query: str, source_count: int, *, default: int = 32) -> int:
    if AGGREGATE_CENSUS_RE.search(query):
        return min(max(0, int(source_count)), 768)
    if CENSUS_INTENT_RE.search(query):
        return min(max(0, int(source_count)), 1024)
    return galactic_census_budget(query, source_count, default=default)


def collapse_census_singularity(
    query: str, capsules: Sequence[tuple[int, ...]], *, horizon: int = 10,
) -> list[tuple[int, ...]]:
    """Represent the complete census horizon as one provenance capsule.

    This changes capsule granularity, not the selected source set.  The first
    capsule is backed by a compact structured payload generated below; the old
    striped capsules remain available at ranks 2..10 for inspection.
    """
    output = [tuple(capsule) for capsule in capsules]
    if not output or not is_census_query(query):
        return output
    merged = tuple(dict.fromkeys(
        source for capsule in output[:max(1, horizon)] for source in capsule
    ))
    output[0] = merged
    return output


@dataclass(frozen=True)
class VersionedFact:
    fact_id: str
    source_id: str
    field: str
    value: Any
    state: str
    category: str
    date: str


def _leaf_values(value: Any, prefix: str = "") -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            field = f"{prefix}.{key}" if prefix else str(key)
            yield from _leaf_values(child, field)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _leaf_values(child, f"{prefix}[{index}]")
    elif value is not None and value != "":
        yield prefix, value


def compile_versioned_facts(records: Sequence[dict[str, Any]], source_ids: Sequence[str]) -> tuple[VersionedFact, ...]:
    """Compile current, superseded, and tombstone facts with stable IDs."""
    output: list[VersionedFact] = []
    for record, source_id in zip(records, source_ids):
        operation = str(record.get("operation", "")).casefold()
        details = record.get("operation_details") or {}
        category = str(
            details.get("category", details.get("subcategory", record.get("session_type", "")))
            if isinstance(details, dict) else record.get("session_type", "")
        )
        date = str(record.get("date", ""))
        for field, value in _leaf_values(details):
            path_parts = re.split(r"[.\[]", field.casefold())
            if any(part.startswith("old_") for part in path_parts):
                state = "superseded"
            elif operation in {"delete", "remove", "forget", "cancel"}:
                state = "tombstone"
            else:
                state = "current"
            output.append(VersionedFact(
                fact_id=f"{source_id}#{state}:{field}", source_id=str(source_id),
                field=field, value=value, state=state, category=category, date=date,
            ))
    return tuple(output)


class EmergentDomainGraph:
    """Query-independent token/session graph with query-time PMI expansion."""

    def __init__(self, token_sets: Sequence[frozenset[str]]) -> None:
        self.token_sets = [frozenset(
            token for token in tokens
            if "_" not in token and not token.isdigit() and len(token) > 2
        ) for tokens in token_sets]
        postings: dict[str, set[int]] = defaultdict(set)
        for source, tokens in enumerate(self.token_sets):
            for token in tokens:
                postings[token].add(source)
        self.postings = dict(postings)
        self.size = len(self.token_sets)

    def expand(self, query: str, *, limit: int = 24) -> set[str]:
        query_tokens = set(tokenize(query))
        counts: Counter[str] = Counter()
        anchor_docs: set[int] = set()
        for token in query_tokens:
            docs = self.postings.get(token, set())
            anchor_docs.update(docs)
            for source in docs:
                counts.update(self.token_sets[source])
        if not anchor_docs:
            return set()
        scored: list[tuple[float, str]] = []
        for token, overlap in counts.items():
            if token in query_tokens or overlap < 2:
                continue
            df = len(self.postings.get(token, ()))
            if df > max(8, int(0.40 * max(1, self.size))):
                continue
            # Ochiai/PMI-like association rewards repeated local contact while
            # suppressing globally common dialogue terms.
            score = overlap / math.sqrt(max(1, len(anchor_docs) * df))
            scored.append((score, token))
        return {token for _, token in sorted(scored, key=lambda item: (-item[0], item[1]))[:limit]}


def compact_census_payload(
    query: str,
    provenance: Sequence[str],
    facts: Sequence[VersionedFact],
) -> dict[str, Any]:
    """Build an answer-ready structured payload instead of raw source text."""
    selected = set(str(source) for source in provenance)
    current = [fact for fact in facts if fact.source_id in selected and fact.state == "current"]
    aggregate_mode = bool(AGGREGATE_CENSUS_RE.search(query))
    numeric_sums: dict[tuple[str, str], float] = defaultdict(float)
    numeric_counts: Counter[tuple[str, str]] = Counter()
    temporal_sums: dict[tuple[str, str, str], float] = defaultdict(float)
    temporal_counts: Counter[tuple[str, str, str]] = Counter()
    weekly_sums: dict[tuple[str, str, str], float] = defaultdict(float)
    weekly_counts: Counter[tuple[str, str, str]] = Counter()
    dimension_sums: dict[tuple[str, str, str, str], float] = defaultdict(float)
    dimension_counts: Counter[tuple[str, str, str, str]] = Counter()
    window_sums: dict[tuple[str, str, str], float] = defaultdict(float)
    window_counts: Counter[tuple[str, str, str]] = Counter()
    dimension_window_sums: dict[tuple[str, str, str, str, str], float] = defaultdict(float)
    dimension_window_counts: Counter[tuple[str, str, str, str, str]] = Counter()
    values: dict[tuple[str, str], list[Any]] = defaultdict(list)
    categories: Counter[str] = Counter()
    facts_by_source: dict[str, list[VersionedFact]] = defaultdict(list)
    for fact in current:
        facts_by_source[fact.source_id].append(fact)
    parsed_dates = []
    for fact in current:
        try:
            parsed_dates.append(calendar_date.fromisoformat(fact.date[:10]))
        except ValueError:
            pass
    window_origin = min(parsed_dates) if parsed_dates else None
    for fact in current:
        categories[fact.category] += 1
        if isinstance(fact.value, (int, float)) and not isinstance(fact.value, bool):
            key = (fact.category, fact.field)
            numeric_sums[key] += float(fact.value)
            numeric_counts[key] += 1
            period = fact.date[:7]
            if period:
                temporal_key = (fact.category, fact.field, period)
                temporal_sums[temporal_key] += float(fact.value)
                temporal_counts[temporal_key] += 1
            date_match = re.match(r"(\d{4}-\d{2})-(\d{2})", fact.date)
            if date_match:
                week = (int(date_match.group(2)) - 1) // 7 + 1
                week_key = (fact.category, fact.field, f"{date_match.group(1)}-W{week}")
                weekly_sums[week_key] += float(fact.value)
                weekly_counts[week_key] += 1
            fact_date = None
            try:
                fact_date = calendar_date.fromisoformat(fact.date[:10])
            except ValueError:
                pass
            window = None
            if fact_date is not None and window_origin is not None:
                window = f"W{(fact_date - window_origin).days // 30 + 1}"
                window_key = (fact.category, fact.field, window)
                window_sums[window_key] += float(fact.value)
                window_counts[window_key] += 1
            parent = fact.field.rsplit(".", 1)[0] if "." in fact.field else ""
            for sibling in facts_by_source[fact.source_id]:
                sibling_parent = sibling.field.rsplit(".", 1)[0] if "." in sibling.field else ""
                if sibling is fact or sibling_parent != parent or not isinstance(sibling.value, str):
                    continue
                if len(sibling.value) > 80:
                    continue
                dimension_key = (
                    fact.category, fact.field, sibling.field, sibling.value.casefold(),
                )
                dimension_sums[dimension_key] += float(fact.value)
                dimension_counts[dimension_key] += 1
                if window is not None:
                    dimension_window_key = dimension_key + (window,)
                    dimension_window_sums[dimension_window_key] += float(fact.value)
                    dimension_window_counts[dimension_window_key] += 1
        elif not aggregate_mode and len(str(fact.value)) <= 160:
            leaf = fact.field.rsplit(".", 1)[-1].casefold()
            if leaf not in {
                "category", "subcategory", "created_at", "actual_operation",
                "update_type", "fallback_reason",
            }:
                values[(fact.category, fact.field)].append(fact.value)
    # Preserve unique current values, but never old/tombstoned facts.
    unique_values: dict[str, dict[str, list[str]]] = defaultdict(dict)
    for (category, field), field_values in values.items():
        unique_values[category][field] = list(dict.fromkeys(str(value) for value in field_values))
    if not aggregate_mode:
        query_tokens = set(tokenize(query))
        query_tokens.update(token for value in list(query_tokens) for token in tokenize(value))
        selected_domains: set[str] = set()
        domain_anchors = {
            "movie": {"movie", "film", "watch"},
            "music": {"music", "song", "album", "listen"},
            "book": {"book", "read", "author"},
            "travel": {"travel", "trip", "visit", "place", "destination"},
        }
        for domain, anchors in domain_anchors.items():
            anchor_tokens = set(token for anchor in anchors for token in tokenize(anchor))
            if query_tokens & anchor_tokens:
                selected_domains.add(domain)
        allowed_categories = set().union(*(
            COLLECTION_CATEGORIES[domain] for domain in selected_domains
        )) if selected_domains else set()
        if allowed_categories:
            unique_values = defaultdict(dict, {
                category: fields for category, fields in unique_values.items()
                if category in allowed_categories
            })
            categories = Counter({
                category: count for category, count in categories.items()
                if category in allowed_categories
            })
    numeric: dict[str, dict[str, dict[str, float | int]]] = defaultdict(dict)
    for (category, field), total in sorted(numeric_sums.items()):
        numeric[category][field] = {
            "sum": round(total, 6), "count": numeric_counts[(category, field)],
        }
    by_period: dict[str, dict[str, dict[str, dict[str, float | int]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for (category, field, period), total in sorted(temporal_sums.items()):
        by_period[category][field][period] = {
            "sum": round(total, 6),
            "count": temporal_counts[(category, field, period)],
        }
    by_week: dict[str, dict[str, dict[str, dict[str, float | int]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for (category, field, week), total in sorted(weekly_sums.items()):
        by_week[category][field][week] = {
            "sum": round(total, 6), "count": weekly_counts[(category, field, week)],
        }
    by_dimension: dict[str, dict[str, dict[str, dict[str, dict[str, float | int]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(dict))
    )
    for (category, field, dimension_field, dimension_value), total in sorted(dimension_sums.items()):
        by_dimension[category][field][dimension_field][dimension_value] = {
            "sum": round(total, 6),
            "count": dimension_counts[(category, field, dimension_field, dimension_value)],
        }
    by_window: dict[str, dict[str, dict[str, dict[str, float | int]]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for (category, field, window), total in sorted(window_sums.items()):
        by_window[category][field][window] = {
            "sum": round(total, 6), "count": window_counts[(category, field, window)],
        }
    by_dimension_window: dict[str, dict[str, dict[str, dict[str, dict[str, dict[str, float | int]]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    )
    for (category, field, dimension_field, dimension_value, window), total in sorted(dimension_window_sums.items()):
        by_dimension_window[category][field][dimension_field][dimension_value][window] = {
            "sum": round(total, 6),
            "count": dimension_window_counts[(category, field, dimension_field, dimension_value, window)],
        }
    if aggregate_mode:
        query_terms = set(tokenize(query))
        query_terms.discard("total")
        if query_terms & {"spend", "spent", "spending", "cost", "total"}:
            query_terms.update({"expense", "amount"})
        if query_terms & {"step", "steps"}:
            query_terms.update({"step", "tracker"})

        def relevant_label(*parts: str) -> bool:
            label_tokens = set(tokenize(" ".join(parts).replace("_", " ")))
            return bool(label_tokens & query_terms)

        relevant_categories = {
            category for category, fields in numeric.items()
            if relevant_label(category) or any(relevant_label(field) for field in fields)
        }
        # A dimension value such as coffee/lunch may select an otherwise broad
        # category such as food_expenses.
        for category, fields in by_dimension.items():
            for field, dimensions in fields.items():
                for dimension_field, dimension_values in dimensions.items():
                    if any(relevant_label(value) for value in dimension_values):
                        relevant_categories.add(category)
        if relevant_categories:
            numeric = defaultdict(dict, {
                category: fields for category, fields in numeric.items()
                if category in relevant_categories
            })
            by_period = defaultdict(lambda: defaultdict(dict), {
                category: fields for category, fields in by_period.items()
                if category in relevant_categories
            })
            by_week = defaultdict(lambda: defaultdict(dict), {
                category: fields for category, fields in by_week.items()
                if category in relevant_categories
            })
            by_window = defaultdict(lambda: defaultdict(dict), {
                category: fields for category, fields in by_window.items()
                if category in relevant_categories
            })
            by_dimension_window = defaultdict(
                lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))),
                {
                    category: fields for category, fields in by_dimension_window.items()
                    if category in relevant_categories
                },
            )
            filtered_dimensions: dict[str, dict[str, dict[str, dict[str, dict[str, float | int]]]]] = {}
            for category, fields in by_dimension.items():
                if category not in relevant_categories:
                    continue
                kept_fields: dict[str, dict[str, dict[str, dict[str, float | int]]]] = {}
                for field, dimensions in fields.items():
                    kept_dimensions: dict[str, dict[str, dict[str, float | int]]] = {}
                    for dimension_field, dimension_values in dimensions.items():
                        matches = {
                            value: stats for value, stats in dimension_values.items()
                            if relevant_label(value)
                        }
                        if matches:
                            kept_dimensions[dimension_field] = matches
                    if kept_dimensions:
                        kept_fields[field] = kept_dimensions
                if kept_fields:
                    filtered_dimensions[category] = kept_fields
            by_dimension = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)), filtered_dimensions)
    payload = {
        "mode": "aggregate" if aggregate_mode else "collection",
        "source_count": len(selected),
        "current_fact_count": len(current),
        "provenance": list(dict.fromkeys(str(source) for source in provenance)),
        "numeric": dict(numeric),
        "numeric_by_period": {
            category: {field: dict(periods) for field, periods in fields.items()}
            for category, fields in by_period.items()
        },
        "numeric_by_week": {
            category: {field: dict(weeks) for field, weeks in fields.items()}
            for category, fields in by_week.items()
        },
        "numeric_by_dimension": {
            category: {
                field: {dimension: dict(values) for dimension, values in dimensions.items()}
                for field, dimensions in fields.items()
            }
            for category, fields in by_dimension.items()
        },
        "numeric_by_30d_window": {
            category: {field: dict(windows) for field, windows in fields.items()}
            for category, fields in by_window.items()
        },
        "numeric_by_dimension_30d_window": {
            category: {
                field: {
                    dimension: {value: dict(windows) for value, windows in values.items()}
                    for dimension, values in dimensions.items()
                }
                for field, dimensions in fields.items()
            }
            for category, fields in by_dimension_window.items()
        },
        "category_fact_counts": dict(sorted(categories.items())),
        "values": {category: dict(sorted(fields.items())) for category, fields in sorted(unique_values.items())},
    }
    payload["serialized_chars"] = len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    return payload
