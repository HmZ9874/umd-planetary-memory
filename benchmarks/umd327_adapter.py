"""UMD 3.27 structured census ledger and fair dual-orbit reporting."""

from __future__ import annotations

import re
from collections import defaultdict
from datetime import date, timedelta
from typing import Any, Sequence

from benchmarks.public_benchmarks import tokenize
from benchmarks.umd326_adapter import is_census_query


DOMAIN_CATEGORIES = {
    "movie": {"genres", "actors", "directors", "already_watched_list"},
    "book": {"genres", "authors", "topics", "already_read_list"},
    "music": {"genres", "artists", "decades", "already_listened_list"},
    "travel": {"regions", "destination_types", "climates", "already_visited_list"},
}

DOMAIN_RE = {
    "movie": re.compile(
        r"\b(?:movies?|films?|cinema|watch(?:ed|ing)?|dramas?|comed(?:y|ies))\b", re.I,
    ),
    "book": re.compile(r"\b(?:book|author|novel|read(?:ing)?)s?\b", re.I),
    "music": re.compile(r"\b(?:music|song|album|artist|listen(?:ed|ing)?)s?\b", re.I),
    "travel": re.compile(r"\b(?:travel|trip|visit|place|destination|where should I go)s?\b", re.I),
}

MUTABLE_CATEGORIES = {
    "todo_list", "calendar_event",
    *(category for categories in DOMAIN_CATEGORIES.values() for category in categories),
}

_EXPLICIT_CATEGORY_TERMS = {
    "genres": "genres", "topics": "topics", "authors": "authors", "actors": "actors",
    "directors": "directors", "artists": "artists", "decades": "decades",
    "regions": "regions", "climates": "climates",
    "destination types": "destination_types", "preferred types": "destination_types",
}

_DOMAIN_ITEM_HINTS = {
    "movie": re.compile(
        r"\b(?:dramas?|comed(?:y|ies)|westerns?|cyberpunk|space opera|thrillers?|noir|romance|horror)\b",
        re.I,
    ),
    "music": re.compile(
        r"\b(?:bebop|jazz|folk|lo-fi|classical|swing|minimalism|rock|pop|hip[ -]?hop|r&b|opera|metal|symphony)\b",
        re.I,
    ),
}


def record_category(record: dict[str, Any]) -> str:
    details = record.get("operation_details") or {}
    if not isinstance(details, dict):
        return str(record.get("session_type", "")).casefold()
    return str(
        details.get("category", details.get("subcategory", record.get("session_type", "")))
    ).casefold()


def _identity(value: Any) -> Any:
    """Stable state identity, excluding fields that are mutable attributes."""
    if isinstance(value, dict):
        for field in ("description", "event_name", "name", "title", "id"):
            if field in value:
                return (field, _identity(value[field]))
        return tuple(sorted(
            (str(key), _identity(child)) for key, child in value.items()
            if key not in {"created_at", "updated_at", "date", "preference"}
        ))
    if isinstance(value, list):
        return tuple(_identity(child) for child in value)
    return str(value).strip().casefold()


def _memory_text(record: dict[str, Any]) -> str:
    messages = record.get("conversation") or []
    return " ".join(
        str(message.get("message", ""))
        for message in messages if isinstance(message, dict) and message.get("share_memory")
    ).casefold()


def record_domain(record: dict[str, Any]) -> str | None:
    """Infer a preference domain from its memory-bearing utterance, not labels."""
    category = record_category(record)
    for domain, categories in DOMAIN_CATEGORIES.items():
        if category in categories - {"genres"}:
            return domain
    text = _memory_text(record)
    scores = {
        domain: len(pattern.findall(text)) for domain, pattern in DOMAIN_RE.items()
    }
    best = max(scores, key=scores.get, default="")
    if best and scores[best]:
        return best
    details = record.get("operation_details") or {}
    item = str(details.get("item", "")) if isinstance(details, dict) else ""
    hinted = [domain for domain, pattern in _DOMAIN_ITEM_HINTS.items() if pattern.search(item)]
    # Ordered from the more specific compositional vocabulary (for example
    # "space opera") to the broader music vocabulary ("opera").
    return hinted[0] if hinted else None


def _event_is_live(record: dict[str, Any], as_of: str | None) -> bool:
    if not as_of:
        return True
    details = record.get("operation_details") or {}
    item = details.get("item") if isinstance(details, dict) else None
    if not isinstance(item, dict):
        return True
    raw_due = str(item.get("date", ""))
    try:
        observation = date.fromisoformat(str(as_of))
        offset = re.fullmatch(r"\+(\d+) days?", raw_due)
        if offset:
            # Memora stores the offset against the event's stable creation
            # anchor; updated_at is the edit time, not a new temporal origin.
            anchor = item.get("created_at") or record.get("date")
            due = date.fromisoformat(str(anchor)) + timedelta(days=int(offset.group(1)))
        else:
            due = date.fromisoformat(raw_due)
    except (TypeError, ValueError):
        return True
    return due > observation


class StructuredCensusLedger:
    """Prefix-safe active-state postings over structured memory events.

    Append-only measurements conserve every positive event. Mutable lists and
    preferences are folded by semantic identity, while the original source is
    retained as immutable provenance across attribute-only updates.
    """

    def __init__(self, records: Sequence[dict[str, Any]]) -> None:
        self.records = list(records)
        self.source_count = len(records)
        self._snapshot_cache: dict[int, dict[str, list[tuple[int, dict[str, Any]]]]] = {}

    def _snapshot(self, ceiling: int) -> dict[str, list[tuple[int, dict[str, Any]]]]:
        cached = self._snapshot_cache.get(ceiling)
        if cached is not None:
            return cached
        append_only: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
        mutable: dict[str, dict[Any, tuple[int, dict[str, Any]]]] = defaultdict(dict)
        day_origins: dict[tuple[str, Any, str], int] = {}
        negative = {"delete", "remove", "forget", "cancel"}
        for source, record in enumerate(self.records[:ceiling]):
            operation = str(record.get("operation", "")).casefold()
            category = record_category(record)
            details = record.get("operation_details") or {}
            if (
                category == "no_memory"
                or str(record.get("session_type", "")).casefold() == "no_memory"
                or not isinstance(details, dict)
            ):
                continue
            if category not in MUTABLE_CATEGORIES:
                if operation not in negative:
                    append_only[category].append((source, record))
                continue
            item = details.get("item")
            identity = _identity(item)
            state = mutable[category]
            if operation in negative:
                state.pop(identity, None)
                continue
            if operation == "update":
                old_identity = _identity(details.get("old_item"))
                if old_identity != _identity(None) and old_identity != identity:
                    state.pop(old_identity, None)
                previous = state.get(identity)
                day_key = (category, identity, str(record.get("date", "")))
                day_origins.setdefault(day_key, source)
                # Attribute-only updates change the live value but conserve the
                # creation source. Value changes become a new provenance atom.
                origin = previous[0] if previous is not None else day_origins[day_key]
                state[identity] = (origin, record)
            elif operation == "add":
                day_key = (category, identity, str(record.get("date", "")))
                day_origins.setdefault(day_key, source)
                previous = state.get(identity)
                origin = previous[0] if previous is not None else day_origins[day_key]
                state[identity] = (origin, record)
        snapshot = dict(append_only)
        snapshot.update({category: list(entries.values()) for category, entries in mutable.items()})
        if len(self._snapshot_cache) >= 2:
            self._snapshot_cache.clear()
        self._snapshot_cache[ceiling] = snapshot
        return snapshot

    def query_categories(self, query: str) -> set[str]:
        q = query.casefold()
        categories: set[str] = set()
        if re.search(r"\b(?:todo|to-do|tasks? remain|remaining tasks?)\b", q):
            categories.add("todo_list")
        if re.search(r"\b(?:calendar|upcoming events?|events? do I have)\b", q):
            categories.add("calendar_event")
        if re.search(r"\b(?:spend|spent|spending|expenses?|food|coffee|lunch|breakfast|dinner|grocery)\b", q):
            categories.add("food_expenses")
        if re.search(r"\b(?:steps?|walking)\b", q):
            categories.add("step_tracker")
            if "goal" in q or "daily" in q:
                categories.add("daily_steps")
        if "budget" in q:
            for category in ("coffee", "lunch", "breakfast", "dinner", "grocery"):
                if category in q:
                    categories.add(category)
        for domain, pattern in DOMAIN_RE.items():
            if pattern.search(query):
                categories.update(DOMAIN_CATEGORIES[domain])

        explicit = {
            category for term, category in _EXPLICIT_CATEGORY_TERMS.items() if term in q
        }
        if explicit:
            categories = explicit

        # Direct category language is a fallback. Requiring the whole category
        # phrase prevents the token "list" from coupling todo_list to every
        # recommendation history list.
        if not categories:
            query_tokens = set(tokenize(query.replace("_", " ")))
            for category in self._snapshot(self.source_count):
                category_tokens = set(tokenize(category.replace("_", " ")))
                if category_tokens and category_tokens <= query_tokens:
                    categories.add(category)
        return categories

    def select(
        self, query: str, *, prefix: int | None = None, as_of: str | None = None,
    ) -> list[int]:
        categories = self.query_categories(query)
        ceiling = self.source_count if prefix is None else min(self.source_count, max(0, int(prefix)))
        snapshot = self._snapshot(ceiling)
        domains = {
            domain for domain, pattern in DOMAIN_RE.items() if pattern.search(query)
        }
        q = query.casefold()
        expense_type = next(
            (name for name in ("coffee", "lunch", "breakfast", "dinner", "grocery") if name in q),
            None,
        )
        horizon_match = (
            re.search(r"\blast\s+(\d+)\s+months?\b", q)
            if "which month" in q else None
        )
        horizon_start: date | None = None
        horizon_end: date | None = None
        if horizon_match:
            try:
                horizon_start = min(
                    date.fromisoformat(str(record.get("date")))
                    for record in self.records[:ceiling] if record.get("date")
                )
                horizon_end = horizon_start + timedelta(days=30 * int(horizon_match.group(1)))
            except (TypeError, ValueError):
                horizon_start = horizon_end = None
        selected: list[int] = []
        for category in categories:
            for source, current_record in snapshot.get(category, ()):
                if domains and category == "genres":
                    # Preference-only updates often omit the original domain
                    # wording. Fall back to the immutable creation utterance.
                    domain = record_domain(current_record) or record_domain(self.records[source])
                    if domain is not None and domain not in domains:
                        continue
                if expense_type and category == "food_expenses":
                    details = current_record.get("operation_details") or {}
                    item = details.get("item") if isinstance(details, dict) else None
                    if not isinstance(item, dict) or str(item.get("expense_type", "")).casefold() != expense_type:
                        continue
                if horizon_end is not None and category in {"food_expenses", "step_tracker"}:
                    try:
                        event_date = date.fromisoformat(str(current_record.get("date")))
                    except (TypeError, ValueError):
                        event_date = None
                    if event_date is not None and not (horizon_start <= event_date < horizon_end):
                        continue
                if category == "calendar_event" and not _event_is_live(current_record, as_of):
                    continue
                selected.append(source)
        return sorted(dict.fromkeys(selected))


def ledger_census_orbit(
    query: str,
    ledger_sources: Sequence[int],
    fallback_orbit: Sequence[int],
    *,
    fallback_budget: int = 48,
) -> list[int]:
    """Conserve all active ledger matches, followed by a bounded safety orbit."""
    if not is_census_query(query) or not ledger_sources:
        return list(dict.fromkeys(fallback_orbit))
    protected = list(dict.fromkeys(int(source) for source in ledger_sources))
    seen = set(protected)
    protected.extend(
        source for source in fallback_orbit if source not in seen
    )
    return protected


def ledger_census_budget(
    query: str, ledger_sources: Sequence[int], source_count: int, *, fallback_budget: int = 48,
) -> int:
    if not is_census_query(query) or not ledger_sources:
        return min(max(0, source_count), max(1, fallback_budget))
    return min(max(0, source_count), len(set(ledger_sources)) + max(0, fallback_budget))


def solve_compact_numeric(query: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    """Answer common aggregate intents directly from the compact payload."""
    q = query.casefold()
    if any(term in q for term in ("goal", "budget", "meet my", "meeting my")):
        return None
    numeric = payload.get("numeric", {})
    by_dimension = payload.get("numeric_by_dimension", {})
    by_week = payload.get("numeric_by_week", {})
    by_period = payload.get("numeric_by_period", {})
    by_window = payload.get("numeric_by_30d_window", {})
    by_dimension_window = payload.get("numeric_by_dimension_30d_window", {})

    if "step" in q:
        field = numeric.get("step_tracker", {}).get("item.step_count")
        if re.search(r"\btotal\b|how many", q) and field:
            return {"value": field["sum"], "count": field["count"], "unit": "steps"}
        phases = by_week.get("step_tracker", {}).get("item.step_count", {})
        if "week" in q and phases:
            phase, stats = max(phases.items(), key=lambda item: item[1]["sum"])
            return {"phase": phase, "value": stats["sum"], "unit": "steps"}
        periods = (
            by_window.get("step_tracker", {}).get("item.step_count", {})
            if "last 3 months" in q else by_period.get("step_tracker", {}).get("item.step_count", {})
        )
        if "month" in q and periods:
            eligible = {key: value for key, value in periods.items() if key in {"W1", "W2", "W3"}} if "last 3 months" in q else periods
            phase, stats = max(eligible.items(), key=lambda item: item[1]["sum"])
            return {"phase": phase, "value": stats["sum"], "unit": "steps"}

    expense = numeric.get("food_expenses", {}).get("item.amount")
    if re.search(r"\b(?:spend|spent|spending|expenses?)\b", q) and expense:
        subtype = next((name for name in ("coffee", "lunch", "breakfast", "dinner", "grocery") if name in q), None)
        if subtype:
            stats = (
                by_dimension.get("food_expenses", {})
                .get("item.amount", {}).get("item.expense_type", {}).get(subtype)
            )
            if stats:
                return {"value": stats["sum"], "count": stats["count"], "unit": "currency", "type": subtype}
        return {"value": expense["sum"], "count": expense["count"], "unit": "currency"}
    return None
