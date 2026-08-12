"""UMD 3.35 background ghost matter for provenance-safe organization.

Ghosts are query-independent navigation traces compiled from visible sources.
They never become answer provenance.  A ghost distills autobiographical user
clauses and projects absolute dates into prefix-relative aliases; retrieval
must always re-anchor a ghost to its immutable source ID.
"""

from __future__ import annotations

import calendar
import datetime as dt
import re
from dataclasses import dataclass
from typing import Sequence

from benchmarks.public_benchmarks import BM25, ranking, unit_scores
from benchmarks.umd334_adapter import QueryFissionResult, episodic_nucleus


_DATE_RE = re.compile(r"\b(20\d{2})[/-](\d{1,2})[/-](\d{1,2})\b")
_CLAUSE_RE = re.compile(r"(?<=[.!?])\s+|\s+(?:but|however|by the way|also)\s+", re.I)
_FACT_RE = re.compile(
    r"\b(?:i|i'm|i've|i'd|my|mine|we|our)\b.*\b(?:have|had|use|used|own|"
    r"prefer|preferred|like|liked|love|loved|enjoy|enjoyed|want|wanted|"
    r"bought|got|made|started|finished|attended|visited|went|joined|"
    r"currently|usually|always|never|recently|last|ago)\b",
    re.I,
)
_STATE_RE = re.compile(
    r"\b(?:prefer|like|love|enjoy|favorite|favourite|currently|usually|"
    r"always|never|allergic|own|have|use|setup|routine|habit)\b",
    re.I,
)
_EVENT_RE = re.compile(
    r"\b(?:bought|got|made|started|finished|attended|visited|went|joined|"
    r"met|received|ordered|fixed|watched|read|flew|traveled|travelled)\b",
    re.I,
)
_REQUEST_RE = re.compile(
    r"\b(?:can you|could you|would you|please|recommend|suggest|tips?|"
    r"advice|help me|how (?:do|can|should)|what should)\b",
    re.I,
)
_PREFERENCE_QUERY_RE = re.compile(
    r"\b(?:recommend|suggest|tips?|advice|what should|what to|which one|"
    r"do you think|any ideas|helpful)\b",
    re.I,
)
_TEMPORAL_QUERY_RE = re.compile(
    r"\b(?:ago|before|after|between|first|last|earliest|latest|recently|"
    r"days?|weeks?|months?|years?|when|order|since|until|most recently|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    re.I,
)
_TEMPORAL_STRICT_RE = re.compile(
    r"\b(?:how many (?:days?|weeks?|months?|years?) (?:ago|before|after|"
    r"between|passed)|how long (?:ago|before|after|between|had|have)|"
    r"what is the order|which .{0,80} first|who .{0,80} first|"
    r"happened first|from earliest to latest|from first to last|"
    r"days? passed between|weeks? passed between|months? passed between)\b",
    re.I,
)
_OBSERVER_QUERY_RE = re.compile(
    r"\b(?:you (?:said|made|recommended|suggested|answered|responded)|"
    r"your (?:answer|response|recommendation|suggestion)|our previous (?:chat|"
    r"conversation|game)|what was the move you made|last venue you recommended)\b",
    re.I,
)
_EVOLUTION_QUERY_RE = re.compile(
    r"\b(?:current(?:ly)?|previous(?:ly)?|initially|originally|used to|"
    r"switch(?:ed)?|change(?:d)?|new(?:est)?|latest|now|before getting|"
    r"after (?:the|my|her|his)|still|anymore)\b",
    re.I,
)
_COMPOSITE_QUERY_RE = re.compile(
    r"\b(?:percentage|percent|combined|both|pair|two|three|four|five|six|"
    r"seven|eight|nine|ten|how long|how many|day before|led me to|"
    r"from earliest to latest|from first to last|in the order)\b",
    re.I,
)
_NUMBER_WORDS = {
    "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}


@dataclass(frozen=True)
class GhostTrace:
    source: int
    distilled: str
    date: dt.date | None
    fact_mass: float


@dataclass(frozen=True)
class GhostResult:
    active: bool = False
    modes: tuple[str, ...] = ()
    order: tuple[int, ...] = ()
    scores: tuple[float, ...] = ()
    promoted_source: int | None = None
    confidence_ratio: float = 1.0
    confidence_margin: float = 0.0
    constellation_budget: int = 0


def _parse_date(value: str) -> dt.date | None:
    match = _DATE_RE.search(value)
    if not match:
        return None
    try:
        return dt.date(*(int(part) for part in match.groups()))
    except ValueError:
        return None


def distill_ghost(text: str) -> tuple[str, float]:
    """Keep fact-bearing user clauses and suppress requests/recommendations."""
    nucleus = episodic_nucleus(text)
    selected: list[tuple[int, str]] = []
    for clause in _CLAUSE_RE.split(nucleus):
        clause = " ".join(clause.strip().split())
        if len(clause) < 12:
            continue
        score = 0
        score += 3 if _FACT_RE.search(clause) else 0
        score += 2 if _STATE_RE.search(clause) else 0
        score += 2 if _EVENT_RE.search(clause) else 0
        score += 1 if re.search(r"\b\d+(?:\.\d+)?\b", clause) else 0
        score -= 3 if "?" in clause or _REQUEST_RE.search(clause) else 0
        if score > 0:
            selected.append((score, clause))
    selected.sort(key=lambda item: (-item[0], -len(item[1]), item[1]))
    clauses = [clause for _, clause in selected[:8]]
    if not clauses:
        # A silent ghost is safer than turning a generic request into a fact.
        return "", 0.0
    mass = min(1.0, sum(score for score, _ in selected[:8]) / 20.0)
    return " ".join(clauses), mass


def temporal_aliases(date: dt.date | None, reference: dt.date | None) -> str:
    if date is None:
        return ""
    aliases = [
        date.isoformat(), calendar.month_name[date.month],
        calendar.day_name[date.weekday()], str(date.day), str(date.year),
    ]
    if reference is None or date > reference:
        return " ".join(aliases)
    days = (reference - date).days
    aliases.extend([f"{days} days ago", f"{days} day ago"])
    if days == 0:
        aliases.extend(["today", "most recently", "latest"])
    elif days == 1:
        aliases.append("yesterday")
    if 1 <= days <= 7:
        aliases.append(f"last {calendar.day_name[date.weekday()]}")
    weeks = round(days / 7)
    if weeks:
        aliases.extend([f"{weeks} weeks ago", f"{weeks} week ago"])
        if weeks == 1:
            aliases.append("last week")
    months = (reference.year - date.year) * 12 + reference.month - date.month
    if months:
        aliases.extend([f"{months} months ago", f"{months} month ago"])
        if months == 1:
            aliases.append("last month")
    return " ".join(dict.fromkeys(alias.casefold() for alias in aliases))


def constellation_budget(query: str, modes: Sequence[str]) -> int:
    """Estimate evidence multiplicity from the query without using gold."""
    lowered = query.casefold()
    requested = max(
        (value for word, value in _NUMBER_WORDS.items() if re.search(
            rf"\b{word}\b", lowered,
        )),
        default=0,
    )
    explicit = [int(value) for value in re.findall(r"\b([2-9]|10)\b", lowered)]
    requested = max([requested, *explicit])
    if requested:
        # List/order questions need a small distractor allowance because each
        # requested item is an independent event source.
        return 12
    if "evolution" in modes:
        return 5
    if "composite" in modes or "resonance" in modes:
        return 6
    if "temporal" in modes:
        return 10
    return 2


class BackgroundGhostCatalog:
    """Query-independent distilled traces with prefix-safe temporal projection."""

    def __init__(self, texts: Sequence[str], dates: Sequence[str] | None = None) -> None:
        date_values = list(dates or [])
        # Ghost matter is deliberately restricted to role-marked dialogue.
        # Plain documents, tools and benchmark passages remain bit-for-bit on
        # the established retrieval path instead of being anthropomorphized.
        self.role_marked = any(
            re.search(r"(?:^|\s)(?:user|human)\s*:", text, re.I)
            for text in texts
        )
        traces: list[GhostTrace] = []
        for source, text in enumerate(texts):
            distilled, mass = distill_ghost(text)
            date_text = date_values[source] if source < len(date_values) else text
            traces.append(GhostTrace(source, distilled, _parse_date(date_text), mass))
        self.traces = tuple(traces)

    def solve(
        self, query: str, *, size: int, fission: QueryFissionResult,
        adaptive_constellation: bool = True,
    ) -> GhostResult:
        visible = self.traces[:max(0, min(size, len(self.traces)))]
        modes: list[str] = []
        if _PREFERENCE_QUERY_RE.search(query):
            modes.append("preference")
        if _TEMPORAL_QUERY_RE.search(query):
            modes.append("temporal")
        if adaptive_constellation and _EVOLUTION_QUERY_RE.search(query):
            modes.append("evolution")
        if adaptive_constellation and _COMPOSITE_QUERY_RE.search(query):
            modes.append("composite")
        if (
            not self.role_marked or not visible
            or not any(trace.fact_mass > 0.0 for trace in visible)
        ):
            return GhostResult()
        distilled_documents = [
            trace.distilled or "silent ghost" for trace in visible
        ]
        if not modes and adaptive_constellation:
            resonance = unit_scores(BM25(distilled_documents).scores(query))
            contacts = sum(
                visible[index].fact_mass > 0.0 and score >= 0.05
                for index, score in enumerate(resonance)
            )
            if contacts >= 2:
                modes.append("resonance")
            else:
                return GhostResult()
        if not modes:
            return GhostResult()
        reference = max((trace.date for trace in visible if trace.date), default=None)
        documents = [
            " ".join((
                trace.distilled,
                temporal_aliases(trace.date, reference) if "temporal" in modes else "",
            )).strip() or "silent ghost"
            for trace in visible
        ]
        lexical = unit_scores(BM25(documents).scores(query))
        masses = [trace.fact_mass for trace in visible]
        base = list(fission.periapsis_scores[:len(visible)])
        if len(base) < len(visible):
            base.extend([0.0] * (len(visible) - len(base)))
        ghost_weight = (
            0.36 if "temporal" in modes or "evolution" in modes
            else 0.34 if "composite" in modes or "resonance" in modes else 0.30
        )
        scores = [
            (
                (1.0 - ghost_weight) * base[index]
                + ghost_weight * (0.82 * lexical[index] + 0.18 * masses[index])
            )
            if visible[index].fact_mass > 0.0 else -1.0
            for index in range(len(visible))
        ]
        # A silent trace is not ghost matter. It stays auditable through the
        # ordinary source path but cannot enter the background navigation
        # orbit merely because its complete source had a high base score.
        promotion_order = tuple(
            index for index in ranking(scores)
            if visible[index].fact_mass > 0.0
        )
        # Constellation packaging emphasizes derived fact/date contact more
        # strongly than Strict promotion. The latter retains the frozen 3.35
        # field so packaging improvements cannot silently change atomic rank.
        navigation_weight = (
            0.65 if "temporal" in modes
            else 0.55 if any(mode in modes for mode in ("evolution", "composite", "resonance"))
            else ghost_weight
        ) if adaptive_constellation else ghost_weight
        navigation_scores = [
            (
                (1.0 - navigation_weight) * base[index]
                + navigation_weight * (0.82 * lexical[index] + 0.18 * masses[index])
            )
            if visible[index].fact_mass > 0.0 else -1.0
            for index in range(len(visible))
        ]
        order = tuple(
            index for index in ranking(navigation_scores)
            if visible[index].fact_mass > 0.0
        )
        top = promotion_order[0]
        second = promotion_order[1] if len(promotion_order) > 1 else top
        ratio = scores[top] / max(1e-9, scores[second])
        margin = scores[top] - scores[second]
        promote = (
            top if (
                "temporal" in modes
                and _TEMPORAL_STRICT_RE.search(query)
                and not _OBSERVER_QUERY_RE.search(query)
                and lexical[top] >= 0.35
                and ratio >= 1.055
                and margin >= 0.035
            )
            else None
        )
        return GhostResult(
            active=True, modes=tuple(modes), order=tuple(order[:32]),
            scores=tuple(scores), promoted_source=promote,
            confidence_ratio=ratio, confidence_margin=margin,
            constellation_budget=(
                constellation_budget(query, modes) if adaptive_constellation else 2
            ),
        )
