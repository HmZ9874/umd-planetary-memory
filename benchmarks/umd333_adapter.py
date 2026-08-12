"""UMD 3.33 gold-blind document-star hierarchy.

Long synthetic contexts often preserve explicit ``Document N:`` boundaries,
while the frozen evaluator exposes only overlapping character windows.  This
adapter reconstructs the lossless window stream, indexes complete documents as
stars, and maps each selected star back to its immutable source-window moons.
Only visible text and the query are accepted; answers and evidence IDs are not
part of the interface.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from benchmarks.public_benchmarks import BM25, ranking
from benchmarks.umd332_adapter import (
    RelationSuperpositionGraph,
    superposed_terminal_relation_weights,
)


DOCUMENT_BOUNDARY_RE = re.compile(r"(?i)(?<!\w)Document\s+\d+\s*:")

ABSORBING_HEAD_RULES: tuple[tuple[re.Pattern[str], tuple[str, ...]], ...] = (
    (re.compile(r"\b(?:place|city|location)\b.{0,140}\b(?:first )?(?:established|founded)\b|\bwhere\b.{0,100}\b(?:first )?(?:established|founded)\b", re.I), ("founded_city",)),
    (re.compile(r"\b(?:partner|wife|husband|spouse) of\b", re.I), ("married",)),
    (re.compile(r"\breport(?:s|ed)? to\b|\bhighest-ranking official\b|\bchief of state\b", re.I), ("head_state",)),
    (re.compile(r"\b(?:prime minister|leader of (?:the )?government|government leader|head of (?:the )?.{0,30}government)\b", re.I), ("head_government",)),
    (re.compile(r"\b(?:head office|office of the company|office of the organization|currently headquartered|operate from)\b", re.I), ("headquarters_city",)),
    (re.compile(r"\b(?:proceedings|official medium|officially spoken|primary language)\b.{0,120}\blanguage\b|\blanguage\b.{0,120}\b(?:proceedings|official medium|officially spoken|primary)\b", re.I), ("official_language",)),
    (re.compile(r"\b(?:languages?|language)\b.{0,120}\b(?:fluent|speaking|spoken|signed|communicat)\w*\b|\b(?:fluent|speaking|spoken|signed)\b.{0,120}\blanguages?\b", re.I), ("speaks_language",)),
    (re.compile(r"\b(?:place of demise|where .{0,80} breathe\w* .{0,20} last)\b", re.I), ("died_city",)),
    (re.compile(r"\b(?:leads?|in charge of|highest-ranking official of)\b.{0,100}\b(?:company|organization|institution)\b", re.I), ("ceo", "chairperson")),
)


def absorbing_terminal_relation_weights(query: str) -> dict[str, float]:
    weights = superposed_terminal_relation_weights(query)
    for position, (pattern, relations) in enumerate(ABSORBING_HEAD_RULES):
        if not pattern.search(query):
            continue
        for relation_position, relation in enumerate(relations):
            weights[relation] = max(
                weights.get(relation, 0.0),
                1.72 - 0.05 * relation_position - 0.01 * position,
            )
    return dict(sorted(weights.items()))


@dataclass(frozen=True)
class DocumentStarResult:
    """A bounded document orbit mapped back to source-window indices."""

    enabled: bool = False
    top_sources: tuple[int, ...] = ()
    orbit_sources: tuple[int, ...] = ()
    best_sources: tuple[int, ...] = ()
    score_ratio: float = 1.0
    score_margin: float = 0.0
    documents: int = 0


def _common_overlap(left: str, right: str, ceiling: int = 512) -> int:
    """Return the largest exact suffix/prefix overlap within a small bound."""
    upper = min(len(left), len(right), ceiling)
    for width in range(upper, 31, -1):
        if left[-width:] == right[:width]:
            return width
    return 0


class DocumentStarField:
    """Recover document-level stars from a regular overlapping source stream."""

    def __init__(
        self, texts: Sequence[str], *, max_documents: int = 16,
        focal_documents: int = 2,
    ) -> None:
        self.texts = list(texts)
        self.max_documents = min(16, max(1, int(max_documents)))
        self.focal_documents = min(
            self.max_documents, max(1, int(focal_documents)),
        )
        self.enabled = False
        self.documents: list[str] = []
        self.document_sources: list[tuple[int, ...]] = []
        self._document_index: BM25 | None = None
        self._source_index: BM25 | None = None
        if len(self.texts) < 20:
            return
        # A cheap lexical gate avoids rebuilding long EventQA/dialogue streams
        # that cannot possibly satisfy the 20-document activation boundary.
        marker_hits = sum(
            len(DOCUMENT_BOUNDARY_RE.findall(text)) for text in self.texts
        )
        if marker_hits < 10:
            return

        overlaps = [
            _common_overlap(left, right)
            for left, right in zip(self.texts, self.texts[1:])
        ]
        positive = [value for value in overlaps if value]
        if len(positive) < max(4, int(0.80 * len(overlaps))):
            return
        overlap = max(set(positive), key=lambda value: (positive.count(value), value))
        if sum(value == overlap for value in overlaps) < int(0.75 * len(overlaps)):
            return

        starts = [0]
        rebuilt = self.texts[0]
        for text in self.texts[1:]:
            starts.append(len(rebuilt) - overlap)
            rebuilt += text[overlap:]
        boundaries = list(DOCUMENT_BOUNDARY_RE.finditer(rebuilt))
        if len(boundaries) < 20:
            return

        documents: list[str] = []
        sources: list[tuple[int, ...]] = []
        for number, match in enumerate(boundaries):
            end = (
                boundaries[number + 1].start()
                if number + 1 < len(boundaries)
                else len(rebuilt)
            )
            documents.append(rebuilt[match.start():end])
            source_orbit = tuple(
                source
                for source, start in enumerate(starts)
                if start < end and start + len(self.texts[source]) > match.start()
            )
            sources.append(source_orbit)

        self.documents = documents
        self.document_sources = sources
        self._document_index = BM25(documents)
        self._source_index = BM25(self.texts)
        self.enabled = True

    def solve(self, query: str, *, size: int) -> DocumentStarResult:
        if (
            not self.enabled
            or self._document_index is None
            or self._source_index is None
            or size != len(self.texts)
        ):
            return DocumentStarResult()
        scores = self._document_index.scores(query)
        order = ranking(scores)[:self.max_documents]
        if not order or scores[order[0]] <= 0.0:
            return DocumentStarResult(enabled=True, documents=len(self.documents))
        source_scores = self._source_index.scores(query)
        visible_orbits = [
            tuple(source for source in self.document_sources[document] if source < size)
            for document in order
        ]
        visible_orbits = [orbit for orbit in visible_orbits if orbit]
        if not visible_orbits:
            return DocumentStarResult(enabled=True, documents=len(self.documents))
        best_sources = tuple(
            max(orbit, key=lambda source: (source_scores[source], -source))
            for orbit in visible_orbits
        )
        second = scores[order[1]] if len(order) > 1 else 0.0
        return DocumentStarResult(
            enabled=True,
            top_sources=tuple(dict.fromkeys(
                source
                for orbit in visible_orbits[:self.focal_documents]
                for source in orbit
            )),
            orbit_sources=tuple(dict.fromkeys(
                source for orbit in visible_orbits for source in orbit
            )),
            best_sources=best_sources,
            score_ratio=scores[order[0]] / max(1e-9, second),
            score_margin=scores[order[0]] - second,
            documents=len(self.documents),
        )


class AbsorbingRelationGraph(RelationSuperpositionGraph):
    """Collapse once all query-supported terminal relations are satisfied."""

    complete_terminal_absorption = True
    terminal_repeat_penalty = 1.25
    entity_cycle_guard = True

    def _terminal_weights(self, query: str) -> dict[str, float]:
        return absorbing_terminal_relation_weights(query)
