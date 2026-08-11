"""UMD 3.30 gold-blind multi-hop gravitational slingshot retrieval.

The adapter is deliberately narrow: it activates only for dense numbered
subject-relation-object fact universes.  Dialogue and ordinary prose keep the
frozen UMD 3.29 path.  A query entity becomes the launch star, current facts
form directed planetary orbits, and a bounded breadth-first slingshot follows
at most three relations to the query's requested relation.  No answer or
evidence label is an input.
"""

from __future__ import annotations

import re
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Sequence


NUMBERED_FACT_RE = re.compile(r"^\s*(\d+)\.\s*(.+?)\s*$")
WORD_RE = re.compile(r"[\w]+", re.UNICODE)


def _norm(value: str) -> str:
    return " ".join(WORD_RE.findall(value.casefold()))


# Ordered from the most syntactically specific templates to generic ones.
FACT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("music", re.compile(r"^The type of music that (.+?) plays is (.+?)\.?$")),
    ("author", re.compile(r"^The author of (.+?) is (.+?)\.?$")),
    ("capital", re.compile(r"^The capital of (.+?) is (.+?)\.?$")),
    ("educated_at", re.compile(r"^The univer(?:sity|isty) where (.+?) was educated is (.+?)\.?$")),
    ("produced_by", re.compile(r"^The company that produced (.+?) is (.+?)\.?$")),
    ("ceo", re.compile(r"^The chief executive officer of (.+?) is (.+?)\.?$")),
    ("head_state", re.compile(r"^The name of the current head of state in (.+?) is (.+?)\.?$")),
    ("head_government", re.compile(r"^The name of the current head of the (.+?) government is (.+?)\.?$")),
    ("head_government", re.compile(r"^The name of the current head of government in (.+?) is (.+?)\.?$")),
    ("head_coach", re.compile(r"^The head coach of (.+?) is (.+?)\.?$")),
    ("headquarters_city", re.compile(r"^The headquarters of (.+?) is located in the city of (.+?)\.?$")),
    ("official_language", re.compile(r"^The official language of (.+?) is (.+?)\.?$")),
    ("original_language", re.compile(r"^The original language of (.+?) is (.+?)\.?$")),
    ("director", re.compile(r"^The director of (.+?) is (.+?)\.?$")),
    ("chairperson", re.compile(r"^The chairperson of (.+?) is (.+?)\.?$")),
    ("child", re.compile(r"^(.+?)'s child is (.+?)\.?$")),
    ("born_city", re.compile(r"^(.+?) was born in the city of (.+?)\.?$")),
    ("died_city", re.compile(r"^(.+?) died in the city of (.+?)\.?$")),
    ("founded_city", re.compile(r"^(.+?) was founded in the city of (.+?)\.?$")),
    ("founded_by", re.compile(r"^(.+?) was founded by (.+?)\.?$")),
    ("developed_by", re.compile(r"^(.+?) was developed by (.+?)\.?$")),
    ("created_country", re.compile(r"^(.+?) was created in the country of (.+?)\.?$")),
    ("created_by", re.compile(r"^(.+?) was created by (.+?)\.?$")),
    ("performed_by", re.compile(r"^(.+?) was performed by (.+?)\.?$")),
    ("written_language", re.compile(r"^(.+?) was written in the language of (.+?)\.?$")),
    ("married", re.compile(r"^(.+?) is married to (.+?)\.?$")),
    ("sport", re.compile(r"^(.+?) is associated with the sport of (.+?)\.?$")),
    ("religion", re.compile(r"^(.+?) is affiliated with the religion of (.+?)\.?$")),
    ("citizen", re.compile(r"^(.+?) is a citizen of (.+?)\.?$")),
    ("continent", re.compile(r"^(.+?) is located in the continent of (.+?)\.?$")),
    ("position", re.compile(r"^(.+?) plays the position of (.+?)\.?$")),
    ("speaks_language", re.compile(r"^(.+?) speaks the language of (.+?)\.?$")),
    ("works_field", re.compile(r"^(.+?) works in the field of (.+?)\.?$")),
    ("employed_by", re.compile(r"^(.+?) is employed by (.+?)\.?$")),
    ("famous_for", re.compile(r"^(.+?) is famous for (.+?)\.?$")),
)


RELATION_CUES: dict[str, tuple[str, ...]] = {
    "music": ("music", "genre"),
    "author": ("author", "wrote", "written by"),
    "capital": ("capital",),
    "educated_at": ("educated", "university", "school"),
    "produced_by": ("produced", "manufacturer", "company"),
    "ceo": ("chief executive", "ceo"),
    "head_state": ("head of state", "chief public representative"),
    "head_government": ("head of government", "government leader"),
    "head_coach": ("head coach", "coach"),
    "headquarters_city": ("headquarters",),
    "official_language": ("official language", "language is recognized", "language is used"),
    "original_language": ("original language",),
    "director": ("director",),
    "chairperson": ("chairperson", "chairman", "chairwoman"),
    "child": ("child", "son", "daughter"),
    "born_city": ("birthplace", "born"),
    "died_city": ("died", "death"),
    "founded_city": ("come into existence", "founded in", "origin city"),
    "founded_by": ("founder", "founded by", "who founded"),
    "developed_by": ("developed", "developer"),
    "created_country": ("country where", "country did", "country of origin", "originated", "originally hail", "came from"),
    "created_by": ("creator", "created by"),
    "performed_by": ("performer", "performed", "sang", "singer"),
    "written_language": ("written language", "written in"),
    "married": ("spouse", "married", "wife", "husband"),
    "sport": ("sport",),
    "religion": ("religion", "faith"),
    "citizen": ("citizen", "citizenship", "nationality"),
    "continent": ("continent",),
    "position": ("position",),
    "speaks_language": ("speaks", "spoken language"),
    "works_field": ("occupation", "profession", "works in", "field of"),
    "employed_by": ("employed", "employer", "works for"),
    "famous_for": ("famous", "known for"),
}


@dataclass(frozen=True)
class GraphFact:
    subject: str
    relation: str
    object: str
    ordinal: int
    sources: tuple[int, ...]


@dataclass(frozen=True)
class SlingshotResult:
    primary_sources: tuple[int, ...] = ()
    echo_sources: tuple[int, ...] = ()
    answer: str = ""
    path: tuple[tuple[str, str, str], ...] = ()


def _parse_statement(statement: str) -> tuple[str, str, str] | None:
    for relation, pattern in FACT_PATTERNS:
        match = pattern.match(statement)
        if match:
            subject, obj = (part.strip().rstrip(".") for part in match.groups())
            if subject and obj:
                return subject, relation, obj
    return None


def _target_relations(query: str) -> set[str]:
    lowered = query.casefold()
    scored: list[tuple[int, str]] = []
    for relation, cues in RELATION_CUES.items():
        for cue in cues:
            if cue in lowered:
                scored.append((len(cue), relation))
    if not scored:
        return set()
    longest = max(length for length, _ in scored)
    # Preserve equally explicit targets, but do not let a generic word such as
    # "sport" override the requested output "continent".
    return {relation for length, relation in scored if length == longest}


def _relation_contact(query: str, relation: str) -> float:
    lowered = query.casefold()
    return max(
        (float(len(cue)) for cue in RELATION_CUES.get(relation, ()) if cue in lowered),
        default=0.0,
    )


class FactSlingshotGraph:
    """Version-aware graph over dense numbered natural-language triples."""

    def __init__(self, texts: Sequence[str]) -> None:
        collected: dict[tuple[int, str], tuple[str, str, str, set[int]]] = {}
        for source, text in enumerate(texts):
            for raw_line in text.splitlines():
                numbered = NUMBERED_FACT_RE.match(raw_line)
                if not numbered:
                    continue
                ordinal = int(numbered.group(1))
                statement = numbered.group(2).strip()
                parsed = _parse_statement(statement)
                if parsed is None:
                    continue
                key = (ordinal, _norm(statement))
                current = collected.get(key)
                if current is None:
                    subject, relation, obj = parsed
                    collected[key] = (subject, relation, obj, {source})
                else:
                    current[3].add(source)
        self.facts = tuple(
            GraphFact(subject, relation, obj, ordinal, tuple(sorted(sources)))
            for (ordinal, _), (subject, relation, obj, sources) in sorted(collected.items())
        )
        self.texts = list(texts)
        self.enabled = len(self.facts) >= max(20, 2 * len(texts))
        self._subjects = sorted(
            {_norm(fact.subject) for fact in self.facts}, key=lambda value: (-len(value), value),
        )

    def _current_adjacency(self, size: int) -> dict[str, list[GraphFact]]:
        current: dict[tuple[str, str], GraphFact] = {}
        for fact in self.facts:
            visible_sources = tuple(source for source in fact.sources if source < size)
            if not visible_sources:
                continue
            visible = GraphFact(
                fact.subject, fact.relation, fact.object, fact.ordinal, visible_sources,
            )
            key = (_norm(fact.subject), fact.relation)
            prior = current.get(key)
            if prior is None or visible.ordinal > prior.ordinal:
                current[key] = visible
        adjacency: dict[str, list[GraphFact]] = defaultdict(list)
        for (subject, _), fact in current.items():
            adjacency[subject].append(fact)
        return adjacency

    def solve(self, query: str, *, size: int, max_hops: int = 3) -> SlingshotResult:
        if not self.enabled or size <= 0:
            return SlingshotResult()
        targets = _target_relations(query)
        if not targets:
            return SlingshotResult()
        query_norm = _norm(query)
        anchors = [subject for subject in self._subjects if subject in query_norm][:6]
        if not anchors:
            return SlingshotResult()
        adjacency = self._current_adjacency(size)
        queue = deque((anchor, (), 0.0) for anchor in anchors)
        visited: dict[tuple[str, int], float] = {}
        solutions: list[tuple[float, int, int, GraphFact, tuple[GraphFact, ...]]] = []
        while queue:
            entity, path, score = queue.popleft()
            depth = len(path)
            if depth > max_hops:
                continue
            state = (entity, depth)
            if visited.get(state, float("-inf")) >= score:
                continue
            visited[state] = score
            for fact in adjacency.get(entity, ()):
                contact = _relation_contact(query, fact.relation)
                next_score = score + 0.20 + min(3.0, contact / 5.0)
                next_path = path + (fact,)
                if fact.relation in targets:
                    solutions.append((
                        next_score + 6.0, -len(next_path), fact.ordinal, fact, next_path,
                    ))
                if len(next_path) < max_hops:
                    queue.append((_norm(fact.object), next_path, next_score))
        if not solutions:
            return SlingshotResult()
        _, _, _, answer_fact, best_path = max(solutions, key=lambda item: item[:3])
        answer_folded = answer_fact.object.casefold()
        echo = tuple(
            source for source, text in enumerate(self.texts[:size])
            if answer_folded and answer_folded in text.casefold()
        )
        return SlingshotResult(
            primary_sources=answer_fact.sources,
            echo_sources=echo,
            answer=answer_fact.object,
            path=tuple((fact.subject, fact.relation, fact.object) for fact in best_path),
        )

