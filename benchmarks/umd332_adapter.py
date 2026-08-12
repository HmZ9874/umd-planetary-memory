"""UMD 3.32 gold-blind relation-superposition evidence closure.

UMD 3.31 fixed several output-slot mistakes, but it still committed to the
first matching terminal rule.  Nested questions can mention three legitimate
relations (for example author -> spouse -> citizenship), so a single terminal
choice may stop at an intermediate edge.  UMD 3.32 keeps every query-supported
relation in a bounded superposition field and lets path coverage collapse the
field.  Only query text and visible source text are accepted as inputs.
"""

from __future__ import annotations

import re
from collections import defaultdict, deque

from benchmarks.umd330_adapter import RELATION_CUES, GraphFact, _norm, _relation_contact
from benchmarks.umd331_adapter import (
    OUTPUT_SLOT_RULES,
    EvidenceClosureGraph,
    EvidenceClosureResult,
)


# High-confidence answer-head grammar.  These patterns identify the requested
# value type while remaining independent of any benchmark answer or evidence.
ANSWER_HEAD_RULES: tuple[tuple[re.Pattern[str], tuple[str, ...]], ...] = (
    (re.compile(r"\b(?:which|what|in which)\s+languages?\b.{0,100}\b(?:official|government|officially|official documents?)\b", re.I), ("official_language",)),
    (re.compile(r"\b(?:which|what|in which)\s+languages?\b.{0,100}\b(?:speak|speaks|spoken|write|sign|communicat)\w*\b", re.I), ("speaks_language",)),
    (re.compile(r"\blanguage of (?:the )?(?:work|book|novel|composition)\b|\b(?:work|book|novel)\b.{0,60}\bwritten (?:in|using)\b", re.I), ("written_language", "original_language")),
    (re.compile(r"\b(?:which|what)\s+(?:country|nation)\b.{0,140}\b(?:citizenship|citizen|nationality)\b|\bhold\w* citizenship\b", re.I), ("citizen",)),
    (re.compile(r"\b(?:which|what|from what|from which)\s+country\b.{0,160}\b(?:created|creation|originat\w*|founded|founding|hail\w*|come from|came from)\b", re.I), ("created_country",)),
    (re.compile(r"\b(?:which|what|at which|in what)\s+(?:city|place|location)\b.{0,120}\b(?:die|died|death|pass(?:ed)? away|breathe\w* (?:his|her|their) last)\b|\bwhere\b.{0,80}\b(?:die|died|pass(?:ed)? away)\b", re.I), ("died_city",)),
    (re.compile(r"\b(?:which|what)\s+city\b.{0,100}\b(?:born|birth)\b|\bwhere\b.{0,80}\bborn\b", re.I), ("born_city",)),
    (re.compile(r"\b(?:which|what)\s+(?:city|place|location)\b.{0,100}\bfound(?:ed|ing)?\b|\bwhere was\b.{0,100}\bfounded\b|\bplace of origin\b", re.I), ("founded_city",)),
    (re.compile(r"\b(?:who|which (?:historical |religious )?(?:person|leader|figure)|what (?:person|leader)|name of the person)\b.{0,140}\b(?:found|establish)\w*\b", re.I), ("founded_by",)),
    (re.compile(r"\b(?:significant|notable|famous)\s+(?:creation|work)\b|\b(?:known|famous) for\b", re.I), ("famous_for",)),
    (re.compile(r"\b(?:current )?head of (?:the )?.{0,50}government\b|\bgovernment (?:leader|chief executive)\b", re.I), ("head_government",)),
    (re.compile(r"\b(?:current )?head of state\b|\bchief public representative\b", re.I), ("head_state",)),
    (re.compile(r"\b(?:seat|center|centre) of government\b|\bcapital city\b", re.I), ("capital",)),
    (re.compile(r"\b(?:educational institution|educational organization|which school|which university|in which school)\b", re.I), ("educated_at",)),
    (re.compile(r"\b(?:directed or managed|who directed|person who directed|manages? the (?:network|broadcaster))\b", re.I), ("director",)),
    (re.compile(r"\b(?:headquarter|headquarters)\b", re.I), ("headquarters_city",)),
    (re.compile(r"\b(?:location|place) of (?:the )?work\b|\bwork location\b|\bwhere\b.{0,80}\bwork(?:ed|s)?\b|\bspend\w* most of their work hours\b", re.I), ("worked_city",)),
    (re.compile(r"\b(?:what kind of work|occupation|profession|field of work)\b", re.I), ("works_field",)),
    (re.compile(r"\bnotable work\b.{0,100}\b(?:language|write|written)\b|\b(?:language|write|written)\b.{0,100}\bnotable work\b", re.I), ("written_language",)),
    (re.compile(r"\b(?:die|dies|died|dying|death|pass(?:ed)? away)\b", re.I), ("died_city",)),
)


def superposed_terminal_relation_weights(query: str) -> dict[str, float]:
    """Return every query-supported terminal relation with confidence mass."""
    weights: dict[str, float] = {}
    for position, (pattern, relations) in enumerate(ANSWER_HEAD_RULES):
        if not pattern.search(query):
            continue
        for relation_position, relation in enumerate(relations):
            mass = 1.60 - 0.05 * relation_position - 0.01 * position
            weights[relation] = max(weights.get(relation, 0.0), mass)

    # Unlike 3.31, do not return after the first rule.  Every matched relation
    # becomes a possible boundary; the connected path decides which one wins.
    for position, (pattern, relations) in enumerate(OUTPUT_SLOT_RULES):
        if not pattern.search(query):
            continue
        for relation_position, relation in enumerate(relations):
            mass = 0.92 - 0.04 * relation_position - 0.005 * position
            weights[relation] = max(weights.get(relation, 0.0), mass)

    lowered = query.casefold()
    for relation, cues in RELATION_CUES.items():
        cue_length = max((len(cue) for cue in cues if cue in lowered), default=0)
        if cue_length:
            mass = 0.56 + min(0.24, cue_length / 80.0)
            weights[relation] = max(weights.get(relation, 0.0), mass)
    return dict(sorted(weights.items()))


class RelationSuperpositionGraph(EvidenceClosureGraph):
    """Five-hop, multi-terminal closure whose field collapses by path coverage."""

    complete_terminal_absorption = False
    terminal_repeat_penalty = 0.0
    entity_cycle_guard = False

    def _terminal_weights(self, query: str) -> dict[str, float]:
        return superposed_terminal_relation_weights(query)

    def __init__(self, texts) -> None:
        super().__init__(texts)
        versions: dict[tuple[str, str], list[GraphFact]] = defaultdict(list)
        for fact in self.facts:
            versions[(_norm(fact.subject), fact.relation)].append(fact)
        self._versions_by_key = {
            key: tuple(sorted(facts, key=lambda fact: -fact.ordinal))
            for key, facts in versions.items()
        }

    def _current_adjacency(self, size: int) -> dict[str, list[GraphFact]]:
        """Select the newest edge and repair equal-ordinal overlap fragments."""
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
            prefer_equal_ordinal = (
                prior is not None
                and visible.ordinal == prior.ordinal
                and (len(_norm(visible.object)), len(visible.sources), visible.object)
                > (len(_norm(prior.object)), len(prior.sources), prior.object)
            )
            if prior is None or visible.ordinal > prior.ordinal or prefer_equal_ordinal:
                current[key] = visible
        adjacency: dict[str, list[GraphFact]] = defaultdict(list)
        for (subject, _), fact in current.items():
            adjacency[subject].append(fact)
        return adjacency

    def solve(
        self, query: str, *, size: int, max_hops: int = 5,
    ) -> EvidenceClosureResult:
        if not self.enabled or size <= 0:
            return EvidenceClosureResult()
        terminal = self._terminal_weights(query)
        if not terminal:
            return EvidenceClosureResult()
        query_norm = _norm(query)
        anchors = [subject for subject in self._subjects if subject in query_norm][:12]
        if not anchors:
            return EvidenceClosureResult(terminal_relations=tuple(terminal))

        adjacency = self._current_adjacency(size)
        queue = deque((anchor, (), 0.0, frozenset()) for anchor in anchors)
        visited: dict[tuple[str, int, frozenset[str]], float] = {}
        solutions: list[
            tuple[float, int, int, GraphFact, tuple[GraphFact, ...]]
        ] = []
        while queue:
            entity, path, score, covered = queue.popleft()
            depth = len(path)
            if depth > max_hops:
                continue
            state = (entity, depth, covered)
            if visited.get(state, float("-inf")) >= score:
                continue
            visited[state] = score
            for fact in adjacency.get(entity, ()):
                if self.entity_cycle_guard:
                    visited_entities = {
                        _norm(step.subject) for step in path
                    } | {_norm(step.object) for step in path}
                    if _norm(fact.object) in visited_entities:
                        continue
                contact = _relation_contact(query, fact.relation)
                if fact.relation in terminal:
                    contact = max(contact, 5.0 * terminal[fact.relation])
                # Unmentioned bridges remain possible but lose energy.  A
                # fully query-supported dependency chain therefore outranks a
                # short intermediate stop or an arbitrary long walk.
                edge_energy = (
                    0.20 + min(3.0, contact / 5.0) if contact else -0.18
                )
                edge_energy -= self.terminal_repeat_penalty * sum(
                    step.relation == fact.relation for step in path
                )
                next_path = path + (fact,)
                next_covered = covered | (
                    {fact.relation} if fact.relation in terminal else set()
                )
                next_score = score + edge_energy
                terminal_mass = terminal.get(fact.relation)
                if terminal_mass is not None:
                    coverage_mass = len(next_covered) / max(1, len(terminal))
                    solutions.append((
                        next_score + 8.0 + 1.35 * terminal_mass + 1.10 * coverage_mass,
                        len(next_path), fact.ordinal, fact, next_path,
                    ))
                complete_terminal = set(terminal) <= set(next_covered)
                if (
                    len(next_path) < max_hops
                    and not (
                        self.complete_terminal_absorption
                        and terminal_mass is not None
                        and complete_terminal
                    )
                ):
                    queue.append((
                        _norm(fact.object), next_path, next_score,
                        frozenset(next_covered),
                    ))
        if not solutions:
            return EvidenceClosureResult(terminal_relations=tuple(terminal))

        # Aggregate independent paths that predict the same object.  This is a
        # small deterministic analogue of measurement consensus: one noisy
        # path cannot beat several mutually supporting paths merely by tie.
        by_answer: dict[str, list[tuple[float, int, int, GraphFact, tuple[GraphFact, ...]]]] = defaultdict(list)
        for solution in solutions:
            by_answer[_norm(solution[3].object)].append(solution)
        collapsed: list[
            tuple[float, int, int, GraphFact, tuple[GraphFact, ...]]
        ] = []
        for answer_solutions in by_answer.values():
            best = max(answer_solutions, key=lambda item: item[:3])
            consensus = min(0.45, 0.08 * (len(answer_solutions) - 1))
            collapsed.append((best[0] + consensus, *best[1:]))

        _, _, _, answer_fact, best_path = max(collapsed, key=lambda item: item[:3])
        answer_norm = _norm(answer_fact.object)
        echo = tuple(
            source for source, text in enumerate(self._normalized_texts[:size])
            if answer_norm and answer_norm in text
        )
        dependency = tuple(dict.fromkeys(
            source
            for fact in best_path
            for source in fact.sources
            if source < size and source not in answer_fact.sources
        ))
        # Preserve superseded answer versions as a shadow orbit after the
        # current answer echoes.  This supports conflict audit and uncertainty
        # without displacing the newest fact from rank one.
        shadow: list[int] = []
        answer_key = (_norm(answer_fact.subject), answer_fact.relation)
        for version in self._versions_by_key.get(answer_key, ()):
            if version.ordinal == answer_fact.ordinal and _norm(version.object) == answer_norm:
                continue
            visible_sources = [source for source in version.sources if source < size]
            version_norm = _norm(version.object)
            version_echo = [
                source for source, text in enumerate(self._normalized_texts[:size])
                if version_norm and version_norm in text
            ]
            shadow.extend(visible_sources)
            shadow.extend(version_echo)
        return EvidenceClosureResult(
            primary_sources=answer_fact.sources,
            echo_sources=echo,
            shadow_sources=tuple(dict.fromkeys(shadow)),
            dependency_sources=dependency,
            answer=answer_fact.object,
            path=tuple((fact.subject, fact.relation, fact.object) for fact in best_path),
            terminal_relations=tuple(terminal),
        )
