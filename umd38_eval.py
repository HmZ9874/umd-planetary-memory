"""Model-neutral end-to-end QA evaluation for UMD memory systems.

The same answerer and optional judge instance can be applied to multiple
retrievers, avoiding the invalid comparison between retrieval recall and an
end-to-end answer score.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Iterable, Protocol


@dataclass(frozen=True)
class QACase:
    id: str
    question: str
    answer: str | None
    evidence_ids: frozenset[str] = field(default_factory=frozenset)
    answerable: bool = True


@dataclass(frozen=True)
class AnswerOutput:
    text: str
    citations: tuple[str, ...] = ()


class Answerer(Protocol):
    def __call__(self, question: str, evidence: list[dict[str, object]]) -> AnswerOutput: ...


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+|[\u3400-\u9fff]", text.casefold())


def _f1(prediction: str, reference: str) -> float:
    left, right = Counter(_tokens(prediction)), Counter(_tokens(reference))
    shared = sum((left & right).values())
    if not left or not right:
        return float(left == right)
    precision, recall = shared / sum(left.values()), shared / sum(right.values())
    return 2 * precision * recall / max(1e-12, precision + recall)


class EndToEndQAEvaluator:
    def __init__(
        self, answerer: Answerer,
        judge: Callable[[QACase, AnswerOutput], float] | None = None,
        top_k: int = 8,
    ) -> None:
        self.answerer, self.judge, self.top_k = answerer, judge, top_k

    def evaluate(self, cases: Iterable[QACase], memory) -> dict[str, object]:
        cases = tuple(cases)
        rows = []
        for case in cases:
            hits = memory.retrieve(case.question, top_k=self.top_k)
            evidence = [
                {"memory_id": hit.memory_id, "text": hit.text, "score": hit.force}
                for hit in hits
            ]
            output = self.answerer(case.question, evidence)
            hit_ids = {item["memory_id"] for item in evidence}
            evidence_recall = (
                len(case.evidence_ids & hit_ids) / len(case.evidence_ids)
                if case.evidence_ids else 1.0
            )
            if case.answerable:
                exact = float(" ".join(_tokens(output.text)) == " ".join(_tokens(case.answer or "")))
                token_f1 = _f1(output.text, case.answer or "")
                abstention_correct = 0.0
            else:
                abstained = not _tokens(output.text) or output.text.casefold().strip() in {
                    "unknown", "i don't know", "不知道", "无法回答",
                }
                exact = token_f1 = 0.0
                abstention_correct = float(abstained)
            grounded_citations = (
                sum(item in hit_ids for item in output.citations) / len(output.citations)
                if output.citations else float(not case.answerable)
            )
            rows.append({
                "id": case.id, "prediction": output.text, "citations": output.citations,
                "exact": exact, "token_f1": token_f1,
                "evidence_recall": evidence_recall,
                "grounded_citations": grounded_citations,
                "abstention_correct": abstention_correct,
                "judge": float(self.judge(case, output)) if self.judge else None,
            })
        total = max(1, len(rows))
        unanswerable = [row for row, case in zip(rows, cases) if not case.answerable]
        judge_values = [row["judge"] for row in rows if row["judge"] is not None]
        return {
            "cases": len(rows),
            "exact_match": sum(row["exact"] for row in rows) / total,
            "token_f1": sum(row["token_f1"] for row in rows) / total,
            "evidence_recall": sum(row["evidence_recall"] for row in rows) / total,
            "grounded_citation_rate": sum(row["grounded_citations"] for row in rows) / total,
            "abstention_accuracy": (
                sum(row["abstention_correct"] for row in rows) / max(1, len(unanswerable))
                if unanswerable else None
            ),
            "judge_score": sum(judge_values) / len(judge_values) if judge_values else None,
            "rows": rows,
            "qualification": "end-to-end score; comparability requires the identical answerer and judge",
        }


def compare_retrievers(
    cases: Iterable[QACase], retrievers: dict[str, object], answerer: Answerer,
    judge: Callable[[QACase, AnswerOutput], float] | None = None, top_k: int = 8,
) -> dict[str, dict[str, object]]:
    frozen = tuple(cases)
    evaluator = EndToEndQAEvaluator(answerer, judge, top_k)
    return {name: evaluator.evaluate(frozen, memory) for name, memory in retrievers.items()}
