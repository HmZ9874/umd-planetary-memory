"""Validate the audited public UMD result snapshot without benchmark data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmarks" / "results"


FILES: dict[str, tuple[str, int]] = {
    "umd3282_exam_locomo.json": ("locomo", 1986),
    "umd3282_exam_longmemeval.json": ("longmemeval", 500),
    "umd3282_exam_membench.json": ("membench", 550),
    "umd3282_exam_memora.json": ("memora", 600),
    "umd3282_exam_memoryarena.json": ("memoryarena", 4149),
    "umd3282_exam_memorybench.json": ("memorybench", 51),
    "umd329_exam_evermembench_full.json": ("evermembench", 3121),
    "umd329_exam_longmemeval_v2_full.json": ("longmemeval_v2", 451),
    "umd329_exam_memoryagentbench_full.json": ("memoryagentbench", 2800),
    "umd330_exam_memoryagentbench_full.json": ("memoryagentbench", 2800),
}


def _question_count(result: dict[str, Any]) -> int:
    for key in ("questions", "sequential_queries", "trajectories"):
        value = result.get(key)
        if isinstance(value, int):
            return value
    raise AssertionError("result has no recognized question-count field")


def _assert_probability_tree(value: Any, path: str = "root") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if key in {"mrr", "accuracy", "exact_set_rate", "micro_precision"}:
                assert isinstance(child, (int, float)) and 0.0 <= child <= 1.0, child_path
            if key in {"any_evidence_recall", "full_evidence_recall", "micro_evidence_recall"}:
                assert isinstance(child, dict), child_path
                for rank, score in child.items():
                    assert rank.isdigit(), child_path
                    assert isinstance(score, (int, float)) and 0.0 <= score <= 1.0, child_path
            _assert_probability_tree(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_probability_tree(child, f"{path}[{index}]")


def main() -> None:
    validated = 0
    for filename, (root_key, expected_questions) in FILES.items():
        path = RESULTS / filename
        assert path.is_file(), f"missing {path}"
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert root_key in payload, f"{filename}: missing root key {root_key}"
        result = payload[root_key]
        assert _question_count(result) == expected_questions, filename
        ranking_flag = result.get("gold_used_for_ranking")
        if ranking_flag is None:
            metadata = payload.get("metadata", {})
            ranking_flag = metadata.get("gold_used_for_ranking")
        assert ranking_flag is False, filename
        _assert_probability_tree(result, filename)
        validated += 1
    print(f"validated_results={validated} gold_used_for_ranking=false status=pass")


if __name__ == "__main__":
    main()
