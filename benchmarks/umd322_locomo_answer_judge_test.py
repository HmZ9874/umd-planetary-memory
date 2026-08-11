from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.umd322_locomo_answer_judge import (
    ANSWER_SYSTEM,
    answer_prompt,
    estimate,
    estimated_cost,
    parse_judgment,
    strip_answer,
    summarize,
    token_f1,
)


class UMD322LoCoMoAnswerJudgeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = [
            {
                "question_id": "0:0",
                "category": "1",
                "question": "Where did Ana go?",
                "reference_answer": "Paris",
                "context": "[source=D1:1] Ana: I went to Paris.",
                "gold_used_for_ranking": False,
            },
            {
                "question_id": "0:1",
                "category": "2",
                "question": "When?",
                "reference_answer": "7 May 2023",
                "context": "[source=D1:2] Ana: It happened yesterday.",
                "gold_used_for_ranking": False,
            },
        ]

    def test_answer_prompt_does_not_contain_gold(self) -> None:
        prompt = answer_prompt(self.rows[0])
        self.assertIn("Where did Ana go?", prompt)
        self.assertNotIn("Reference answer", prompt)
        self.assertNotIn(self.rows[0]["reference_answer"], ANSWER_SYSTEM)

    def test_answer_and_judge_parsing(self) -> None:
        self.assertEqual(strip_answer("thinking\nANSWER: Paris"), "Paris")
        self.assertEqual(parse_judgment('{"reasoning":"same", "label":"correct"}'), ("CORRECT", "same"))
        self.assertEqual(parse_judgment("Result: WRONG")[0], "WRONG")
        self.assertEqual(parse_judgment("unknown")[0], "PARSE_ERROR")

    def test_estimate_and_cost(self) -> None:
        planning = estimate(self.rows, assumed_answer_output=10, assumed_judge_output=5)
        self.assertEqual(planning["requests"], 4)
        self.assertEqual(planning["answer_output_tokens_assumed"], 20)
        cost = estimated_cost(planning, 1.0, 2.0, 3.0, 4.0)
        self.assertIsNotNone(cost)
        assert cost is not None
        expected = (
            planning["answer_input_tokens_estimated"] * 1.0
            + planning["answer_output_tokens_assumed"] * 2.0
            + planning["judge_input_tokens_estimated"] * 3.0
            + planning["judge_output_tokens_assumed"] * 4.0
        ) / 1_000_000
        self.assertAlmostEqual(cost["total_usd"], expected)

    def test_token_f1(self) -> None:
        self.assertEqual(token_f1("Berlin", "Paris"), 0.0)
        self.assertGreater(token_f1("She went to Paris", "Paris"), 0.0)

    def test_summary_uses_only_valid_judgments(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            answers = base / "answers.jsonl"
            judgments = base / "judgments.jsonl"
            answers.write_text(
                "\n".join(
                    json.dumps({"question_id": row["question_id"], "generated_answer": row["reference_answer"]})
                    for row in self.rows
                )
                + "\n",
                encoding="utf-8",
            )
            judgments.write_text(
                json.dumps({"question_id": "0:0", "label": "CORRECT"})
                + "\n"
                + json.dumps({"question_id": "0:1", "label": "PARSE_ERROR"})
                + "\n",
                encoding="utf-8",
            )
            report = summarize(self.rows, answers, judgments)
            self.assertEqual(report["judge_accuracy"], 1.0)
            self.assertEqual(report["judge_parse_errors"], 1)
            self.assertEqual(report["diagnostic_token_f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
