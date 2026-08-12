from __future__ import annotations

import unittest

import numpy as np

from benchmarks.ai_memory_extended import OrbitIndex
from benchmarks.umd331_adapter import EvidenceClosureGraph, terminal_relation_weights


def _filler(start: int, count: int) -> list[str]:
    return [
        f"{number}. Person {number} is a citizen of Country {number}."
        for number in range(start, start + count)
    ]


class UMD331EvidenceClosureTests(unittest.TestCase):
    def test_output_slot_beats_intermediate_spouse_cue(self) -> None:
        texts = [
            "\n".join(_filler(1, 20) + [
                "30. Igor of Kiev is married to Olga of Kiev.",
                "31. Olga of Kiev died in the city of Rodez.",
            ]),
            "22. A separate source mentions Rodez.",
        ]
        result = EvidenceClosureGraph(texts).solve(
            "In which location did the spouse of Igor of Kiev pass away?",
            size=len(texts),
        )
        self.assertEqual(result.answer, "Rodez")
        self.assertEqual(result.terminal_relations, ("died_city",))
        self.assertEqual(result.echo_sources, (0, 1))
        self.assertEqual([step[1] for step in result.path], ["married", "died_city"])

    def test_terminal_continent_continues_past_citizenship(self) -> None:
        texts = [
            "\n".join(_filler(1, 20) + [
                "30. Colin Irwin is a citizen of Israel.",
                "31. Israel is located in the continent of North America.",
            ]),
        ]
        result = EvidenceClosureGraph(texts).solve(
            "To which continent does the country of citizenship of Colin Irwin pertain?",
            size=1,
        )
        self.assertEqual(result.answer, "North America")
        self.assertEqual([step[1] for step in result.path], ["citizen", "continent"])

    def test_worked_city_template_closes_three_dependencies(self) -> None:
        texts = [
            "\n".join(_filler(1, 20) + [
                "30. John Walsh is affiliated with the religion of Methodism.",
                "31. Methodism was founded by John Knox.",
                "32. John Knox worked in the city of Edinburgh.",
            ]),
        ]
        result = EvidenceClosureGraph(texts).solve(
            "What is the work location of the founder of the religion to which John Walsh belongs?",
            size=1,
        )
        self.assertEqual(result.answer, "Edinburgh")
        self.assertEqual(
            [step[1] for step in result.path],
            ["religion", "founded_by", "worked_city"],
        )

    def test_capital_is_terminal_not_country_of_origin(self) -> None:
        self.assertEqual(
            terminal_relation_weights(
                "Which city serves as the capital of the country of origin for the sport?"
            ),
            {"capital": 1.0},
        )

    def test_ordinary_dialogue_remains_disabled(self) -> None:
        graph = EvidenceClosureGraph([
            "User: My spouse and I visited Rodez.",
            "Assistant: That sounds interesting.",
        ])
        self.assertFalse(graph.enabled)
        self.assertFalse(graph.solve("Where did my spouse visit?", size=2).primary_sources)

    def test_orbit_index_prioritizes_source_atomic_answer_closure(self) -> None:
        class Encoder:
            def encode_many(self, texts):
                return [np.asarray([1.0, 0.0], dtype=np.float32) for _ in texts]

        texts = [
            "\n".join(_filler(1, 20) + [
                "30. Igor of Kiev is married to Olga of Kiev.",
                "31. Olga of Kiev died in the city of Rodez.",
            ]),
            "32. A separate source mentions Rodez.",
        ]
        index = OrbitIndex(
            texts, [0, 1], ["source-a", "source-b"], Encoder(),
            physics_v331=True,
        )
        channels = index.retrieve_compact_channels(
            "In which location did the spouse of Igor of Kiev pass away?"
        )
        self.assertEqual(channels["atomic_answer"][:2], [
            ("source-a",), ("source-b",),
        ])
        self.assertTrue({"source-a", "source-b"} <= {
            source for capsule in channels["answer"][:10] for source in capsule
        })
        self.assertEqual(channels["slingshot"]["terminal_relations"], ["died_city"])


if __name__ == "__main__":
    unittest.main()
