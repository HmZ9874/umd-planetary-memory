from __future__ import annotations

import unittest

from benchmarks.umd330_adapter import FactSlingshotGraph


def _filler(start: int, count: int) -> list[str]:
    return [
        f"{number}. Person {number} is a citizen of Country {number}."
        for number in range(start, start + count)
    ]


class UMD330SlingshotTests(unittest.TestCase):
    def test_three_hop_slingshot_finds_requested_relation(self) -> None:
        texts = [
            "\n".join(_filler(1, 8) + [
                "20. Phoenix Inferno is associated with the sport of skyball.",
            ]),
            "\n".join(_filler(30, 8) + [
                "40. skyball was created in the country of Italy.",
            ]),
            "\n".join(_filler(50, 8) + [
                "60. The name of the current head of state in Italy is Sergio Mattarella.",
            ]),
        ]
        result = FactSlingshotGraph(texts).solve(
            "What is the name of the current head of state of the country "
            "where the sport Phoenix Inferno came from?",
            size=len(texts),
        )
        self.assertEqual(result.answer, "Sergio Mattarella")
        self.assertEqual(result.primary_sources, (2,))
        self.assertEqual([relation for _, relation, _ in result.path], [
            "sport", "created_country", "head_state",
        ])

    def test_latest_visible_fact_wins_and_echoes_are_label_blind(self) -> None:
        texts = [
            "\n".join(_filler(1, 10) + [
                "20. Bagratuni Dynasty is affiliated with the religion of Christianity.",
                "21. Christianity was founded in the city of Jerusalem.",
            ]),
            "\n".join(_filler(30, 10) + [
                "45. Christianity was founded in the city of Taipei.",
            ]),
            "46. A separate note mentions Taipei.",
        ]
        graph = FactSlingshotGraph(texts)
        current = graph.solve(
            "Where did the religion associated with the Bagratuni Dynasty "
            "come into existence?",
            size=3,
        )
        historical_prefix = graph.solve(
            "Where did the religion associated with the Bagratuni Dynasty "
            "come into existence?",
            size=1,
        )
        self.assertEqual(current.answer, "Taipei")
        self.assertEqual(current.primary_sources, (1,))
        self.assertEqual(current.echo_sources, (1, 2))
        self.assertEqual(historical_prefix.answer, "Jerusalem")

    def test_ordinary_dialogue_does_not_activate_fact_graph(self) -> None:
        graph = FactSlingshotGraph([
            "User: I visited Taipei yesterday.",
            "Assistant: That sounds fun.",
        ])
        self.assertFalse(graph.enabled)
        self.assertFalse(graph.solve("Where did I visit?", size=2).primary_sources)


if __name__ == "__main__":
    unittest.main()
