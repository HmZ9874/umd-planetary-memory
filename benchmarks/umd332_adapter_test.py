from __future__ import annotations

import unittest

import numpy as np

from benchmarks.ai_memory_extended import OrbitIndex, _bounded_lexical_bridge
from benchmarks.public_benchmarks import BM25, ranking
from benchmarks.umd332_adapter import (
    RelationSuperpositionGraph,
    superposed_terminal_relation_weights,
)


def _filler(start: int = 1, count: int = 20) -> list[str]:
    return [
        f"{number}. Person {number} is a citizen of Country {number}."
        for number in range(start, start + count)
    ]


class UMD332RelationSuperpositionTests(unittest.TestCase):
    def test_bounded_lexical_bridge_follows_a_rare_intermediate_entity(self) -> None:
        texts = [
            "The fight song Blue Sky belongs to North University.",
            "North University is based in the city of Avalon.",
            "South College is based in another city.",
        ]
        index = BM25(texts)
        query = "Which city is home to the university whose fight song is Blue Sky?"
        order = ranking(index.scores(query))
        bridge = _bounded_lexical_bridge(query, texts, index, order)
        self.assertGreater(bridge[1], bridge[2])

    def test_nested_relations_collapse_at_deepest_supported_boundary(self) -> None:
        texts = ["\n".join(_filler() + [
            "30. Our Mutual Friend was written by Charles Dickens.",
            "31. Charles Dickens is married to Catherine Dickens.",
            "32. Catherine Dickens is a citizen of Belgium.",
        ])]
        # Use the fact grammar's author wording in the fixture.
        texts[0] = texts[0].replace(
            "Our Mutual Friend was written by Charles Dickens.",
            "The author of Our Mutual Friend is Charles Dickens.",
        )
        result = RelationSuperpositionGraph(texts).solve(
            "What is the country of citizenship of the spouse of the author of Our Mutual Friend?",
            size=1,
        )
        self.assertEqual(result.answer, "Belgium")
        self.assertEqual(
            [step[1] for step in result.path], ["author", "married", "citizen"],
        )
        self.assertTrue({"author", "married", "citizen"} <= set(result.terminal_relations))

    def test_origin_inflection_adds_created_country_terminal(self) -> None:
        weights = superposed_terminal_relation_weights(
            "What country does the genre of music associated with the artist originate from?"
        )
        self.assertGreater(weights["created_country"], weights["music"])

    def test_equal_ordinal_overlap_fragment_prefers_complete_object(self) -> None:
        texts = ["\n".join(_filler() + [
            "30. The company that produced Ford F-Series is Ford Moto",
            "30. The company that produced Ford F-Series is Ford Motor Company.",
        ])]
        result = RelationSuperpositionGraph(texts).solve(
            "Which company is Ford F-Series produced by?", size=1,
        )
        self.assertEqual(result.answer, "Ford Motor Company")

    def test_breathe_last_maps_to_death_location(self) -> None:
        texts = ["\n".join(_filler() + [
            "30. The author of Sketches by Boz is Charles Dickens.",
            "31. Charles Dickens died in the city of London.",
        ])]
        result = RelationSuperpositionGraph(texts).solve(
            'At what location did the author of "Sketches by Boz" breathe his last?',
            size=1,
        )
        self.assertEqual(result.answer, "London")

    def test_superseded_answer_is_a_shadow_not_the_primary(self) -> None:
        texts = [
            "\n".join(_filler() + ["30. Ada is a citizen of Belgium."]),
            "40. Ada is a citizen of France.",
        ]
        result = RelationSuperpositionGraph(texts).solve(
            "What is the country of citizenship of Ada?", size=2,
        )
        self.assertEqual(result.answer, "France")
        self.assertEqual(result.primary_sources, (1,))
        self.assertIn(0, result.shadow_sources)

    def test_ordinary_dialogue_remains_outside_fact_universe(self) -> None:
        graph = RelationSuperpositionGraph([
            "User: Which country did the author come from?",
            "Assistant: We were discussing a novel, not a fact table.",
        ])
        self.assertFalse(graph.enabled)

    def test_strict_channel_remains_one_source_per_rank(self) -> None:
        class Encoder:
            def encode_many(self, texts):
                return [np.asarray([1.0, 0.0], dtype=np.float32) for _ in texts]

        texts = ["\n".join(_filler() + [
            "30. The author of Our Mutual Friend is Charles Dickens.",
            "31. Charles Dickens is married to Catherine Dickens.",
            "32. Catherine Dickens is a citizen of Belgium.",
        ]), "33. Belgium appears in an independent evidence source."]
        index = OrbitIndex(
            texts, [0, 1], ["a", "b"], Encoder(), physics_v332=True,
        )
        channels = index.retrieve_compact_channels(
            "What is the country of citizenship of the spouse of the author of Our Mutual Friend?"
        )
        self.assertTrue(all(len(capsule) == 1 for capsule in channels["atomic_answer"]))
        self.assertEqual(channels["atomic_answer"][:2], [("a",), ("b",)])


if __name__ == "__main__":
    unittest.main()
