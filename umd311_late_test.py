from __future__ import annotations

import unittest

import numpy as np

from umd311_late import UMD311LateConfig, UMD311LateInteractionReranker


class FakeLateEncoder:
    def query_embed(self, texts, **kwargs):
        return iter([np.array([[1.0, 0.0]]) for _ in texts])

    def passage_embed(self, texts, **kwargs):
        vectors = {
            "weak": np.array([[0.0, 1.0]]),
            "strong": np.array([[1.0, 0.0]]),
            "neighbor": np.array([[0.8, 0.2]]),
            "outside": np.array([[1.0, 0.0]]),
        }
        return iter([vectors[text] for text in texts])


class UMD311LateTests(unittest.TestCase):
    def test_maxsim_promotes_semantic_candidate(self) -> None:
        reranker = UMD311LateInteractionReranker(FakeLateEncoder())
        order = reranker.rerank(
            "query", ["weak", "strong", "neighbor"], [0, 1, 2],
            [0.9, 0.4, 0.3], [0, 1, 1],
        )
        self.assertEqual(order[0], 1)

    def test_candidate_limit_bounds_encoding_and_preserves_tail(self) -> None:
        reranker = UMD311LateInteractionReranker(
            FakeLateEncoder(), config=UMD311LateConfig(candidate_limit=2)
        )
        order = reranker.rerank(
            "query", ["weak", "strong", "outside"], [0, 1, 2],
            [0.9, 0.4, 1.0], [0, 0, 1],
        )
        self.assertEqual(order[-1], 2)
        self.assertEqual(set(order), {0, 1, 2})


if __name__ == "__main__":
    unittest.main()
