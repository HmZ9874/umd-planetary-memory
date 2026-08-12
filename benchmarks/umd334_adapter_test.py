from __future__ import annotations

import numpy as np

from benchmarks.public_benchmarks import BM25
from benchmarks.umd334_adapter import (
    QueryFissionField,
    QueryFissionResult,
    declarative_variants,
    decompose_query,
    episodic_nucleus,
    should_promote_periapsis,
)


class _Encoder:
    def encode_many(self, texts):
        values = []
        for text in texts:
            lowered = text.casefold()
            vector = np.asarray([
                float("coffee" in lowered),
                float("commute" in lowered),
                float("degree" in lowered),
            ], dtype=np.float32)
            norm = float(np.linalg.norm(vector))
            values.append(vector / norm if norm else np.ones(3, dtype=np.float32) / np.sqrt(3))
        return values


def test_query_fission_conserves_central_and_splits_real_clauses():
    query = "Where did I buy the coffee and how long was my commute?"
    bodies = decompose_query(query)
    assert bodies[0] == "Where did I buy the coffee and how long was my commute"
    assert "Where did I buy the coffee" in bodies
    assert "how long was my commute" in bodies


def test_query_fission_does_not_split_short_noun_conjunction():
    assert decompose_query("What color were the walls and ceiling?") == (
        "What color were the walls and ceiling",
    )


def test_declarative_rewrite_is_bounded_and_first_person():
    assert declarative_variants("Where did I redeem a coupon?") == (
        "Where did I redeem a coupon", "I redeem a coupon at a place",
    )


def test_episodic_nucleus_excludes_assistant_topic_expansion():
    text = "2023-01-01 user: I use Spotify. assistant: Here are ten music services."
    assert episodic_nucleus(text) == "I use Spotify."


def test_field_discovers_each_clause_without_gold():
    texts = [
        "I commute for forty minutes every day.",
        "I redeemed the coffee coupon at North Market.",
        "The weather was pleasant.",
    ]
    field = QueryFissionField(_Encoder()).solve(
        "Where did I redeem the coffee coupon and how long is my commute?",
        texts=texts,
        lexical_index=BM25(texts),
        source_vectors={},
        base_candidates=[2, 1, 0],
    )
    assert {0, 1}.issubset(field.coverage_sources)
    assert field.discovery_sources[0] == 2


def test_periapsis_promotion_requires_both_confidence_and_lexical_contact():
    field = QueryFissionResult(
        periapsis_source=1, confidence_ratio=1.20, confidence_margin=0.10,
        episodic_nucleus_active=True,
    )
    assert should_promote_periapsis(field, 0, [1.0, 0.7])
    assert not should_promote_periapsis(field, 0, [1.0, 0.4])


def test_plain_documents_cannot_change_strict_rank_one():
    field = QueryFissionResult(
        periapsis_source=1, confidence_ratio=2.0, confidence_margin=1.0,
        episodic_nucleus_active=False,
    )
    assert not should_promote_periapsis(field, 0, [1.0, 1.0])
