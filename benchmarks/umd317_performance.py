"""Reproducible microbenchmark for UMD 3.17 prefix BM25 conservation."""

from __future__ import annotations

import collections
import json
import math
import time

from benchmarks.public_benchmarks import BM25


def brute_prefix(index: BM25, query: str, visible: int) -> list[float]:
    scores = [0.0] * visible
    avg_length = sum(index.lengths[:visible]) / visible
    for term, qtf in collections.Counter(index.tokenizer(query)).items():
        posting = [(doc_id, tf) for doc_id, tf in index.postings.get(term, []) if doc_id < visible]
        if not posting:
            continue
        df = len(posting)
        idf = math.log(1.0 + (visible - df + 0.5) / (df + 0.5))
        for doc_id, tf in posting:
            norm = 1.0 - index.b + index.b * index.lengths[doc_id] / max(1e-9, avg_length)
            scores[doc_id] += qtf * idf * tf * (index.k1 + 1.0) / (tf + index.k1 * norm)
    return scores


def run(documents: int = 6000, queries: int = 240) -> dict[str, float | int | bool]:
    corpus = [
        f"session {i} project-{i % 97} preference-{i % 41} task-{i % 211} common memory"
        for i in range(documents)
    ]
    index = BM25(corpus)
    workload = [
        (f"project-{i % 97} task-{i % 211} common", 100 + (i * 23) % (documents - 100))
        for i in range(queries)
    ]
    started = time.perf_counter()
    brute = [brute_prefix(index, query, size) for query, size in workload]
    brute_seconds = time.perf_counter() - started
    started = time.perf_counter()
    optimized = [index.scores_prefix(query, size) for query, size in workload]
    optimized_seconds = time.perf_counter() - started
    equivalent = all(
        len(left) == len(right)
        and all(abs(a - b) < 1e-12 for a, b in zip(left, right))
        for left, right in zip(brute, optimized)
    )
    return {
        "documents": documents,
        "queries": queries,
        "equivalent": equivalent,
        "brute_seconds": brute_seconds,
        "optimized_seconds": optimized_seconds,
        "speedup": brute_seconds / max(1e-12, optimized_seconds),
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
