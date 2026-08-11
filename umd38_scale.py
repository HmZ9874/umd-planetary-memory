"""Restartable-sized synthetic scale probe for the UMD 3.8 orbital index.

This is an algorithmic index probe, not a substitute for a deployed million-
memory endurance report. Use ``--count 1000000`` on suitably provisioned
hardware when running the production qualification.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import numpy as np

from umd35_core import FactFrame, MemoryNode, WriteFeatures, utcnow
from umd38_cosmic import IncrementalOrbitalIndex


def node(memory_id: str, vector: np.ndarray) -> MemoryNode:
    now = utcnow()
    return MemoryNode(
        id=memory_id,
        text=f"orbital telemetry record {memory_id} component C-{memory_id}",
        vector=vector,
        created_at=now,
        updated_at=now,
        last_decay_at=now,
        source="synthetic",
        source_trust=0.5,
        scope="scale",
        kind="observation",
        state="stable",
        write_probability=0.9,
        write_features=WriteFeatures(1, 1, 0, 0, 0.5, 0, 0, 0, 0),
        fact_key=memory_id,
        entities={memory_id},
        fact=FactFrame(memory_id, "status", "active"),
    )


def run(count: int = 10_000, dimensions: int = 64, queries: int = 200) -> dict[str, object]:
    count = max(100, count)
    rng = np.random.default_rng(380091)
    index = IncrementalOrbitalIndex(dimensions, tables=5, bits=12, degree=12, search_width=64)
    vectors: dict[int, np.ndarray] = {}
    latencies = []
    started = time.perf_counter()
    sample_every = max(1, count // max(queries, 1))
    for item in range(count):
        vector = rng.normal(size=dimensions).astype(np.float32)
        vector /= max(1e-9, float(np.linalg.norm(vector)))
        before = time.perf_counter()
        index.upsert(node(f"m{item:08d}", vector))
        latencies.append((time.perf_counter() - before) * 1000.0)
        if item % sample_every == 0:
            vectors[item] = vector
    index.revision = 1
    insert_seconds = time.perf_counter() - started
    found, query_latencies = 0, []
    for item, vector in list(vectors.items())[:queries]:
        before = time.perf_counter()
        candidates = index.ann_candidates(vector)
        query_latencies.append((time.perf_counter() - before) * 1000.0)
        found += f"m{item:08d}" in candidates
    edge_count = sum(len(edges) for edges in index.neighbors.values())
    estimated_bytes = (
        sum(vector.nbytes for vector in index.vectors.values())
        + edge_count * 40
        + sum(len(text.encode("utf-8")) for terms in index.tokens.values() for text in terms)
    )
    return {
        "version": "UMD 3.8 orbital scale probe",
        "records": count,
        "dimensions": dimensions,
        "insert_seconds": insert_seconds,
        "inserts_per_second": count / max(insert_seconds, 1e-9),
        "insert_latency_ms": {
            "p50": statistics.median(latencies),
            "p95": sorted(latencies)[int(0.95 * (len(latencies) - 1))],
            "first_10pct_mean": statistics.mean(latencies[: max(1, count // 10)]),
            "last_10pct_mean": statistics.mean(latencies[-max(1, count // 10):]),
        },
        "query_latency_ms": {
            "p50": statistics.median(query_latencies),
            "p95": sorted(query_latencies)[int(0.95 * (len(query_latencies) - 1))],
        },
        "self_recall": found / max(1, len(query_latencies)),
        "index_full_rebuilds": index.full_rebuilds,
        "incremental_upserts": index.incremental_upserts,
        "directed_neighbor_edges": edge_count,
        "estimated_index_bytes": estimated_bytes,
        "estimated_bytes_per_record": estimated_bytes / count,
        "qualification": "synthetic algorithm probe; not production endurance evidence",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=10_000)
    parser.add_argument("--dimensions", type=int, default=64)
    parser.add_argument("--queries", type=int, default=200)
    args = parser.parse_args()
    print(json.dumps(run(args.count, args.dimensions, args.queries), indent=2))

