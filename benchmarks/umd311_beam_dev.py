"""UMD 3.11 BEAM development-split experiment.

Only the first six conversations are used for parameter selection.  Gold IDs
are consumed after all label-blind retrieval signals have been computed.  The
remaining fourteen conversations are reserved for the final frozen check.
"""

from __future__ import annotations

import ast
import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from fastembed import LateInteractionTextEmbedding

from benchmarks.public_benchmarks import (
    BM25,
    RecallAccumulator,
    char_tokenize,
    ranking,
    reciprocal_ranks,
    unit_scores,
)
from benchmarks.umd39_benchmarks import (
    BEAM_TYPES,
    MODEL_CACHE,
    _beam_batches,
    _dot_scores,
    _entity_value_resonance,
    _flatten_source_ids,
    _force,
    _graph_flux,
    _orbit_expand,
    _temporal_phase,
)
from umd35_neural import FastEmbedEncoder


DATA = Path(__file__).resolve().parent / "data" / "beam_hf" / "data" / "100K-00000-of-00001.parquet"
OUT = Path(__file__).resolve().parent / "results" / "umd311_beam_dev.json"


def _late_score(query: np.ndarray, passage: np.ndarray) -> float:
    return float(np.max(query @ passage.T, axis=1).sum())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=6)
    parser.add_argument("--fixed", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    rows = pq.read_table(DATA).to_pylist()[args.start:args.stop]
    dense = FastEmbedEncoder(cache_dir=MODEL_CACHE, batch_size=64, cache_size=8192, threads=4)
    late = LateInteractionTextEmbedding(
        model_name="answerdotai/answerai-colbert-small-v1",
        cache_dir=str(MODEL_CACHE),
        threads=4,
    )
    parameter_grid = (
        [(0.60, 0.30, 0.10)] if args.fixed else list(itertools.product(
            (0.45, 0.60, 0.75),  # late interaction
            (0.10, 0.20, 0.30),  # UMD 3.9 force
            (0.00, 0.10, 0.20),  # adjacent-turn resonance
        ))
    )
    metrics = {str(params): RecallAccumulator((1, 5, 10, 20)) for params in parameter_grid}
    baseline = RecallAccumulator((1, 5, 10, 20))

    for row_number, row in enumerate(rows, 1):
        batches = _beam_batches(row["chat"])
        turns = [turn for batch in batches for turn in batch if isinstance(turn, dict)]
        texts = [f'{turn.get("role", "")}: {turn.get("content", "")}' for turn in turns]
        ids = [str(turn.get("id", turn.get("index", index))) for index, turn in enumerate(turns)]
        groups: list[int] = []
        batch_indices: list[list[int]] = []
        cursor = 0
        for batch_number, batch in enumerate(batches):
            valid = [turn for turn in batch if isinstance(turn, dict)]
            indices = list(range(cursor, cursor + len(valid)))
            batch_indices.append(indices)
            groups.extend([batch_number] * len(valid))
            cursor += len(valid)
        dates = [
            next((str(turn.get("time_anchor")) for turn in batch if turn.get("time_anchor")), "")
            for batch in batches
        ]
        session_texts = [
            f"{dates[i]} " + " ".join(texts[j] for j in indices)
            for i, indices in enumerate(batch_indices)
        ]
        lexical_index = BM25(texts)
        char_index = BM25(texts, tokenizer=char_tokenize)
        planet_index = BM25(session_texts)
        dense_passages = dense.encode_many(texts)
        late_passages = list(late.passage_embed(texts, batch_size=16))

        questions = ast.literal_eval(row["probing_questions"])
        question_items = [
            (kind, question)
            for kind in BEAM_TYPES if kind != "abstention"
            for question in questions.get(kind, [])
            if _flatten_source_ids(question.get("source_chat_ids"))
        ]
        question_texts = [str(q.get("question") or q.get("question_text") or "") for _, q in question_items]
        dense_queries = dense.encode_many(question_texts)
        late_queries = list(late.query_embed(question_texts, batch_size=16))

        for question_number, (_, question_data) in enumerate(question_items):
            query = question_texts[question_number]
            gold = _flatten_source_ids(question_data.get("source_chat_ids"))
            lexical_raw = lexical_index.scores(query)
            lexical_order = ranking(lexical_raw)
            planet_raw = planet_index.scores(query)
            planet_order = ranking(planet_raw)
            lexical_rr = reciprocal_ranks(lexical_order)
            planet_rr = reciprocal_ranks(planet_order)
            seed = ranking([
                lexical_rr[i] + 0.72 * planet_rr[groups[i]] for i in range(len(texts))
            ])
            orbit = _orbit_expand(seed, groups)
            protected = orbit[:20]
            semantic = _dot_scores(dense_queries[question_number], dense_passages)
            lexical = unit_scores(lexical_raw)
            planet_unit = unit_scores(planet_raw)
            planetary = [planet_unit[groups[i]] for i in range(len(texts))]
            char = unit_scores(char_index.scores(query))
            entity = _entity_value_resonance(query, texts)
            temporal_session = _temporal_phase(query, dates)
            temporal = [temporal_session[groups[i]] for i in range(len(texts))]
            graph = _graph_flux(lexical, groups)
            force = _force(semantic, lexical, planetary, char, entity, temporal, graph)
            head = sorted(protected, key=lambda i: (-force[i], i))
            protected_set = set(protected)
            base_order = head + [i for i in orbit if i not in protected_set]
            baseline.add([ids[i] for i in base_order], gold)

            # ColBERT scores only raw text/query. Labels are not visible here.
            late_raw = [_late_score(late_queries[question_number], passage) for passage in late_passages]
            late_unit = unit_scores(late_raw)
            adjacent = []
            for index, value in enumerate(late_unit):
                neighbors = [value]
                if index and groups[index - 1] == groups[index]:
                    neighbors.append(late_unit[index - 1])
                if index + 1 < len(texts) and groups[index + 1] == groups[index]:
                    neighbors.append(late_unit[index + 1])
                adjacent.append(max(neighbors))
            base_rr = reciprocal_ranks(base_order)
            for params in parameter_grid:
                late_weight, force_weight, adjacent_weight = params
                rank_score = [
                    late_weight * late_unit[i]
                    + force_weight * force[i]
                    + adjacent_weight * adjacent[i]
                    + 0.10 * base_rr[i]
                    for i in range(len(texts))
                ]
                order = ranking(rank_score)
                metrics[str(params)].add([ids[i] for i in order], gold)
        print(f"UMD311 BEAM dev progress: {row_number}/{len(rows)}", flush=True)

    results = {key: value.result() for key, value in metrics.items()}
    ranked = sorted(
        results.items(),
        key=lambda item: (
            -item[1]["full_evidence_recall"]["10"],
            -item[1]["any_evidence_recall"]["10"],
            -item[1]["mrr"],
        ),
    )
    output = {
        "split": f"BEAM rows {args.start}:{args.stop} " + ("frozen evaluation" if args.fixed else "development"),
        "gold_used_for_signal_or_candidate_generation": False,
        "gold_used_for_parameter_selection": not args.fixed,
        "baseline": baseline.result(),
        "best": {"parameters": ranked[0][0], "metrics": ranked[0][1]},
        "top_five": [{"parameters": key, "metrics": value} for key, value in ranked[:5]],
        "runtime_seconds": time.perf_counter() - started,
    }
    output_path = (
        OUT.parent / "umd311_beam_heldout.json" if args.fixed
        else OUT
    )
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
