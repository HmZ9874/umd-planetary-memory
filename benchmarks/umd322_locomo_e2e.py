"""Export gold-blind UMD 3.22 contexts for the 1,540-question LoCoMo QA scope.

This stage does not call an answer model or judge. Reference answers and
evidence IDs are attached only after retrieval so the JSONL can be consumed by
an external, protocol-matched answer/judge harness without rerunning retrieval.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import joblib

from benchmarks.ai_memory_extended import OrbitIndex
from benchmarks.public_benchmarks import _locomo_sessions, normalize_locomo_evidence_ids
from benchmarks.umd318_cross import OnnxCrossEncoder
from benchmarks.umd39_benchmarks import DATA, FastEmbedEncoder, MODEL_CACHE, RESULTS


PUBLIC_CATEGORIES = {"1", "2", "3", "4"}


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    low = int(position)
    high = min(len(ordered) - 1, low + 1)
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def evidence_summary(rows: list[dict]) -> dict:
    answerable = [row for row in rows if row["evidence"]]
    result = {
        "questions": len(rows),
        "evidence_questions": len(answerable),
        "no_evidence_questions": len(rows) - len(answerable),
    }
    for depth in (1, 5, 10):
        any_hits = full_hits = found = total = 0
        for row in answerable:
            retrieved = {
                source for capsule in row["retrieved_capsules"][:depth]
                for source in capsule
            }
            gold = set(row["evidence"])
            overlap = gold & retrieved
            any_hits += bool(overlap)
            full_hits += gold <= retrieved
            found += len(overlap)
            total += len(gold)
        result[f"any_evidence_recall@{depth}"] = any_hits / max(1, len(answerable))
        result[f"full_evidence_recall@{depth}"] = full_hits / max(1, len(answerable))
        result[f"micro_evidence_recall@{depth}"] = found / max(1, total)
    return result


def load_completed(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    completed = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            completed[row["question_id"]] = row
    return completed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA / "locomo10.json")
    parser.add_argument("--contexts", type=Path, default=RESULTS / "umd322_locomo1540_contexts.jsonl")
    parser.add_argument("--metrics", type=Path, default=RESULTS / "umd322_locomo1540_retrieval.json")
    parser.add_argument("--ranker", type=Path, default=RESULTS / "umd322_blended.joblib")
    parser.add_argument("--cross-limit", type=int, default=32)
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()

    dataset = json.loads(args.data.read_text(encoding="utf-8"))
    expected = sum(
        str(row.get("category")) in PUBLIC_CATEGORIES
        for sample in dataset for row in sample["qa"]
    )
    if expected != 1540:
        raise RuntimeError(f"public LoCoMo category scope changed: expected 1540, found {expected}")
    if args.fresh and args.contexts.exists():
        args.contexts.unlink()
    completed = load_completed(args.contexts)
    encoder = FastEmbedEncoder(
        model_name="BAAI/bge-base-en-v1.5", dimensions=768,
        cache_dir=MODEL_CACHE, batch_size=64, cache_size=16384, threads=4,
    )
    ranker = joblib.load(args.ranker)
    cross = OnnxCrossEncoder(MODEL_CACHE)
    args.contexts.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    ingest_seconds: list[float] = []
    new_rows = 0

    with args.contexts.open("a", encoding="utf-8") as output:
        for conversation_number, sample in enumerate(dataset):
            public_rows = [
                (qa_number, row) for qa_number, row in enumerate(sample["qa"])
                if str(row.get("category")) in PUBLIC_CATEGORIES
            ]
            pending = [
                (qa_number, row) for qa_number, row in public_rows
                if f"{conversation_number}:{qa_number}" not in completed
            ]
            if not pending:
                print(f"LoCoMo context progress: {conversation_number + 1}/10 (cached)", flush=True)
                continue
            _, sessions, dates = _locomo_sessions(sample)
            turns = [turn for session in sessions for turn in session]
            texts = [
                f'{turn.get("speaker", "")}: {turn.get("text", "")} {turn.get("blip_caption", "")}'.strip()
                for turn in turns
            ]
            ids = [str(turn["dia_id"]) for turn in turns]
            groups = [session for session, values in enumerate(sessions) for _ in values]
            source_text = dict(zip(ids, texts))
            source_date = {
                ids[index]: dates[groups[index]] if groups[index] < len(dates) else ""
                for index in range(len(ids))
            }
            ingest_started = time.perf_counter()
            index = OrbitIndex(
                texts, groups, ids, encoder, dates,
                physics_v322=True, roche_budget_v318=10,
                matter_neighbor_budget=0, cross_encoder=cross,
                cross_limit=args.cross_limit, roche_ranker=ranker,
            )
            ingest_seconds.append(time.perf_counter() - ingest_started)
            for qa_number, row in pending:
                question = str(row["question"])
                query_started = time.perf_counter()
                capsules = index.retrieve(question)
                query_seconds = time.perf_counter() - query_started
                ordered_sources = list(dict.fromkeys(
                    source for capsule in capsules[:10] for source in capsule
                ))
                context = "\n\n".join(
                    f"[source={source} date={source_date.get(source, '')}]\n{source_text.get(source, '')}"
                    for source in ordered_sources
                )
                unsafe = index.last_roche_diagnostics.get("unsafe_reasons", {})
                unsafe_by_id = {
                    ids[int(source)]: reasons for source, reasons in unsafe.items()
                }
                payload = {
                    "question_id": f"{conversation_number}:{qa_number}",
                    "sample_id": sample.get("sample_id", conversation_number),
                    "conversation_index": conversation_number,
                    "qa_index": qa_number,
                    "category": str(row.get("category")),
                    "question": question,
                    "reference_answer": row.get("answer"),
                    "evidence": sorted(normalize_locomo_evidence_ids(row.get("evidence"))),
                    "retrieved_capsules": [list(capsule) for capsule in capsules[:10]],
                    "context_source_ids": ordered_sources,
                    "context": context,
                    "context_characters": len(context),
                    "estimated_context_tokens_chars_div_4": math.ceil(len(context) / 4),
                    "retrieval_seconds": query_seconds,
                    "unsafe_reasons": unsafe_by_id,
                    "gold_used_for_ranking": False,
                }
                output.write(json.dumps(payload, ensure_ascii=False) + "\n")
                output.flush()
                completed[payload["question_id"]] = payload
                new_rows += 1
            print(
                f"LoCoMo context progress: {conversation_number + 1}/10 "
                f"({len(pending)} new)", flush=True,
            )

    rows = [completed[key] for key in sorted(
        completed, key=lambda value: tuple(int(part) for part in value.split(":"))
    )]
    if len(rows) != expected:
        raise RuntimeError(f"incomplete context export: {len(rows)}/{expected}")
    latencies = [float(row["retrieval_seconds"]) for row in rows]
    characters = [int(row["context_characters"]) for row in rows]
    estimated_tokens = [int(row["estimated_context_tokens_chars_div_4"]) for row in rows]
    metrics = {
        "benchmark": "LoCoMo",
        "scope": "categories_1_to_4",
        "questions": len(rows),
        "gold_used_for_ranking": False,
        "answer_model": None,
        "judge_model": None,
        "retrieval": evidence_summary(rows),
        "context": {
            "mean_characters": statistics.fmean(characters),
            "median_characters": statistics.median(characters),
            "p95_characters": percentile([float(value) for value in characters], 0.95),
            "mean_estimated_tokens_chars_div_4": statistics.fmean(estimated_tokens),
            "estimation_warning": "Character/4 is a planning estimate, not an answer-model tokenizer count.",
        },
        "latency": {
            "query_p50_seconds": statistics.median(latencies),
            "query_p95_seconds": percentile(latencies, 0.95),
            "query_mean_seconds": statistics.fmean(latencies),
            "new_run_ingest_seconds": sum(ingest_seconds),
            "new_run_wall_seconds": time.perf_counter() - started,
            "new_rows": new_rows,
        },
        "encoder": encoder.metadata(),
    }
    args.metrics.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
