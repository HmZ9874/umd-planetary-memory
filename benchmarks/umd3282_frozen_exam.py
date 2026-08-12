"""Gold-blind frozen UMD 3.28.2 retrieval exam.

The retriever is called before an evaluator reads evidence labels. This file is
an evaluation harness only: it does not tune or mutate retrieval parameters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from benchmarks.ai_memory_extended import (
    OrbitIndex,
    _flatten_membench_messages,
    run_memora,
)
from benchmarks.public_benchmarks import _locomo_sessions, normalize_locomo_evidence_ids
from benchmarks.umd39_benchmarks import (
    MODEL_CACHE,
    RESULTS,
    CapsuleRecallAccumulator,
    FastEmbedEncoder,
)


DATA = Path(__file__).resolve().parent / "data"
VENDOR = Path(__file__).resolve().parent / "vendor"
KS = (1, 5, 10)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _index(
    texts: Sequence[str], groups: Sequence[int], source_ids: Sequence[str],
    encoder: FastEmbedEncoder, dates: Sequence[str] | None = None,
    *, physics_v330: bool = False, physics_v331: bool = False,
    physics_v332: bool = False, physics_v333: bool = False,
    physics_v334: bool = False,
    physics_v335: bool = False,
    physics_v336: bool = False,
) -> OrbitIndex:
    return OrbitIndex(
        texts, groups, source_ids, encoder, dates,
        neural_candidate_pool=64,
        physics_v316=True,
        physics_v317=True,
        physics_v325=True,
        physics_v326=True,
        physics_v327=True,
        physics_v330=physics_v330 or physics_v331 or physics_v332 or physics_v333 or physics_v334 or physics_v335 or physics_v336,
        physics_v331=physics_v331 or physics_v332 or physics_v333 or physics_v334 or physics_v335 or physics_v336,
        physics_v332=physics_v332 or physics_v333 or physics_v334 or physics_v335 or physics_v336,
        physics_v333=physics_v333 or physics_v334 or physics_v335 or physics_v336,
        physics_v334=physics_v334 or physics_v335 or physics_v336,
        physics_v335=physics_v335 or physics_v336,
        physics_v336=physics_v336,
        first_orbit_v316=False,
    )


def _aggregate(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    queries = sum(int(result["evaluated_queries"]) for result in results)
    gold = sum(int(result["gold_evidence_items"]) for result in results)
    return {
        "evaluated_queries": queries,
        "gold_evidence_items": gold,
        "retrieval_unit": "evidence_capsule",
        "mrr": sum(result["mrr"] * result["evaluated_queries"] for result in results) / max(1, queries),
        "any_evidence_recall": {
            str(k): sum(result["any_evidence_recall"][str(k)] * result["evaluated_queries"] for result in results) / max(1, queries)
            for k in KS
        },
        "full_evidence_recall": {
            str(k): sum(result["full_evidence_recall"][str(k)] * result["evaluated_queries"] for result in results) / max(1, queries)
            for k in KS
        },
        "micro_evidence_recall": {
            str(k): sum(result["micro_evidence_recall"][str(k)] * result["gold_evidence_items"] for result in results) / max(1, gold)
            for k in KS
        },
    }


def _one_query(capsules: Sequence[Sequence[str]], gold: Iterable[str]) -> dict[str, Any]:
    metric = CapsuleRecallAccumulator(KS)
    metric.add(capsules, gold)
    return metric.result()


def _checkpoint(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _write_checkpoint(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def run_locomo(encoder: FastEmbedEncoder) -> dict[str, Any]:
    started = time.perf_counter()
    path = DATA / "locomo10.json"
    dataset = json.loads(path.read_text(encoding="utf-8"))
    checkpoint_path = RESULTS / "umd3282_exam_locomo_checkpoint.json"
    checkpoint = _checkpoint(checkpoint_path)
    for conversation_number, sample in enumerate(dataset, 1):
        key = str(conversation_number - 1)
        if key in checkpoint:
            print(f"LoCoMo exam checkpoint: {conversation_number}/{len(dataset)}", flush=True)
            continue
        _, sessions, dates = _locomo_sessions(sample)
        turns = [turn for session in sessions for turn in session]
        texts = [
            f'{turn.get("speaker", "")}: {turn.get("text", "")} {turn.get("blip_caption", "")}'
            for turn in turns
        ]
        ids = [str(turn["dia_id"]) for turn in turns]
        groups = [session for session, values in enumerate(sessions) for _ in values]
        index = _index(texts, groups, ids, encoder, dates)
        final_results: list[dict[str, Any]] = []
        strict_results: list[dict[str, Any]] = []
        categories: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
            lambda: {"final": [], "strict": []}
        )
        final_chars: list[int] = []
        strict_chars: list[int] = []
        id_to_text = dict(zip(ids, texts))
        no_evidence = 0
        for row in sample["qa"]:
            question = str(row["question"])
            # Frozen ranking happens before the evaluator reads evidence.
            channels = index.retrieve_compact_channels(question)
            final_capsules = channels["answer"]
            strict_capsules = channels["atomic_answer"]
            gold = normalize_locomo_evidence_ids(row.get("evidence"))
            if not gold:
                no_evidence += 1
                continue
            final_result = _one_query(final_capsules, gold)
            strict_result = _one_query(strict_capsules, gold)
            final_results.append(final_result)
            strict_results.append(strict_result)
            category = str(row.get("category", "unknown"))
            categories[category]["final"].append(final_result)
            categories[category]["strict"].append(strict_result)
            final_sources = {source for capsule in final_capsules[:10] for source in capsule}
            strict_sources = {source for capsule in strict_capsules[:10] for source in capsule}
            final_chars.append(sum(len(id_to_text.get(source, "")) for source in final_sources))
            strict_chars.append(sum(len(id_to_text.get(source, "")) for source in strict_sources))
        checkpoint[key] = {
            "qa_total": len(sample["qa"]),
            "qa_without_evidence": no_evidence,
            "final": _aggregate(final_results),
            "strict": _aggregate(strict_results),
            "by_category": {
                category: {
                    "final": _aggregate(values["final"]),
                    "strict": _aggregate(values["strict"]),
                }
                for category, values in categories.items()
            },
            "mean_final_chars_at_10": statistics.mean(final_chars) if final_chars else 0.0,
            "mean_strict_chars_at_10": statistics.mean(strict_chars) if strict_chars else 0.0,
        }
        _write_checkpoint(checkpoint_path, checkpoint)
        print(f"LoCoMo exam progress: {conversation_number}/{len(dataset)}", flush=True)
    records = [checkpoint[str(number)] for number in range(len(dataset))]
    category_names = sorted({name for record in records for name in record["by_category"]})
    return {
        "benchmark": "LoCoMo",
        "scope": "full 10 conversations",
        "questions": sum(record["qa_total"] for record in records),
        "questions_without_evidence": sum(record["qa_without_evidence"] for record in records),
        "final_capsule_retrieval": _aggregate([record["final"] for record in records]),
        "strict_source_atomic_retrieval": _aggregate([record["strict"] for record in records]),
        "by_category": {
            category: {
                "final": _aggregate([record["by_category"][category]["final"] for record in records if category in record["by_category"]]),
                "strict": _aggregate([record["by_category"][category]["strict"] for record in records if category in record["by_category"]]),
            }
            for category in category_names
        },
        "mean_retrieved_characters_at_10": {
            "final": statistics.mean(record["mean_final_chars_at_10"] for record in records),
            "strict": statistics.mean(record["mean_strict_chars_at_10"] for record in records),
        },
        "data_sha256": _sha256(path),
        "gold_used_for_ranking": False,
        "official_end_to_end_qa_run": False,
        "runtime_seconds": time.perf_counter() - started,
    }


def _long_session_text(item: dict[str, Any], number: int) -> str:
    session = item["haystack_sessions"][number]
    date = item["haystack_dates"][number]
    return f"{date} " + " ".join(
        f'{turn.get("role", "")}: {turn.get("content", "")}' for turn in session
    )


def run_longmemeval(
    encoder: FastEmbedEncoder, *, physics_v334: bool = False,
    physics_v335: bool = False,
    physics_v336: bool = False,
    question_start: int = 0, question_limit: int | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    path = DATA / "longmemeval_s_cleaned.json"
    dataset = json.loads(path.read_text(encoding="utf-8"))
    checkpoint_path = RESULTS / (
        "umd336_longmemeval_checkpoint.json"
        if physics_v336 else "umd335_longmemeval_checkpoint.json"
        if physics_v335 else
        "umd334_longmemeval_checkpoint.json"
        if physics_v334 else "umd3282_exam_longmemeval_checkpoint.json"
    )
    checkpoint = _checkpoint(checkpoint_path)
    start = min(max(0, int(question_start)), len(dataset))
    stop = len(dataset) if question_limit is None else min(len(dataset), start + max(0, int(question_limit)))
    selected_numbers = list(range(start, stop))
    for number in selected_numbers:
        item = dataset[number]
        key = str(number)
        if key in checkpoint:
            if (number + 1) % 10 == 0:
                print(f"LongMemEval exam checkpoint: {number + 1}/{len(dataset)}", flush=True)
            continue
        ids = [str(value) for value in item["haystack_session_ids"]]
        texts = [_long_session_text(item, index) for index in range(len(ids))]
        index = _index(
            texts, list(range(len(ids))), ids, encoder, item["haystack_dates"],
            physics_v334=physics_v334 or physics_v335 or physics_v336,
            physics_v335=physics_v335 or physics_v336,
            physics_v336=physics_v336,
        )
        question = str(item["question"])
        # Frozen ranking happens before answer_session_ids is read.
        channels = index.retrieve_compact_channels(question)
        final_capsules = channels["answer"]
        strict_capsules = channels["atomic_answer"]
        gold = {str(value) for value in item.get("answer_session_ids") or []}
        is_abstention = str(item.get("question_id", "")).endswith("_abs")
        checkpoint[key] = {
            "is_abstention": is_abstention,
            "has_evidence": bool(gold),
            "question_type": str(item.get("question_type", "unknown")),
            "final": _one_query(final_capsules, gold),
            "strict": _one_query(strict_capsules, gold),
            "final_chars_at_10": sum(len(texts[ids.index(source)]) for source in {source for capsule in final_capsules[:10] for source in capsule} if source in ids),
            "strict_chars_at_10": sum(len(texts[ids.index(source)]) for source in {source for capsule in strict_capsules[:10] for source in capsule} if source in ids),
        }
        if (number + 1) % 10 == 0 or number == selected_numbers[-1]:
            _write_checkpoint(checkpoint_path, checkpoint)
            print(f"LongMemEval exam progress: {number + 1}/{len(dataset)}", flush=True)
    records = [checkpoint[str(number)] for number in selected_numbers]
    answerable = [record for record in records if not record["is_abstention"] and record["has_evidence"]]
    all_evidence = [record for record in records if record["has_evidence"]]
    question_types = sorted({record["question_type"] for record in answerable})
    return {
        "benchmark": "LongMemEval_S_cleaned",
        "umd_version": (
            "3.36-post-evaluation-regression" if physics_v336
            else "3.35-post-evaluation-regression" if physics_v335
            else "3.34-development-replay" if physics_v334
            else "3.28.2-frozen"
        ),
        "question_slice": {"start": start, "stop": stop},
        "questions": len(records),
        "answerable_questions": len(answerable),
        "abstention_questions": sum(record["is_abstention"] for record in records),
        "questions_without_evidence": sum(not record["has_evidence"] for record in records),
        "final_capsule_retrieval": _aggregate([record["final"] for record in answerable]),
        "strict_source_atomic_retrieval": _aggregate([record["strict"] for record in answerable]),
        "all_500_diagnostic_including_abstention": {
            "final": _aggregate([record["final"] for record in all_evidence]),
            "strict": _aggregate([record["strict"] for record in all_evidence]),
        },
        "by_question_type": {
            kind: {
                "final": _aggregate([record["final"] for record in answerable if record["question_type"] == kind]),
                "strict": _aggregate([record["strict"] for record in answerable if record["question_type"] == kind]),
            }
            for kind in question_types
        },
        "mean_retrieved_characters_at_10": {
            "final": statistics.mean(record["final_chars_at_10"] for record in answerable),
            "strict": statistics.mean(record["strict_chars_at_10"] for record in answerable),
        },
        "data_sha256": _sha256(path),
        "gold_used_for_ranking": False,
        "official_end_to_end_qa_run": False,
        "runtime_seconds": time.perf_counter() - started,
    }


def run_membench(encoder: FastEmbedEncoder, per_group: int = 10) -> dict[str, Any]:
    started = time.perf_counter()
    root = VENDOR / "MemBenchOfficial" / "MemData"
    final_metric = CapsuleRecallAccumulator(KS)
    strict_metric = CapsuleRecallAccumulator(KS)
    trajectories = groups_seen = 0
    for path in sorted(root.rglob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        for rows in data.values():
            if not isinstance(rows, list):
                continue
            groups_seen += 1
            for trajectory in rows[:per_group]:
                texts, groups, ids = _flatten_membench_messages(trajectory.get("message_list", []))
                if not texts:
                    continue
                qa = trajectory.get("QA", {})
                index = _index(texts, groups, ids, encoder)
                channels = index.retrieve_compact_channels(str(qa.get("question", "")))
                final_capsules = channels["answer"]
                strict_capsules = channels["atomic_answer"]
                gold = {
                    str(value[0] if isinstance(value, list) else value)
                    for value in qa.get("target_step_id", [])
                }
                if not gold:
                    continue
                final_metric.add(final_capsules, gold)
                strict_metric.add(strict_capsules, gold)
                trajectories += 1
    return {
        "benchmark": "MemBench",
        "status": "stratified_retrieval_proxy",
        "sampling": f"first {per_group} trajectories from each of {groups_seen} official groups",
        "trajectories": trajectories,
        "official_trajectory_count_local": 26637,
        "final_capsule_retrieval": final_metric.result(),
        "strict_source_atomic_retrieval": strict_metric.result(),
        "gold_used_for_ranking": False,
        "official_end_to_end_accuracy_run": False,
        "runtime_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=("locomo", "longmemeval", "membench", "memora"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--membench-per-group", type=int, default=10)
    parser.add_argument("--physics-v334", action="store_true")
    parser.add_argument("--physics-v335", action="store_true")
    parser.add_argument("--physics-v336", action="store_true")
    parser.add_argument("--question-start", type=int, default=0)
    parser.add_argument("--question-limit", type=int)
    parser.add_argument(
        "--frozen-holdout", action="store_true",
        help="Mark a predeclared post-development slice as parameter-frozen.",
    )
    args = parser.parse_args()
    if args.frozen_holdout and (
        not (args.physics_v334 or args.physics_v335 or args.physics_v336)
        or args.question_start < 100
    ):
        parser.error("--frozen-holdout requires UMD 3.34+ and --question-start >= 100")
    encoder = FastEmbedEncoder(
        cache_dir=MODEL_CACHE, batch_size=128, cache_size=32768, threads=16,
    )
    if args.benchmark == "locomo":
        result = run_locomo(encoder)
    elif args.benchmark == "longmemeval":
        result = run_longmemeval(
            encoder, physics_v334=args.physics_v334,
            physics_v335=args.physics_v335,
            physics_v336=args.physics_v336,
            question_start=args.question_start, question_limit=args.question_limit,
        )
    elif args.benchmark == "membench":
        result = run_membench(encoder, args.membench_per_group)
    else:
        result = run_memora(
            encoder, 10, physics_v327=True, persona_start_per_period=0,
            checkpoint_name="memora_umd3282_exam_v2_gold_blind_checkpoint.json",
        )
    payload = {
        args.benchmark: result,
        "metadata": {
            "adapter": (
                "UMD 3.36 post-evaluation regression: bounded ghost constellations"
                if args.physics_v336 else
                "UMD 3.35 post-evaluation regression: background ghost matter"
                if args.physics_v335 else
                "UMD 3.34 post-dataset development replay: query fission and semantic periapsis"
                if args.physics_v334 else "UMD 3.28.2 frozen exam"
            ),
            "gold_read_after_retrieval": True,
            "gold_used_for_ranking": False,
            "parameters_frozen": (
                not (args.physics_v334 or args.physics_v335 or args.physics_v336)
                or bool(args.frozen_holdout)
            ),
            "post_development_holdout": bool(args.frozen_holdout),
            "paid_api_calls": 0,
            "official_answer_or_judge_model": None,
            "encoder": encoder.metadata(),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
