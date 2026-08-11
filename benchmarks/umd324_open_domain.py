"""Development audit for UMD open-domain evidence retrieval.

Gold labels are attached only after retrieval.  This script is diagnostic and
must not be reported as a fresh hidden-test benchmark once its output is used
to change retrieval rules.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib

from benchmarks.ai_memory_extended import OrbitIndex
from benchmarks.public_benchmarks import _locomo_sessions, normalize_locomo_evidence_ids
from benchmarks.umd318_cross import OnnxCrossEncoder
from benchmarks.umd39_benchmarks import DATA, FastEmbedEncoder, MODEL_CACHE, RESULTS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA / "locomo10.json")
    parser.add_argument("--output", type=Path, default=RESULTS / "umd324_open_domain_adaptive.json")
    parser.add_argument("--budget", type=int, default=96)
    parser.add_argument("--cross-limit", type=int, default=64)
    parser.add_argument("--ranker", type=Path, default=RESULTS / "umd322_blended.joblib")
    args = parser.parse_args()

    dataset = json.loads(args.data.read_text(encoding="utf-8"))
    encoder = FastEmbedEncoder(
        model_name="BAAI/bge-base-en-v1.5", dimensions=768,
        cache_dir=MODEL_CACHE, batch_size=64, cache_size=16384, threads=4,
    )
    cross = OnnxCrossEncoder(MODEL_CACHE)
    ranker = joblib.load(args.ranker)
    records: list[dict] = []
    started = time.perf_counter()

    for conversation_index, sample in enumerate(dataset):
        questions = [row for row in sample["qa"] if str(row.get("category")) == "3"]
        if not questions:
            continue
        _, sessions, dates = _locomo_sessions(sample)
        turns = [turn for session in sessions for turn in session]
        texts = [
            f'{turn.get("speaker", "")}: {turn.get("text", "")} {turn.get("blip_caption", "")}'.strip()
            for turn in turns
        ]
        ids = [str(turn["dia_id"]) for turn in turns]
        groups = [session for session, values in enumerate(sessions) for _ in values]
        id_to_text = dict(zip(ids, texts))
        index = OrbitIndex(
            texts, groups, ids, encoder, dates, physics_v324=True,
            roche_budget_v318=args.budget, matter_neighbor_budget=0,
            cross_encoder=cross, cross_limit=args.cross_limit, roche_ranker=ranker,
        )
        for qa_index, row in enumerate(sample["qa"]):
            if str(row.get("category")) != "3":
                continue
            question = str(row["question"])
            capsules = index.retrieve(question)
            diagnostics = index.last_roche_diagnostics
            l1 = set(capsules[0]) if capsules else set()
            gold = normalize_locomo_evidence_ids(row.get("evidence"))
            ranked_ids = [ids[int(source)] for source in diagnostics.get("ranked_sources", [])]
            ranked_position = {source: pos + 1 for pos, source in enumerate(ranked_ids)}
            gold_ranks = [ranked_position[source] for source in gold if source in ranked_position]
            unsafe_by_id = {
                ids[int(source)]: reasons
                for source, reasons in diagnostics.get("unsafe_reasons", {}).items()
            }
            records.append({
                "question_id": f"{conversation_index}:{qa_index}",
                "conversation_index": conversation_index,
                "question": question,
                "reference_answer": row.get("answer"),
                "evidence": sorted(gold),
                "evidence_text": {source: id_to_text.get(source, "") for source in sorted(gold)},
                "any_l1": bool(gold & l1) if gold else None,
                "full_l1": gold <= l1 if gold else None,
                "l1_sources": list(capsules[0]) if capsules else [],
                "l1_characters": sum(len(id_to_text.get(source, "")) for source in l1),
                "best_gold_candidate_rank": min(gold_ranks) if gold_ranks else None,
                "gold_in_candidate_field": bool(gold_ranks),
                "biosphere_triggered": bool(diagnostics.get("biosphere_triggered")),
                "gold_unsafe_reasons": {
                    source: unsafe_by_id[source] for source in gold if source in unsafe_by_id
                },
                "gold_used_for_ranking": False,
            })
        print(f"open-domain audit progress: {conversation_index + 1}/10", flush=True)

    answerable = [record for record in records if record["evidence"]]
    misses = [record for record in answerable if not record["any_l1"]]
    payload = {
        "version": "UMD 3.24",
        "scope": "LoCoMo category 3 development audit",
        "questions": len(records),
        "evidence_questions": len(answerable),
        "any_evidence_recall@1": sum(record["any_l1"] for record in answerable) / len(answerable),
        "full_evidence_recall@1": sum(record["full_l1"] for record in answerable) / len(answerable),
        "misses": len(misses),
        "misses_with_gold_in_candidate_field": sum(record["gold_in_candidate_field"] for record in misses),
        "misses_beyond_candidate_field": sum(not record["gold_in_candidate_field"] for record in misses),
        "biosphere_triggered_questions": sum(record["biosphere_triggered"] for record in answerable),
        "runtime_seconds": time.perf_counter() - started,
        "gold_used_for_ranking": False,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key != "records"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
