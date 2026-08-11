"""Insertion-order metamorphic test for learned UMD orbital rankers."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import joblib

from benchmarks.ai_memory_extended import OrbitIndex
from benchmarks.umd318_cross import OnnxCrossEncoder
from benchmarks.umd39_benchmarks import FastEmbedEncoder, MODEL_CACHE, RESULTS


def jaccard(left: set[str], right: set[str]) -> float:
    return len(left & right) / max(1, len(left | right))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=RESULTS / "umd321_metamorphic_stability.json")
    args = parser.parse_args()
    texts = [
        f"Speaker: Routine duplicated project diary {index} discussed travel, employment, storage and hobbies."
        for index in range(80)
    ]
    ids = [f"noise_{index}" for index in range(80)]
    dates = [f"2024-{index % 12 + 1:02d}-01" for index in range(80)]

    def add(source_id: str, text: str, date: str) -> str:
        ids.append(source_id)
        texts.append(f"Speaker: {text}")
        dates.append(date)
        return source_id

    home = add("gold_home", "Sana's verified current home is Tromso.", "2025-05-01")
    employer = add("gold_employer", "Ivo currently works for Helix Observatory.", "2025-04-01")
    link = add("gold_link", "Uma's mentor is Professor Ren.", "2024-03-01")
    lab = add("gold_lab", "Professor Ren directs the Cinder Laboratory in Delft.", "2024-04-01")
    hobbies = [
        add(f"gold_hobby_{index}", f"Theo confirmed {hobby} as a hobby.", f"2025-0{index + 1}-01")
        for index, hobby in enumerate(("rowing", "painting", "astronomy"))
    ]
    queries = [
        ("Where does Sana currently live?", {home}),
        ("Where does Ivo currently work?", {employer}),
        ("Which city contains the laboratory directed by Uma's mentor?", {link, lab}),
        ("List every confirmed hobby of Theo.", set(hobbies)),
    ]
    encoder = FastEmbedEncoder(
        model_name="BAAI/bge-base-en-v1.5", dimensions=768,
        cache_dir=MODEL_CACHE, batch_size=64, cache_size=16384, threads=4,
    )
    rankers = {
        "umd320": joblib.load(RESULTS / "umd320_lambdarank.joblib"),
        "umd321": joblib.load(RESULTS / "umd321_blended.joblib"),
        "umd322": joblib.load(RESULTS / "umd321_blended.joblib"),
        "umd323": joblib.load(RESULTS / "umd322_blended.joblib"),
        "umd324": joblib.load(RESULTS / "umd322_blended.joblib"),
    }
    rows: dict[str, list[dict]] = {name: [] for name in rankers}
    baselines: dict[tuple[str, int], set[str]] = {}
    started = time.perf_counter()
    for seed in range(8):
        order = list(range(len(texts)))
        random.Random(seed).shuffle(order)
        ordered_texts = [texts[index] for index in order]
        ordered_ids = [ids[index] for index in order]
        ordered_dates = [dates[index] for index in order]
        ordered_groups = list(range(len(order)))
        for system, ranker in rankers.items():
            is_new = system in {"umd321", "umd322", "umd323", "umd324"}
            is_repaired = system in {"umd322", "umd323", "umd324"}
            is_high_recall = system in {"umd323", "umd324"}
            is_adaptive = system == "umd324"
            index = OrbitIndex(
                ordered_texts, ordered_groups, ordered_ids, encoder, ordered_dates,
                physics_v320=True, physics_v321=is_new,
                physics_v322=is_repaired,
                physics_v323=is_high_recall,
                physics_v324=is_adaptive,
                roche_budget_v318=96 if is_high_recall else 10, matter_neighbor_budget=0,
                cross_encoder=OnnxCrossEncoder(MODEL_CACHE) if is_new else None,
                cross_limit=64 if is_high_recall else 32, roche_ranker=ranker,
            )
            for query_number, (query, gold) in enumerate(queries):
                capsules = index.retrieve(query)
                l1 = set(capsules[0])
                baseline_key = (system, query_number)
                if seed == 0:
                    baselines[baseline_key] = l1
                rows[system].append({
                    "seed": seed, "query_number": query_number,
                    "full_gold_l1": gold <= l1,
                    "any_gold_l1": bool(gold & l1),
                    "l1_jaccard_to_seed0": jaccard(l1, baselines[baseline_key]),
                    "l1_sources": list(capsules[0]),
                })
        print(f"metamorphic order progress: {seed + 1}/8", flush=True)
    summary = {}
    for system, values in rows.items():
        summary[system] = {
            "runs": len(values),
            "any_gold_l1_rate": sum(row["any_gold_l1"] for row in values) / len(values),
            "full_gold_l1_rate": sum(row["full_gold_l1"] for row in values) / len(values),
            "mean_l1_jaccard_to_seed0": sum(row["l1_jaccard_to_seed0"] for row in values) / len(values),
            "minimum_l1_jaccard_to_seed0": min(row["l1_jaccard_to_seed0"] for row in values),
        }
    payload = {
        "suite": "UMD learned-orbit insertion-order metamorphic stability",
        "official_benchmark": False,
        "gold_used_for_ranking": False,
        "runtime_seconds": time.perf_counter() - started,
        "summary": summary,
        "results": rows,
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
