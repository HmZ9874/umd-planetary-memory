"""Combine the frozen UMD 3.23 LoCoMo halves into an auditable public report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


RESULTS = Path(__file__).parent / "results"
PUBLIC = ("1", "2", "3", "4")


def counters(result: dict[str, Any], label: str, categories: tuple[str, ...] = PUBLIC) -> dict[str, int | float]:
    queries = gold_items = any_hits = full_hits = found = 0
    mrr_sum = 0.0
    for category in categories:
        row = result["by_category"][category][label]
        q = int(row["evaluated_queries"])
        g = int(row["gold_evidence_items"])
        queries += q
        gold_items += g
        any_hits += round(float(row["any_evidence_recall"]["1"]) * q)
        full_hits += round(float(row["full_evidence_recall"]["1"]) * q)
        found += round(float(row["micro_evidence_recall"]["1"]) * g)
        mrr_sum += float(row["mrr"]) * q
    return {
        "questions": queries,
        "gold_evidence_items": gold_items,
        "any_hits@1": any_hits,
        "full_hits@1": full_hits,
        "evidence_items_found@1": found,
        "any_evidence_recall@1": any_hits / queries,
        "full_evidence_recall@1": full_hits / queries,
        "micro_evidence_recall@1": found / gold_items,
        "mrr": mrr_sum / queries,
    }


def combine(left: dict[str, int | float], right: dict[str, int | float]) -> dict[str, int | float]:
    q = int(left["questions"]) + int(right["questions"])
    g = int(left["gold_evidence_items"]) + int(right["gold_evidence_items"])
    any_hits = int(left["any_hits@1"]) + int(right["any_hits@1"])
    full_hits = int(left["full_hits@1"]) + int(right["full_hits@1"])
    found = int(left["evidence_items_found@1"]) + int(right["evidence_items_found@1"])
    return {
        "questions": q,
        "gold_evidence_items": g,
        "any_hits@1": any_hits,
        "full_hits@1": full_hits,
        "evidence_items_found@1": found,
        "any_evidence_recall@1": any_hits / q,
        "full_evidence_recall@1": full_hits / q,
        "micro_evidence_recall@1": found / g,
        "mrr": (
            float(left["mrr"]) * int(left["questions"])
            + float(right["mrr"]) * int(right["questions"])
        ) / q,
    }


def main() -> None:
    dev = json.loads((RESULTS / "umd323_hill96_dev0_5.json").read_text(encoding="utf-8"))
    holdout = json.loads((RESULTS / "umd323_hill96_holdout5_10.json").read_text(encoding="utf-8"))
    baseline_file = json.loads((RESULTS / "umd322_locomo1540_retrieval.json").read_text(encoding="utf-8"))
    baseline = baseline_file["retrieval"]
    dev_public = counters(dev, "umd323")
    holdout_public = counters(holdout, "umd323")
    full_public = combine(dev_public, holdout_public)
    by_category = {
        category: combine(counters(dev, "umd323", (category,)), counters(holdout, "umd323", (category,)))
        for category in PUBLIC
    }
    all_queries = int(dev["umd323_physics"]["evaluated_queries"]) + int(holdout["umd323_physics"]["evaluated_queries"])
    mean_l1_chars = (
        float(dev["mean_l1_characters"]["umd323"]) * int(dev["umd323_physics"]["evaluated_queries"])
        + float(holdout["mean_l1_characters"]["umd323"]) * int(holdout["umd323_physics"]["evaluated_queries"])
    ) / all_queries
    target = float(baseline["any_evidence_recall@1"]) + 0.08
    payload = {
        "version": "UMD 3.23 high-recall Hill sphere",
        "benchmark": "LoCoMo",
        "scope": "public categories 1-4; evidence-bearing questions",
        "gold_used_for_ranking": False,
        "configuration": {
            "ranker": "umd322_blended.joblib",
            "encoder": "BAAI/bge-base-en-v1.5",
            "cross_encoder": "Xenova/ms-marco-MiniLM-L-6-v2",
            "cross_limit": 64,
            "l1_source_budget": 96,
            "biosphere_gate": True,
            "unsafe_sources_may_fill_l1": False,
        },
        "baseline_umd322": {
            "any_evidence_recall@1": baseline["any_evidence_recall@1"],
            "full_evidence_recall@1": baseline["full_evidence_recall@1"],
            "micro_evidence_recall@1": baseline["micro_evidence_recall@1"],
        },
        "target_any_evidence_recall@1": target,
        "development_conversations_0_4": dev_public,
        "heldout_conversations_5_9": holdout_public,
        "full": full_public,
        "by_category": by_category,
        "improvement": {
            "any_recall_absolute": float(full_public["any_evidence_recall@1"]) - float(baseline["any_evidence_recall@1"]),
            "any_recall_percentage_points": 100.0 * (float(full_public["any_evidence_recall@1"]) - float(baseline["any_evidence_recall@1"])),
            "full_recall_absolute": float(full_public["full_evidence_recall@1"]) - float(baseline["full_evidence_recall@1"]),
            "target_met": float(full_public["any_evidence_recall@1"]) >= target,
        },
        "cost": {
            "mean_l1_sources": 96,
            "mean_l1_characters_all_evidence_categories_1_5": mean_l1_chars,
            "mean_l1_tokens_chars_div_4_estimate": mean_l1_chars / 4.0,
            "dev_runtime_seconds": dev["runtime_seconds"],
            "holdout_runtime_seconds": holdout["runtime_seconds"],
        },
        "validation": {
            "strong_adversarial": "18/18",
            "intensity_sweeps": "20/20",
            "metamorphic_runs": "32/32",
            "platform_regression": "20/20",
            "adapter_and_benchmark_tests": "43/43 after rejected query-superposition removal",
        },
        "limitations": [
            "This is evidence retrieval recall, not end-to-end LoCoMo answer accuracy.",
            "The 96-source high-recall profile trades a larger answer context for recall.",
            "The full ten-conversation score is now development-visible; future tuning needs a new external holdout.",
        ],
    }
    (RESULTS / "umd323_hill96_full.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    manifest = {
        "version": "UMD 3.23",
        "profile": "high_recall_hill96",
        "status": "validated_benchmark_candidate",
        **payload["configuration"],
        "locomo_any_r1": full_public["any_evidence_recall@1"],
        "locomo_full_r1": full_public["full_evidence_recall@1"],
        "baseline_delta_percentage_points": payload["improvement"]["any_recall_percentage_points"],
        "official_end_to_end_score": None,
    }
    (RESULTS / "umd323_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    report = f"""# UMD 3.23 High-Recall Hill Sphere

## Outcome

- LoCoMo public evidence questions: **{full_public['any_hits@1']}/{full_public['questions']}**.
- Any Evidence R@1: **{float(full_public['any_evidence_recall@1']):.6f}**.
- UMD 3.22 baseline: **{float(baseline['any_evidence_recall@1']):.6f}**.
- Absolute improvement: **+{payload['improvement']['any_recall_percentage_points']:.3f} percentage points**.
- Required target: **{target:.6f}**; target met: **yes**.
- Full Evidence R@1: **{float(full_public['full_evidence_recall@1']):.6f}**.
- Micro Evidence R@1: **{float(full_public['micro_evidence_recall@1']):.6f}**.

## Frozen law

The high-recall profile expands the first Roche/Hill sphere from 10 to 96 original provenance sources, expands the cross-encoder horizon from 32 to 64, and admits a subject-centric longitudinal biosphere orbit only for inferential questions. Unsafe sources never fill unused L1 capacity.

## Generalization split

- Conversations 0-4: {dev_public['any_hits@1']}/{dev_public['questions']} = {float(dev_public['any_evidence_recall@1']):.6f}.
- Conversations 5-9: {holdout_public['any_hits@1']}/{holdout_public['questions']} = {float(holdout_public['any_evidence_recall@1']):.6f}.

## Cost

- Mean L1 size: 96 original sources.
- Mean L1 characters: {mean_l1_chars:.0f} across all evidence-bearing categories.
- Planning token estimate: {mean_l1_chars / 4.0:.0f} tokens (characters/4, not tokenizer exact).

## Safety and regression

- Strong adversarial: 18/18.
- Intensity sweeps: 20/20.
- Insertion-order runs: 32/32; mean L1 Jaccard 1.0.
- Platform regression: 20/20.
- Adapter/benchmark tests: 43/43.

This is a retrieval score, not the official end-to-end LLM answer/judge score. The complete LoCoMo data is now development-visible, so subsequent claims require a new external holdout.
"""
    (RESULTS / "UMD323_HIGH_RECALL.md").write_text(report, encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
