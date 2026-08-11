"""Gold-after-retrieval audit for UMD 3.17 LoCoMo failure ceilings."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from benchmarks.ai_memory_extended import OrbitIndex
from benchmarks.public_benchmarks import _locomo_sessions, normalize_locomo_evidence_ids
from benchmarks.umd39_benchmarks import MODEL_CACHE, RESULTS, FastEmbedEncoder


def question_shape(question: str) -> str:
    first = question.strip().split(maxsplit=1)[0].casefold() if question.strip() else "empty"
    if first in {"what", "when", "where", "who", "why", "how", "which", "did", "does", "is", "are", "has", "have", "was", "were"}:
        return first
    return "other"


def run(path: Path, encoder: FastEmbedEncoder, start: int, stop: int | None) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))[start:stop]
    totals = Counter()
    by_category: dict[str, Counter] = defaultdict(Counter)
    by_shape: dict[str, Counter] = defaultdict(Counter)
    miss_examples: list[dict[str, Any]] = []
    l1_sizes: list[int] = []
    gold_counts: list[int] = []
    for sample_number, sample in enumerate(data, start):
        _, sessions, dates = _locomo_sessions(sample)
        turns = [turn for session in sessions for turn in session]
        texts = [
            f'{turn.get("speaker", "")}: {turn.get("text", "")} {turn.get("blip_caption", "")}'
            for turn in turns
        ]
        ids = [str(turn["dia_id"]) for turn in turns]
        groups = [session for session, values in enumerate(sessions) for _ in values]
        index = OrbitIndex(
            texts, groups, ids, encoder, dates,
            physics_v316=True, physics_v317=True,
        )
        for row in sample["qa"]:
            gold = normalize_locomo_evidence_ids(row.get("evidence"))
            if not gold:
                continue
            query = str(row["question"])
            capsules = index.retrieve(query)
            source_sets = [set(capsule) for capsule in capsules]
            first_rank = next(
                (rank for rank, values in enumerate(source_sets, 1) if values & gold),
                None,
            )
            status = (
                "l1_hit" if first_rank == 1
                else "rerankable_top10" if first_rank is not None and first_rank <= 10
                else "candidate_miss"
            )
            totals[status] += 1
            totals["queries"] += 1
            category = str(row.get("category", "unknown"))
            shape = question_shape(query)
            by_category[category][status] += 1
            by_category[category]["queries"] += 1
            by_shape[shape][status] += 1
            by_shape[shape]["queries"] += 1
            l1_sizes.append(len(source_sets[0]) if source_sets else 0)
            gold_counts.append(len(gold))
            if status != "l1_hit" and len(miss_examples) < 80:
                miss_examples.append({
                    "sample": sample_number,
                    "category": category,
                    "shape": shape,
                    "question": query,
                    "gold_count": len(gold),
                    "first_gold_capsule": first_rank,
                    "l1_sources": list(capsules[0]) if capsules else [],
                })
        print(f"UMD318 audit progress: {sample_number - start + 1}/{len(data)}", flush=True)

    def rates(counter: Counter) -> dict[str, float | int]:
        q = counter["queries"]
        return {
            "queries": q,
            "r1": counter["l1_hit"] / max(1, q),
            "rerankable_top10": counter["rerankable_top10"] / max(1, q),
            "candidate_miss": counter["candidate_miss"] / max(1, q),
        }

    return {
        "slice": [start, stop],
        "gold_used_for_ranking": False,
        "gold_used_after_retrieval_for_audit": True,
        "overall": rates(totals),
        "by_category": {key: rates(value) for key, value in sorted(by_category.items())},
        "by_question_shape": {key: rates(value) for key, value in sorted(by_shape.items())},
        "mean_l1_sources": sum(l1_sizes) / max(1, len(l1_sizes)),
        "max_l1_sources": max(l1_sizes, default=0),
        "mean_gold_sources": sum(gold_counts) / max(1, len(gold_counts)),
        "miss_examples": miss_examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path(__file__).parent / "data" / "locomo10.json")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int)
    parser.add_argument("--output", type=Path, default=RESULTS / "umd318_locomo_audit.json")
    args = parser.parse_args()
    encoder = FastEmbedEncoder(cache_dir=MODEL_CACHE, batch_size=64, cache_size=16384, threads=4)
    result = run(args.data, encoder, args.start, args.stop)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "miss_examples"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
