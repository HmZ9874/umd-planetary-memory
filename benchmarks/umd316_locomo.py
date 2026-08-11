"""UMD 3.16 LoCoMo retrieval evaluation with a frozen 3.15 control.

The runner is gold-blind during ranking.  Evidence identifiers are consumed
only after both result lists have been produced.  Conversation slices allow
development and held-out evaluation to remain separate.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks.ai_memory_extended import OrbitIndex
from benchmarks.public_benchmarks import _locomo_sessions, normalize_locomo_evidence_ids
from benchmarks.umd39_benchmarks import MODEL_CACHE, RESULTS, CapsuleRecallAccumulator, FastEmbedEncoder


def _late_maxsim(query: np.ndarray, passage: np.ndarray) -> float:
    if not query.size or not passage.size:
        return 0.0
    return float(np.max(query @ passage.T, axis=1).sum())


def _aggregate(items: list[dict[str, Any]]) -> dict[str, Any]:
    metric = CapsuleRecallAccumulator((1, 5, 10))
    # Rehydrate exact counters instead of averaging conversation rates.
    queries = sum(item["evaluated_queries"] for item in items)
    gold = sum(item["gold_evidence_items"] for item in items)
    if not queries:
        return metric.result()
    return {
        "evaluated_queries": queries,
        "gold_evidence_items": gold,
        "retrieval_unit": "evidence_capsule",
        "mrr": sum(item["mrr"] * item["evaluated_queries"] for item in items) / queries,
        "any_evidence_recall": {
            key: sum(item["any_evidence_recall"][key] * item["evaluated_queries"] for item in items) / queries
            for key in ("1", "5", "10")
        },
        "full_evidence_recall": {
            key: sum(item["full_evidence_recall"][key] * item["evaluated_queries"] for item in items) / queries
            for key in ("1", "5", "10")
        },
        "micro_evidence_recall": {
            key: sum(item["micro_evidence_recall"][key] * item["gold_evidence_items"] for item in items) / max(1, gold)
            for key in ("1", "5", "10")
        },
    }


def run(
    path: Path,
    encoder: FastEmbedEncoder,
    *,
    start: int = 0,
    stop: int | None = None,
    use_late: bool = False,
    use_v317: bool = False,
    use_v318: bool = False,
    use_v320: bool = False,
    use_v321: bool = False,
    use_v322: bool = False,
    use_v323: bool = False,
    use_v324: bool = False,
    roche_budget: int = 10,
    matter_neighbor_budget: int = 0,
    cross_encoder: Any | None = None,
    cross_limit: int = 160,
    roche_ranker: Any | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    dataset = json.loads(path.read_text(encoding="utf-8"))[max(0, start):stop]
    late_encoder = None
    if use_late:
        from fastembed import LateInteractionTextEmbedding

        late_encoder = LateInteractionTextEmbedding(
            model_name="answerdotai/answerai-colbert-small-v1",
            cache_dir=str(MODEL_CACHE),
            threads=4,
        )

    control_records: list[dict[str, Any]] = []
    physics_records: list[dict[str, Any]] = []
    by_category_control: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_category_physics: dict[str, list[dict[str, Any]]] = defaultdict(list)
    chars_control: list[int] = []
    chars_physics: list[int] = []
    l1_chars_control: list[int] = []
    l1_chars_physics: list[int] = []
    l1_sources_control: list[int] = []
    l1_sources_physics: list[int] = []

    for conversation_number, sample in enumerate(dataset, 1):
        _, sessions, dates = _locomo_sessions(sample)
        turns = [turn for session in sessions for turn in session]
        texts = [
            f'{turn.get("speaker", "")}: {turn.get("text", "")} {turn.get("blip_caption", "")}'
            for turn in turns
        ]
        ids = [str(turn["dia_id"]) for turn in turns]
        groups = [session for session, values in enumerate(sessions) for _ in values]
        next_version = use_v317 or use_v318 or use_v320 or use_v321 or use_v322 or use_v323 or use_v324
        control = OrbitIndex(
            texts, groups, ids, encoder, dates,
            physics_v316=next_version,
            physics_v317=use_v318 or use_v320 or use_v321 or use_v322 or use_v323 or use_v324,
        )
        physics = OrbitIndex(
            texts, groups, ids, encoder, dates,
            physics_v316=True, physics_v317=next_version,
            physics_v318=use_v318 or use_v320 or use_v321 or use_v322 or use_v323 or use_v324,
            physics_v320=use_v320 or use_v321 or use_v322 or use_v323 or use_v324,
            physics_v321=use_v321 or use_v322 or use_v323 or use_v324,
            physics_v322=use_v322 or use_v323 or use_v324,
            physics_v323=use_v323 or use_v324,
            physics_v324=use_v324,
            roche_budget_v318=roche_budget,
            matter_neighbor_budget=matter_neighbor_budget,
            cross_encoder=cross_encoder,
            cross_limit=cross_limit,
            roche_ranker=roche_ranker,
        )
        passage_vectors = (
            list(late_encoder.passage_embed(texts, batch_size=16))
            if late_encoder is not None else None
        )
        queries = [str(row["question"]) for row in sample["qa"]]
        query_vectors = (
            list(late_encoder.query_embed(queries, batch_size=16))
            if late_encoder is not None else None
        )
        control_metric = CapsuleRecallAccumulator((1, 5, 10))
        physics_metric = CapsuleRecallAccumulator((1, 5, 10))
        category_control: dict[str, CapsuleRecallAccumulator] = defaultdict(
            lambda: CapsuleRecallAccumulator((1, 5, 10))
        )
        category_physics: dict[str, CapsuleRecallAccumulator] = defaultdict(
            lambda: CapsuleRecallAccumulator((1, 5, 10))
        )
        for number, row in enumerate(sample["qa"]):
            gold = normalize_locomo_evidence_ids(row.get("evidence"))
            if not gold:
                continue
            late_scores = None
            if passage_vectors is not None and query_vectors is not None:
                late_scores = [
                    _late_maxsim(query_vectors[number], passage)
                    for passage in passage_vectors
                ]
            query = queries[number]
            control_capsules = control.retrieve(query)
            physics_capsules = physics.retrieve(query, late_scores=late_scores)
            control_metric.add(control_capsules, gold)
            physics_metric.add(physics_capsules, gold)
            category = str(row.get("category", "unknown"))
            category_control[category].add(control_capsules, gold)
            category_physics[category].add(physics_capsules, gold)
            id_to_text = dict(zip(ids, texts))
            chars_control.append(sum(
                len(id_to_text[source])
                for source in {item for capsule in control_capsules for item in capsule}
            ))
            chars_physics.append(sum(
                len(id_to_text[source])
                for source in {item for capsule in physics_capsules for item in capsule}
            ))
            control_l1 = set(control_capsules[0]) if control_capsules else set()
            physics_l1 = set(physics_capsules[0]) if physics_capsules else set()
            l1_sources_control.append(len(control_l1))
            l1_sources_physics.append(len(physics_l1))
            l1_chars_control.append(sum(len(id_to_text[source]) for source in control_l1))
            l1_chars_physics.append(sum(len(id_to_text[source]) for source in physics_l1))
        control_records.append(control_metric.result())
        physics_records.append(physics_metric.result())
        for category, metric in category_control.items():
            by_category_control[category].append(metric.result())
        for category, metric in category_physics.items():
            by_category_physics[category].append(metric.result())
        version = "324" if use_v324 else "323" if use_v323 else "322" if use_v322 else "321" if use_v321 else "320" if use_v320 else "318" if use_v318 else "317" if use_v317 else "316"
        print(f"UMD{version} LoCoMo progress: {conversation_number}/{len(dataset)}", flush=True)

    control_key = "umd317_control" if (use_v318 or use_v320 or use_v321 or use_v322 or use_v323 or use_v324) else "umd316_control" if use_v317 else "umd315_control"
    candidate_key = "umd324_physics" if use_v324 else "umd323_physics" if use_v323 else "umd322_physics" if use_v322 else "umd321_physics" if use_v321 else "umd320_physics" if use_v320 else "umd318_physics" if use_v318 else "umd317_physics" if use_v317 else "umd316_physics"
    control_label = "umd317" if (use_v318 or use_v320 or use_v321 or use_v322 or use_v323 or use_v324) else "umd316" if use_v317 else "umd315"
    candidate_label = "umd324" if use_v324 else "umd323" if use_v323 else "umd322" if use_v322 else "umd321" if use_v321 else "umd320" if use_v320 else "umd318" if use_v318 else "umd317" if use_v317 else "umd316"
    return {
        "benchmark": "LoCoMo",
        "slice": [max(0, start), stop],
        "gold_used_for_ranking": False,
        "late_interaction": use_late,
        control_key: _aggregate(control_records),
        candidate_key: _aggregate(physics_records),
        "by_category": {
            category: {
                control_label: _aggregate(by_category_control.get(category, [])),
                candidate_label: _aggregate(by_category_physics.get(category, [])),
            }
            for category in sorted(set(by_category_control) | set(by_category_physics))
        },
        "mean_retrieved_characters": {
            control_label: sum(chars_control) / max(1, len(chars_control)),
            candidate_label: sum(chars_physics) / max(1, len(chars_physics)),
        },
        "mean_l1_sources": {
            control_label: sum(l1_sources_control) / max(1, len(l1_sources_control)),
            candidate_label: sum(l1_sources_physics) / max(1, len(l1_sources_physics)),
        },
        "mean_l1_characters": {
            control_label: sum(l1_chars_control) / max(1, len(l1_chars_control)),
            candidate_label: sum(l1_chars_physics) / max(1, len(l1_chars_physics)),
        },
        "runtime_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path(__file__).parent / "data" / "locomo10.json")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int)
    parser.add_argument("--late", action="store_true")
    parser.add_argument("--v317", action="store_true")
    parser.add_argument("--v318", action="store_true")
    parser.add_argument("--v320", action="store_true")
    parser.add_argument("--v321", action="store_true")
    parser.add_argument("--v322", action="store_true")
    parser.add_argument("--v323", action="store_true")
    parser.add_argument("--v324", action="store_true")
    parser.add_argument("--roche-budget", type=int, default=10)
    parser.add_argument("--matter-neighbor-budget", type=int, default=0)
    parser.add_argument("--encoder-model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--encoder-dimensions", type=int, default=384)
    parser.add_argument("--cross", action="store_true")
    parser.add_argument("--cross-limit", type=int, default=32)
    parser.add_argument("--ranker", type=Path)
    parser.add_argument("--output", type=Path, default=RESULTS / "umd316_locomo.json")
    args = parser.parse_args()
    encoder = FastEmbedEncoder(
        model_name=args.encoder_model, dimensions=args.encoder_dimensions,
        cache_dir=MODEL_CACHE, batch_size=64, cache_size=16384, threads=4,
    )
    cross_encoder = None
    if args.cross or args.v321 or args.v322 or args.v323 or args.v324:
        from benchmarks.umd318_cross import OnnxCrossEncoder
        cross_encoder = OnnxCrossEncoder(MODEL_CACHE)
    roche_ranker = None
    if args.ranker is not None:
        import joblib
        roche_ranker = joblib.load(args.ranker)
    result = run(
        args.data, encoder, start=args.start, stop=args.stop,
        use_late=args.late, use_v317=args.v317, use_v318=args.v318, use_v320=args.v320,
        use_v321=args.v321,
        use_v322=args.v322,
        use_v323=args.v323,
        use_v324=args.v324,
        roche_budget=args.roche_budget,
        matter_neighbor_budget=args.matter_neighbor_budget,
        cross_encoder=cross_encoder,
        cross_limit=args.cross_limit,
        roche_ranker=roche_ranker,
    )
    result["encoder"] = encoder.metadata()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
