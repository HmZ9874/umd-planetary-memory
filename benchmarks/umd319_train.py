"""Train a development-only UMD 3.19 Roche-lobe evidence ranker."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from benchmarks.umd319_ranker import LightGBMProbabilityRanker

from benchmarks.ai_memory_extended import OrbitIndex
from benchmarks.public_benchmarks import _locomo_sessions, normalize_locomo_evidence_ids
from benchmarks.umd39_benchmarks import MODEL_CACHE, RESULTS, FastEmbedEncoder


def examples(
    path: Path, encoder: FastEmbedEncoder, start: int, stop: int, *,
    matter_v320: bool = False, matter_v321: bool = False,
    cross_encoder=None, cross_limit: int = 32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))[start:stop]
    features: list[list[float]] = []
    labels: list[int] = []
    queries = positives = candidate_hits = 0
    groups_per_query: list[int] = []
    for offset, sample in enumerate(data, start):
        _, sessions, dates = _locomo_sessions(sample)
        turns = [turn for session in sessions for turn in session]
        texts = [f'{turn.get("speaker", "")}: {turn.get("text", "")} {turn.get("blip_caption", "")}' for turn in turns]
        ids = [str(turn["dia_id"]) for turn in turns]
        groups = [session for session, values in enumerate(sessions) for _ in values]
        index = OrbitIndex(
            texts, groups, ids, encoder, dates,
            physics_v316=True, physics_v317=True, physics_v318=True,
            physics_v320=matter_v320,
            physics_v321=matter_v321,
            roche_budget_v318=10,
            cross_encoder=cross_encoder,
            cross_limit=cross_limit,
        )
        for row in sample["qa"]:
            gold = normalize_locomo_evidence_ids(row.get("evidence"))
            if not gold:
                continue
            index.retrieve(str(row["question"]))
            diagnostic = index.last_roche_diagnostics
            sources = diagnostic.get("sources", [])
            rows = diagnostic.get("features", [])
            source_labels = [int(ids[source] in gold) for source in sources]
            features.extend(rows)
            labels.extend(source_labels)
            groups_per_query.append(len(rows))
            positives += sum(source_labels)
            candidate_hits += bool(sum(source_labels))
            queries += 1
        print(f"UMD319 feature progress: {offset - start + 1}/{len(data)}", flush=True)
    return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.int8), np.asarray(groups_per_query, dtype=np.int32), {
        "conversations": len(data), "queries": queries, "rows": len(labels),
        "positive_rows": positives, "candidate_query_recall": candidate_hits / max(1, queries),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path(__file__).parent / "data" / "locomo10.json")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=5)
    parser.add_argument("--output", type=Path, default=RESULTS / "umd319_roche_ranker.joblib")
    parser.add_argument("--metadata", type=Path, default=RESULTS / "umd319_roche_ranker.json")
    parser.add_argument("--algorithm", choices=("hist", "lambdarank"), default="hist")
    parser.add_argument("--cache-output", type=Path)
    parser.add_argument("--matter-v320", action="store_true")
    parser.add_argument("--matter-v321", action="store_true")
    parser.add_argument("--cross-limit", type=int, default=32)
    parser.add_argument("--features-only", action="store_true")
    args = parser.parse_args()
    encoder = FastEmbedEncoder(
        model_name="BAAI/bge-base-en-v1.5", dimensions=768,
        cache_dir=MODEL_CACHE, batch_size=64, cache_size=16384, threads=4,
    )
    cross_encoder = None
    if args.matter_v321:
        from benchmarks.umd318_cross import OnnxCrossEncoder
        cross_encoder = OnnxCrossEncoder(MODEL_CACHE)
    x, y, group, metadata = examples(
        args.data, encoder, args.start, args.stop,
        matter_v320=args.matter_v320 or args.matter_v321,
        matter_v321=args.matter_v321, cross_encoder=cross_encoder,
        cross_limit=args.cross_limit,
    )
    if args.cache_output is not None:
        np.savez_compressed(args.cache_output, x=x, y=y, group=group)
    if args.features_only:
        metadata.update({
            "feature_dimensions": int(x.shape[1]),
            "matter_v320": args.matter_v320,
            "matter_v321": args.matter_v321,
            "cross_limit": args.cross_limit if args.matter_v321 else 0,
            "features_only": True,
        })
        args.metadata.parent.mkdir(parents=True, exist_ok=True)
        args.metadata.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(metadata, ensure_ascii=False, indent=2))
        return
    if args.algorithm == "lambdarank":
        from lightgbm import LGBMRanker
        base_model = LGBMRanker(
            objective="lambdarank", metric="ndcg", eval_at=(1, 10),
            n_estimators=260, learning_rate=0.04, num_leaves=31,
            min_child_samples=30, reg_lambda=2.0, verbosity=-1,
            random_state=319, n_jobs=4,
        )
        base_model.fit(x, y, group=group.tolist())
        model = LightGBMProbabilityRanker(base_model)
    else:
        model = HistGradientBoostingClassifier(
            learning_rate=0.08, max_iter=180, max_leaf_nodes=31,
            min_samples_leaf=30, l2_regularization=1.0,
            class_weight="balanced", random_state=317,
        )
        model.fit(x, y)
    probability = model.predict_proba(x)[:, 1]
    metadata.update({
        "feature_dimensions": int(x.shape[1]),
        "training_average_precision": float(average_precision_score(y, probability)),
        "training_roc_auc": float(roc_auc_score(y, probability)),
        "gold_used_only_for_development_training": True,
        "heldout_conversations_used": False,
        "algorithm": args.algorithm,
        "matter_v320": args.matter_v320,
        "matter_v321": args.matter_v321,
        "cross_limit": args.cross_limit if args.matter_v321 else 0,
        "model": (
            model.model.get_params() if isinstance(model, LightGBMProbabilityRanker)
            else model.get_params()
        ),
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.output)
    args.metadata.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
