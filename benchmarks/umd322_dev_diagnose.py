"""Development-only audit of UMD 3.22 safety reasons on LoCoMo evidence."""

from __future__ import annotations

import json

import joblib

from benchmarks.ai_memory_extended import OrbitIndex
from benchmarks.public_benchmarks import _locomo_sessions, normalize_locomo_evidence_ids
from benchmarks.umd318_cross import OnnxCrossEncoder
from benchmarks.umd39_benchmarks import FastEmbedEncoder, MODEL_CACHE, RESULTS


def main() -> None:
    sample = json.loads((RESULTS.parent / "data" / "locomo10.json").read_text(encoding="utf-8"))[4]
    _, sessions, dates = _locomo_sessions(sample)
    turns = [turn for session in sessions for turn in session]
    texts = [f'{turn.get("speaker", "")}: {turn.get("text", "")} {turn.get("blip_caption", "")}' for turn in turns]
    ids = [str(turn["dia_id"]) for turn in turns]
    groups = [session for session, values in enumerate(sessions) for _ in values]
    encoder = FastEmbedEncoder(
        model_name="BAAI/bge-base-en-v1.5", dimensions=768,
        cache_dir=MODEL_CACHE, batch_size=64, cache_size=16384, threads=4,
    )
    index = OrbitIndex(
        texts, groups, ids, encoder, dates, physics_v322=True,
        roche_budget_v318=10, matter_neighbor_budget=0,
        cross_encoder=OnnxCrossEncoder(MODEL_CACHE), cross_limit=32,
        roche_ranker=joblib.load(RESULTS / "umd322_blended_cv.joblib"),
    )
    audited = []
    for row in sample["qa"]:
        gold = normalize_locomo_evidence_ids(row.get("evidence"))
        if not gold:
            continue
        capsules = index.retrieve(str(row["question"]))
        l1 = set(capsules[0])
        reasons = index.last_roche_diagnostics.get("unsafe_reasons", {})
        flagged_gold = {
            ids[int(source)]: value for source, value in reasons.items()
            if ids[int(source)] in gold
        }
        if flagged_gold:
            audited.append({
                "question": row["question"], "category": row.get("category"),
                "gold": sorted(gold), "gold_in_l1": sorted(gold & l1),
                "flagged_gold": flagged_gold,
            })
    output = RESULTS / "umd322_dev_safety_audit.json"
    output.write_text(json.dumps({"flagged_questions": len(audited), "items": audited}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"flagged_questions": len(audited), "items": audited}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
