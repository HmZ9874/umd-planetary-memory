"""Check that UMD 3.16 never loses the frozen UMD 3.15 R@10 source set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmarks.ai_memory_extended import OrbitIndex, _evidence_session_ids, _expanded
from benchmarks.umd39_benchmarks import MODEL_CACHE, CapsuleRecallAccumulator, FastEmbedEncoder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("persona", type=Path)
    args = parser.parse_args()
    persona = args.persona
    sessions: list[str] = []
    dates: list[str] = []
    ids: list[str] = []
    for path in sorted((persona / "conversations").glob("session_*.json")):
        item = json.loads(path.read_text(encoding="utf-8"))
        sessions.append(" ".join(
            f"{turn.get('speaker', '')}: {turn.get('message', '')}"
            for turn in item.get("conversation", [])
        ) or "empty conversation")
        dates.append(str(item.get("date", "")))
        ids.append(str(item.get("session_id")))
    encoder = FastEmbedEncoder(cache_dir=MODEL_CACHE, batch_size=128, cache_size=32768, threads=16)
    control = OrbitIndex(
        sessions, list(range(len(sessions))), ids, encoder, dates,
        neural_candidate_pool=64,
    )
    physics = OrbitIndex(
        sessions, list(range(len(sessions))), ids, encoder, dates,
        neural_candidate_pool=192, physics_v316=True,
    )
    evaluation = json.loads(next(persona.glob("evaluation_questions_*.json")).read_text(encoding="utf-8"))
    failures = []
    total = 0
    control_metric = CapsuleRecallAccumulator((1, 5, 10))
    physics_metric = CapsuleRecallAccumulator((1, 5, 10))
    for rows in evaluation.get("questions", {}).values():
        for row in rows:
            total += 1
            query = str(row.get("question", ""))
            control_capsules = control.retrieve(query)
            physics_capsules = physics.retrieve(query)
            old = _expanded(control_capsules)
            new = _expanded(physics_capsules)
            gold = _evidence_session_ids(row.get("memory_evidence"))
            if gold:
                control_metric.add(control_capsules, gold)
                physics_metric.add(physics_capsules, gold)
            if not old <= new:
                failures.append({
                    "question_id": row.get("question_id"),
                    "question": query,
                    "lost": sorted(old - new),
                    "old_count": len(old),
                    "new_count": len(new),
                })
    print(json.dumps({
        "queries": total,
        "failures": failures,
        "umd315_control": control_metric.result(),
        "umd316_physics": physics_metric.result(),
    }, ensure_ascii=False, indent=2))
    raise SystemExit(bool(failures))


if __name__ == "__main__":
    main()
