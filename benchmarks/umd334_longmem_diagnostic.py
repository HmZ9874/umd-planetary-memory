"""Development-only LongMemEval rank-one diagnostics for UMD 3.34.

This utility selects failures from the already-published UMD 3.28.2 checkpoint,
runs retrieval before reading evidence IDs, and prints compact source snippets.
It is not a benchmark runner and its output must not be reported as blind.
"""

from __future__ import annotations

import json

from benchmarks.umd3282_frozen_exam import DATA, _index, _long_session_text
from benchmarks.umd39_benchmarks import MODEL_CACHE, RESULTS, FastEmbedEncoder


def main(limit: int = 100) -> None:
    dataset = json.loads((DATA / "longmemeval_s_cleaned.json").read_text(encoding="utf-8"))
    baseline = json.loads(
        (RESULTS / "umd3282_exam_longmemeval_checkpoint.json").read_text(encoding="utf-8")
    )
    failures = [
        number for number in range(min(limit, len(dataset)))
        if not baseline[str(number)]["is_abstention"]
        and baseline[str(number)]["strict"]["any_evidence_recall"]["1"] < 1.0
    ]
    candidate_path = RESULTS / "umd334_longmemeval_checkpoint.json"
    if candidate_path.exists():
        candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
        failures.extend(
            number for number in range(min(limit, len(dataset)))
            if baseline[str(number)]["strict"]["any_evidence_recall"]["1"]
            > candidate.get(str(number), {}).get("strict", {}).get(
                "any_evidence_recall", {}
            ).get("1", 1.0)
        )
        failures = sorted(set(failures))
    encoder = FastEmbedEncoder(
        cache_dir=MODEL_CACHE, batch_size=128, cache_size=32768, threads=16,
    )
    for number in failures:
        item = dataset[number]
        ids = [str(value) for value in item["haystack_session_ids"]]
        texts = [_long_session_text(item, index) for index in range(len(ids))]
        index = _index(
            texts, list(range(len(ids))), ids, encoder, item["haystack_dates"],
            physics_v334=True,
        )
        channels = index.retrieve_compact_channels(str(item["question"]))
        gold = {str(value) for value in item.get("answer_session_ids") or []}
        top = channels["atomic_answer"][0][0]
        diagnostic = channels["query_fission"]
        periapsis_number = diagnostic.get("periapsis_source")
        periapsis = ids[periapsis_number] if isinstance(periapsis_number, int) else None
        print(json.dumps({
            "number": number,
            "question": item["question"],
            "gold": sorted(gold),
            "top": top,
            "top_hit": top in gold,
            "periapsis": periapsis,
            "periapsis_hit": periapsis in gold,
            "diagnostic": diagnostic,
            "top_text": texts[ids.index(top)][:500],
            "periapsis_text": texts[ids.index(periapsis)][:500] if periapsis in ids else None,
            "gold_text": [texts[ids.index(source)][:500] for source in gold if source in ids],
        }, ensure_ascii=False))


if __name__ == "__main__":
    main()
