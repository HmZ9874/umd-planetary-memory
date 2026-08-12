"""Continued gold-blind UMD 3.28.2 retrieval exams.

These adapters deliberately report retrieval proxies rather than official
end-to-end agent or LLM-judge scores.  Every query is ranked before its answer
or derived relevance set is inspected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import pyarrow.ipc as ipc
import pyarrow.parquet as pq

from benchmarks.umd3282_frozen_exam import KS, _aggregate, _index as _frozen_index
from benchmarks.umd39_benchmarks import (
    MODEL_CACHE,
    CapsuleRecallAccumulator,
    FastEmbedEncoder,
)


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data_exam"
RESULTS = ROOT / "results"
CHUNK_CHARS = 1200
CHUNK_OVERLAP = 200


def _index(
    texts: Sequence[str], groups: Sequence[int], source_ids: Sequence[str],
    encoder: FastEmbedEncoder, dates: Sequence[str] | None = None,
    *, physics_v331: bool = False, physics_v332: bool = False,
    physics_v333: bool = False, physics_v334: bool = False,
    physics_v335: bool = False,
    physics_v336: bool = False,
):
    """UMD 3.30-3.32 index with the 3.28.2 control laws conserved."""
    return _frozen_index(
        texts, groups, source_ids, encoder, dates,
        physics_v330=True,
        physics_v331=physics_v331 or physics_v332 or physics_v333 or physics_v334 or physics_v335 or physics_v336,
        physics_v332=physics_v332 or physics_v333 or physics_v334 or physics_v335 or physics_v336,
        physics_v333=physics_v333 or physics_v334 or physics_v335 or physics_v336,
        physics_v334=physics_v334 or physics_v335 or physics_v336,
        physics_v335=physics_v335 or physics_v336,
        physics_v336=physics_v336,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalize(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    return " ".join(re.findall(r"[\w]+", text, flags=re.UNICODE))


def _answer_strings(value: Any) -> list[str]:
    output: list[str] = []
    if isinstance(value, dict):
        for child in value.values():
            output.extend(_answer_strings(child))
    elif isinstance(value, (list, tuple, set)):
        for child in value:
            output.extend(_answer_strings(child))
    elif value is not None:
        normalized = _normalize(value)
        if len(normalized) >= 2:
            output.append(normalized)
    return sorted(set(output), key=lambda item: (-len(item), item))


def _fixed_chunks(text: str) -> list[str]:
    """Gold-independent character windows with a fixed frozen overlap."""
    cleaned = text.replace("\x00", " ")
    if len(cleaned) <= CHUNK_CHARS:
        return [cleaned]
    step = CHUNK_CHARS - CHUNK_OVERLAP
    return [cleaned[start:start + CHUNK_CHARS] for start in range(0, len(cleaned), step)]


def _session_chunks(text: str) -> list[str]:
    blocks = [
        block.strip()
        for block in re.split(r"(?m)(?=^\[Date: .*?Session #\d+\])", text)
        if block.strip()
    ]
    return blocks or _fixed_chunks(text)


def _gold_answer_sources(
    normalized_texts: Sequence[str], source_ids: Sequence[str], answer: Any,
) -> set[str]:
    accepted = _answer_strings(answer)
    if not accepted:
        return set()
    return {
        source_id
        for source_id, text in zip(source_ids, normalized_texts)
        if any(candidate in text for candidate in accepted)
    }


def _metric_pair() -> tuple[CapsuleRecallAccumulator, CapsuleRecallAccumulator]:
    return CapsuleRecallAccumulator(KS), CapsuleRecallAccumulator(KS)


def _add_pair(
    metrics: tuple[CapsuleRecallAccumulator, CapsuleRecallAccumulator],
    channels: dict[str, Any], gold: Iterable[str],
) -> None:
    metrics[0].add(channels["answer"], gold)
    metrics[1].add(channels["atomic_answer"], gold)


def _pair_result(
    metrics: tuple[CapsuleRecallAccumulator, CapsuleRecallAccumulator],
) -> dict[str, Any]:
    return {
        "final_capsule_retrieval": metrics[0].result(),
        "strict_source_atomic_retrieval": metrics[1].result(),
    }


def _load_checkpoint(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _write_checkpoint(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def run_memoryagentbench(
    encoder: FastEmbedEncoder, *, max_chunks: int | None = None,
    physics_v331: bool = False, physics_v332: bool = False,
    physics_v333: bool = False, physics_v334: bool = False,
    physics_v335: bool = False,
    physics_v336: bool = False,
    checkpoint_name: str | None = None,
) -> dict[str, Any]:
    """Answer-bearing passage recall for the AR and CR competencies."""
    started = time.perf_counter()
    data_root = DATA / "MemoryAgentBench" / "data"
    version = (
        "umd336" if physics_v336
        else "umd335" if physics_v335
        else "umd334" if physics_v334
        else "umd3331" if physics_v333
        else "umd3321" if physics_v332
        else "umd331" if physics_v331
        else "umd330"
    )
    checkpoint_path = RESULTS / (checkpoint_name or (
        f"{version}_memoryagentbench_capacity_{max_chunks}_checkpoint.json"
        if max_chunks is not None else f"{version}_memoryagentbench_checkpoint.json"
    ))
    checkpoint = _load_checkpoint(checkpoint_path)
    selected = {
        "Accurate_Retrieval": data_root / "Accurate_Retrieval-00000-of-00001.parquet",
        "Conflict_Resolution": data_root / "Conflict_Resolution-00000-of-00001.parquet",
    }
    contexts_total = contexts_skipped = skipped_questions = 0
    for competency, path in selected.items():
        table = pq.read_table(path)
        for row_number, row in enumerate(table.to_pylist()):
            contexts_total += 1
            key = f"{competency}/{row_number}"
            if key in checkpoint:
                continue
            texts = _fixed_chunks(str(row["context"]))
            if max_chunks is not None and len(texts) > max_chunks:
                contexts_skipped += 1
                skipped_questions += len(row["questions"])
                continue
            ids = [f"{key}/chunk/{number}" for number in range(len(texts))]
            normalized_texts = [_normalize(text) for text in texts]
            index = _index(
                texts, list(range(len(texts))), ids, encoder,
                physics_v331=physics_v331 or physics_v332 or physics_v333 or physics_v334 or physics_v335 or physics_v336,
                physics_v332=physics_v332 or physics_v333 or physics_v334 or physics_v335 or physics_v336,
                physics_v333=physics_v333 or physics_v334 or physics_v335 or physics_v336,
                physics_v334=physics_v334 or physics_v335 or physics_v336,
                physics_v335=physics_v335 or physics_v336,
                physics_v336=physics_v336,
            )
            metrics = _metric_pair()
            questions = 0
            no_answer_bearing_passage = 0
            for question_number, question in enumerate(row["questions"]):
                # The frozen retriever returns before the paired answer is read.
                channels = index.retrieve_compact_channels(str(question))
                answer = row["answers"][question_number]
                gold = _gold_answer_sources(normalized_texts, ids, answer)
                questions += 1
                if not gold:
                    no_answer_bearing_passage += 1
                    continue
                _add_pair(metrics, channels, gold)
            checkpoint[key] = {
                "questions": questions,
                "no_answer_bearing_passage": no_answer_bearing_passage,
                "chunks": len(texts),
                **_pair_result(metrics),
            }
            _write_checkpoint(checkpoint_path, checkpoint)
            print(f"MemoryAgentBench {competency}: {row_number + 1}/{table.num_rows}", flush=True)
    by_competency: dict[str, Any] = {}
    for competency in selected:
        records = [value for key, value in checkpoint.items() if key.startswith(f"{competency}/")]
        by_competency[competency] = {
            "questions": sum(record["questions"] for record in records),
            "answer_bearing_questions": sum(
                record["questions"] - record["no_answer_bearing_passage"] for record in records
            ),
            "no_answer_bearing_passage": sum(record["no_answer_bearing_passage"] for record in records),
            "chunks": sum(record["chunks"] for record in records),
            "final_capsule_retrieval": _aggregate([record["final_capsule_retrieval"] for record in records]),
            "strict_source_atomic_retrieval": _aggregate([
                record["strict_source_atomic_retrieval"] for record in records
            ]),
        }
    records = list(by_competency.values())
    return {
        "benchmark": "MemoryAgentBench",
        "umd_version": (
            "3.36" if physics_v336
            else "3.35" if physics_v335
            else "3.34" if physics_v334
            else "3.33.1" if physics_v333
            else "3.32.1" if physics_v332
            else "3.31" if physics_v331
            else "3.30"
        ),
        "status": (
            "official_data_capacity_bounded_answer_bearing_retrieval_proxy"
            if max_chunks is not None else "official_data_answer_bearing_retrieval_proxy"
        ),
        "competencies_run": ["Accurate Retrieval", "Conflict Resolution"],
        "competencies_not_scored": ["Test-Time Learning", "Long-Range Understanding"],
        "chunk_chars": CHUNK_CHARS,
        "chunk_overlap": CHUNK_OVERLAP,
        "max_chunks_per_context": max_chunks,
        "contexts_total": contexts_total,
        "contexts_run": len(checkpoint),
        "contexts_skipped_for_capacity": contexts_skipped,
        "questions_skipped_for_capacity": skipped_questions,
        "questions": sum(record["questions"] for record in records),
        "answer_bearing_questions": sum(record["answer_bearing_questions"] for record in records),
        "by_competency": by_competency,
        "final_capsule_retrieval": _aggregate([record["final_capsule_retrieval"] for record in records]),
        "strict_source_atomic_retrieval": _aggregate([
            record["strict_source_atomic_retrieval"] for record in records
        ]),
        "official_end_to_end_accuracy_run": False,
        "gold_used_for_ranking": False,
        "data_sha256": {name: _sha256(path) for name, path in selected.items()},
        "runtime_seconds": time.perf_counter() - started,
    }


def _ever_day_groups(dialogue: dict[str, Any]) -> tuple[list[str], list[int], list[str]]:
    texts: list[str] = []
    groups: list[int] = []
    ids: list[str] = []
    for day_number, (date, day) in enumerate(sorted(dialogue["dialogues"].items())):
        for group_name, messages in sorted(day.items()):
            ids.append(f"{date}/{group_name}")
            texts.append("\n".join(
                f'{date} [{group_name}] {message.get("speaker", "")}: '
                f'{message.get("dialogue", message.get("text", ""))}'
                for message in messages
            ))
            groups.append(day_number)
    return texts, groups, ids


def _even_sample(rows: Sequence[Any], limit: int | None) -> list[Any]:
    if limit is None or len(rows) <= limit:
        return list(rows)
    if limit <= 1:
        return [rows[0]]
    positions = [round(number * (len(rows) - 1) / (limit - 1)) for number in range(limit)]
    return [rows[position] for position in positions]


def run_evermembench(
    encoder: FastEmbedEncoder, *, questions_per_dataset: int | None = None,
    checkpoint_name: str | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    root = DATA / "EverMemBench" / "dataset"
    checkpoint_path = RESULTS / (checkpoint_name or (
        f"umd3282_evermembench_capacity_{questions_per_dataset}_checkpoint.json"
        if questions_per_dataset is not None else "umd3282_evermembench_checkpoint.json"
    ))
    checkpoint = _load_checkpoint(checkpoint_path)
    for dialogue_path in sorted(root.glob("*/dialogue_en.json")):
        user_id = dialogue_path.parent.name
        if user_id in checkpoint:
            continue
        qa_path = dialogue_path.parent / f"qa_{user_id}.json"
        dialogue = json.loads(dialogue_path.read_text(encoding="utf-8"))
        qa = json.loads(qa_path.read_text(encoding="utf-8"))
        texts, groups, ids = _ever_day_groups(dialogue)
        normalized_texts = [_normalize(text) for text in texts]
        index = _index(texts, groups, ids, encoder)
        metrics = _metric_pair()
        no_answer_bearing_message = 0
        by_family: dict[str, tuple[CapsuleRecallAccumulator, CapsuleRecallAccumulator]] = {}
        family_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        selected_questions = _even_sample(qa["qars"], questions_per_dataset)
        for row in selected_questions:
            question = str(row["Q"])
            # The answer field is read only after both rankings return.
            channels = index.retrieve_compact_channels(question)
            answer = row["A"]
            gold = _gold_answer_sources(normalized_texts, ids, answer)
            family = str(row.get("id", "unknown")).split("_")[0]
            family_counts[family][0] += 1
            if not gold:
                no_answer_bearing_message += 1
                continue
            family_counts[family][1] += 1
            if family not in by_family:
                by_family[family] = _metric_pair()
            _add_pair(metrics, channels, gold)
            _add_pair(by_family[family], channels, gold)
        checkpoint[user_id] = {
            "questions_total": len(qa["qars"]),
            "questions": len(selected_questions),
            "answer_bearing_questions": len(selected_questions) - no_answer_bearing_message,
            "no_answer_bearing_day_group": no_answer_bearing_message,
            "day_groups": len(texts),
            **_pair_result(metrics),
            "by_family": {
                family: {
                    "questions": family_counts[family][0],
                    "answer_bearing_questions": family_counts[family][1],
                    **_pair_result(values),
                }
                for family, values in by_family.items()
            },
            "dialogue_sha256": _sha256(dialogue_path),
            "qa_sha256": _sha256(qa_path),
        }
        _write_checkpoint(checkpoint_path, checkpoint)
        print(f"EverMemBench: {user_id}", flush=True)
    records = list(checkpoint.values())
    return {
        "benchmark": "EverMemBench-Dynamic",
        "status": (
            "official_english_data_capacity_bounded_answer_bearing_day_group_retrieval_proxy"
            if questions_per_dataset is not None
            else "official_english_data_answer_bearing_day_group_retrieval_proxy"
        ),
        "datasets": sorted(checkpoint),
        "questions_per_dataset": questions_per_dataset,
        "questions_total": sum(record["questions_total"] for record in records),
        "questions": sum(record["questions"] for record in records),
        "answer_bearing_questions": sum(record["answer_bearing_questions"] for record in records),
        "no_answer_bearing_day_group": sum(record["no_answer_bearing_day_group"] for record in records),
        "day_groups": sum(record["day_groups"] for record in records),
        "final_capsule_retrieval": _aggregate([record["final_capsule_retrieval"] for record in records]),
        "strict_source_atomic_retrieval": _aggregate([
            record["strict_source_atomic_retrieval"] for record in records
        ]),
        "official_end_to_end_answer_or_judge_run": False,
        "gold_used_for_ranking": False,
        "runtime_seconds": time.perf_counter() - started,
    }


def _read_arrow(path: Path) -> list[dict[str, Any]]:
    with path.open("rb") as handle:
        try:
            return ipc.open_file(handle).read_all().to_pylist()
        except Exception:
            handle.seek(0)
            return ipc.open_stream(handle).read_all().to_pylist()


def _prompt_question(prompt: str) -> str:
    match = re.search(r"\[Question\]\s*(.*?)\s*\[Answer\]", prompt, flags=re.DOTALL)
    return match.group(1).strip() if match else prompt


def run_memorybench(encoder: FastEmbedEncoder) -> dict[str, Any]:
    started = time.perf_counter()
    root = DATA / "MemoryBench"
    metrics = _metric_pair()
    by_dataset: dict[str, Any] = {}
    for corpus_path in sorted((root / "corpus").glob("DialSim-*.jsonl")):
        dataset_name = corpus_path.stem
        corpus_rows = [json.loads(line) for line in corpus_path.read_text(encoding="utf-8").splitlines()]
        texts = [chunk for row in corpus_rows for chunk in _session_chunks(str(row["text"]))]
        ids = [f"{dataset_name}/session/{number}" for number in range(len(texts))]
        normalized_texts = [_normalize(text) for text in texts]
        index = _index(texts, list(range(len(texts))), ids, encoder)
        test_path = next((root / "dataset" / dataset_name / "test").glob("data-*.arrow"))
        rows = _read_arrow(test_path)
        dataset_metrics = _metric_pair()
        no_answer_bearing_session = 0
        for row in rows:
            question = _prompt_question(str(row["input_prompt"]))
            # Ranking precedes parsing info.golden_answer.
            channels = index.retrieve_compact_channels(question)
            answer = json.loads(row["info"])["golden_answer"]
            gold = _gold_answer_sources(normalized_texts, ids, answer)
            if not gold:
                no_answer_bearing_session += 1
                continue
            _add_pair(metrics, channels, gold)
            _add_pair(dataset_metrics, channels, gold)
        by_dataset[dataset_name] = {
            "questions": len(rows),
            "answer_bearing_questions": len(rows) - no_answer_bearing_session,
            "no_answer_bearing_session": no_answer_bearing_session,
            "sessions": len(texts),
            **_pair_result(dataset_metrics),
            "corpus_sha256": _sha256(corpus_path),
            "test_sha256": _sha256(test_path),
        }
        print(f"MemoryBench: {dataset_name}", flush=True)
    return {
        "benchmark": "MemoryBench",
        "status": "official_dialsim_answer_bearing_session_retrieval_proxy",
        "datasets": sorted(by_dataset),
        "questions": sum(record["questions"] for record in by_dataset.values()),
        "answer_bearing_questions": sum(
            record["answer_bearing_questions"] for record in by_dataset.values()
        ),
        "by_dataset": by_dataset,
        **_pair_result(metrics),
        "official_continual_learning_or_response_quality_run": False,
        "gold_used_for_ranking": False,
        "runtime_seconds": time.perf_counter() - started,
    }


def run_memorybench_temporal(encoder: FastEmbedEncoder) -> dict[str, Any]:
    """Retrieve the dated session explicitly referenced by each DialSim query."""
    started = time.perf_counter()
    root = DATA / "MemoryBench"
    metrics = _metric_pair()
    by_dataset: dict[str, Any] = {}
    date_pattern = re.compile(
        r"(?:January|February|March|April|May|June|July|August|September|"
        r"October|November|December) \d{1,2}, \d{4}"
    )
    for corpus_path in sorted((root / "corpus").glob("DialSim-*.jsonl")):
        dataset_name = corpus_path.stem
        corpus_rows = [json.loads(line) for line in corpus_path.read_text(encoding="utf-8").splitlines()]
        texts = [chunk for row in corpus_rows for chunk in _session_chunks(str(row["text"]))]
        ids = [f"{dataset_name}/session/{number}" for number in range(len(texts))]
        index = _index(texts, list(range(len(texts))), ids, encoder)
        test_path = next((root / "dataset" / dataset_name / "test").glob("data-*.arrow"))
        rows = _read_arrow(test_path)
        dataset_metrics = _metric_pair()
        eligible = 0
        for row in rows:
            question = _prompt_question(str(row["input_prompt"]))
            channels = index.retrieve_compact_channels(question)
            # The target date is public query text, not a hidden answer label.
            dates = date_pattern.findall(question)
            if not dates:
                continue
            target_date = dates[-1]
            gold = {
                source_id for source_id, text in zip(ids, texts)
                if f"[Date: {target_date}," in text
            }
            if not gold:
                continue
            eligible += 1
            _add_pair(metrics, channels, gold)
            _add_pair(dataset_metrics, channels, gold)
        by_dataset[dataset_name] = {
            "questions": len(rows),
            "dated_questions_with_matching_sessions": eligible,
            "sessions": len(texts),
            **_pair_result(dataset_metrics),
            "corpus_sha256": _sha256(corpus_path),
            "test_sha256": _sha256(test_path),
        }
        print(f"MemoryBench temporal: {dataset_name}", flush=True)
    return {
        "benchmark": "MemoryBench",
        "status": "official_dialsim_query_date_session_retrieval_proxy",
        "datasets": sorted(by_dataset),
        "questions": sum(record["questions"] for record in by_dataset.values()),
        "dated_questions_with_matching_sessions": sum(
            record["dated_questions_with_matching_sessions"] for record in by_dataset.values()
        ),
        "by_dataset": by_dataset,
        **_pair_result(metrics),
        "official_continual_learning_or_response_quality_run": False,
        "hidden_gold_answer_used": False,
        "runtime_seconds": time.perf_counter() - started,
    }


def _dependency_components(value: Any) -> set[str]:
    values = _answer_strings(value)
    return {
        item for item in values
        if len(item) >= 4 and item not in {"none", "true", "false", "days", "day"}
    }


def run_memoryarena(encoder: FastEmbedEncoder) -> dict[str, Any]:
    """Sequential prior-subtask reuse proxy; not an environment success score."""
    started = time.perf_counter()
    root = DATA / "MemoryArena"
    metrics = _metric_pair()
    by_environment: dict[str, Any] = {}
    for path in sorted(root.glob("*/data.jsonl")):
        environment = path.parent.name
        environment_metrics = _metric_pair()
        total_queries = eligible_queries = 0
        for raw in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(raw)
            questions = row["questions"]
            answers = row["answers"]
            for position in range(1, min(len(questions), len(answers))):
                prior_texts = [
                    f"Subtask: {questions[number]}\nSuccessful observation: "
                    f"{json.dumps(answers[number], ensure_ascii=False, sort_keys=True)}"
                    for number in range(position)
                ]
                ids = [f'{environment}/{row["id"]}/{number}' for number in range(position)]
                index = _index(prior_texts, list(range(position)), ids, encoder)
                channels = index.retrieve_compact_channels(str(questions[position]))
                # Current answer is inspected only after retrieval.  Past answers
                # are legitimate observations already present in agent memory.
                current_components = _dependency_components(answers[position])
                gold = {
                    ids[number]
                    for number in range(position)
                    if _dependency_components(answers[number]) & current_components
                }
                total_queries += 1
                if not gold:
                    continue
                eligible_queries += 1
                _add_pair(metrics, channels, gold)
                _add_pair(environment_metrics, channels, gold)
        by_environment[environment] = {
            "sequential_queries": total_queries,
            "queries_with_reused_answer_components": eligible_queries,
            **_pair_result(environment_metrics),
            "data_sha256": _sha256(path),
        }
        print(f"MemoryArena: {environment}", flush=True)
    return {
        "benchmark": "MemoryArena",
        "status": "official_data_prior_subtask_reuse_retrieval_proxy",
        "environments": sorted(by_environment),
        "sequential_queries": sum(record["sequential_queries"] for record in by_environment.values()),
        "queries_with_reused_answer_components": sum(
            record["queries_with_reused_answer_components"] for record in by_environment.values()
        ),
        "by_environment": by_environment,
        **_pair_result(metrics),
        "official_agent_environment_success_run": False,
        "gold_used_for_ranking": False,
        "runtime_seconds": time.perf_counter() - started,
    }


def _lmev2_trajectory_text(trajectory: dict[str, Any]) -> str:
    parts = [
        f'Goal: {trajectory.get("goal", "")}',
        f'Outcome: {trajectory.get("outcome", "")}',
        f'Start URL: {trajectory.get("start_url", "")}',
    ]
    for state in trajectory.get("states", []):
        parts.extend((
            f'State {state.get("state_index", "")}',
            f'URL: {state.get("url", "")}',
            f'Action: {state.get("action", "")}',
            f'Thought: {state.get("thought", "")}',
            f'Accessibility tree:\n{state.get("accessibility_tree", "")}',
        ))
    return "\n".join(parts)


def _lmev2_trajectory_note(trajectory: dict[str, Any]) -> str:
    """Gold-independent compact experience note used by the capacity tier."""
    parts = [
        f'Goal: {trajectory.get("goal", "")}',
        f'Outcome: {trajectory.get("outcome", "")}',
        f'Start URL: {trajectory.get("start_url", "")}',
    ]
    for state in trajectory.get("states", []):
        parts.extend((
            f'URL: {state.get("url", "")}',
            f'Action: {state.get("action", "")}',
            f'Thought: {state.get("thought", "")}',
        ))
    return "\n".join(parts)


def _lmev2_answer_components(answer: Any) -> list[str]:
    value = str(answer)
    pieces = [_normalize(piece) for piece in re.split(r"[;,]", value)]
    return [piece for piece in pieces if len(piece) >= 2]


def _lmev2_gold_sources(
    normalized_texts: Sequence[str], source_ids: Sequence[str], answer: Any,
) -> set[str]:
    pieces = _lmev2_answer_components(answer)
    if not pieces:
        return set()
    return {
        source_id
        for source_id, text in zip(source_ids, normalized_texts)
        if all(piece in text for piece in pieces)
    }


def run_longmemeval_v2(
    encoder: FastEmbedEncoder, *, questions_per_haystack: int | None = None,
) -> dict[str, Any]:
    """Small-tier answer-bearing trajectory retrieval on official data."""
    started = time.perf_counter()
    root = DATA / "LongMemEval-V2"
    questions = [
        json.loads(line)
        for line in (root / "questions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    question_by_id = {str(row["id"]): row for row in questions}
    haystacks = json.loads((root / "haystacks" / "lme_v2_small.json").read_text(encoding="utf-8"))
    required_ids = {str(value) for values in haystacks.values() for value in values}
    trajectories: dict[str, dict[str, Any]] = {}
    with (root / "trajectories.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            trajectory_id = str(row["id"])
            if trajectory_id in required_ids:
                trajectories[trajectory_id] = row
    missing = required_ids - set(trajectories)
    if missing:
        raise RuntimeError(f"LongMemEval-V2 is missing {len(missing)} small-tier trajectories")
    unique_haystacks = {tuple(str(value) for value in values) for values in haystacks.values()}
    metrics = _metric_pair()
    by_domain: dict[str, tuple[CapsuleRecallAccumulator, CapsuleRecallAccumulator]] = {}
    by_type: dict[str, tuple[CapsuleRecallAccumulator, CapsuleRecallAccumulator]] = {}
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    type_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    trajectory_chars = 0
    for haystack_number, ids_tuple in enumerate(sorted(unique_haystacks)):
        ids = list(ids_tuple)
        full_texts = [_lmev2_trajectory_text(trajectories[source_id]) for source_id in ids]
        normalized_texts = [_normalize(text) for text in full_texts]
        trajectory_chars += sum(len(text) for text in full_texts)
        notes = [_lmev2_trajectory_note(trajectories[source_id]) for source_id in ids]
        index = _index(notes, list(range(len(notes))), ids, encoder)
        all_question_ids = [
            question_id for question_id, values in haystacks.items()
            if tuple(str(value) for value in values) == ids_tuple
        ]
        question_ids = _even_sample(all_question_ids, questions_per_haystack)
        for question_id in question_ids:
            row = question_by_id[question_id]
            question = str(row["question"])
            # Frozen trajectory ranking precedes reference-answer inspection.
            channels = index.retrieve_compact_channels(question)
            answer = row["answer"]
            gold = _lmev2_gold_sources(normalized_texts, ids, answer)
            domain = str(row.get("domain", "unknown"))
            question_type = str(row.get("question_type", "unknown"))
            counts[domain][0] += 1
            type_counts[question_type][0] += 1
            if not gold:
                continue
            counts[domain][1] += 1
            type_counts[question_type][1] += 1
            if domain not in by_domain:
                by_domain[domain] = _metric_pair()
            if question_type not in by_type:
                by_type[question_type] = _metric_pair()
            _add_pair(metrics, channels, gold)
            _add_pair(by_domain[domain], channels, gold)
            _add_pair(by_type[question_type], channels, gold)
        print(
            f"LongMemEval-V2 small haystack: {haystack_number + 1}/{len(unique_haystacks)}",
            flush=True,
        )
    return {
        "benchmark": "LongMemEval-V2",
        "tier": "small",
        "status": (
            "official_data_capacity_bounded_answer_bearing_trajectory_retrieval_proxy"
            if questions_per_haystack is not None
            else "official_data_answer_bearing_trajectory_retrieval_proxy"
        ),
        "questions_total": len(questions),
        "questions": sum(value[0] for value in counts.values()),
        "questions_per_haystack": questions_per_haystack,
        "answer_bearing_questions": sum(value[1] for value in counts.values()),
        "trajectories": len(required_ids),
        "trajectory_characters": trajectory_chars,
        "by_domain": {
            domain: {
                "questions": counts[domain][0],
                "answer_bearing_questions": counts[domain][1],
                **_pair_result(values),
            }
            for domain, values in by_domain.items()
        },
        "by_question_type": {
            question_type: {
                "questions": type_counts[question_type][0],
                "answer_bearing_questions": type_counts[question_type][1],
                **_pair_result(values),
            }
            for question_type, values in by_type.items()
        },
        **_pair_result(metrics),
        "official_reader_accuracy_or_lafs_run": False,
        "gold_used_for_ranking": False,
        "data_sha256": {
            "questions": _sha256(root / "questions.jsonl"),
            "small_haystacks": _sha256(root / "haystacks" / "lme_v2_small.json"),
            "trajectories": _sha256(root / "trajectories.jsonl"),
        },
        "runtime_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark",
        choices=(
            "memoryagentbench", "evermembench", "memorybench", "memoryarena",
            "memorybench_temporal", "longmemeval_v2",
        ),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--full", action="store_true",
        help="Run every official context/question instead of the capacity tier.",
    )
    parser.add_argument(
        "--physics-v331", action="store_true",
        help="Enable the gold-blind UMD 3.31 evidence-closure field.",
    )
    parser.add_argument(
        "--physics-v332", action="store_true",
        help="Enable the gold-blind UMD 3.32 relation-superposition field.",
    )
    parser.add_argument(
        "--physics-v333", action="store_true",
        help="Enable the gold-blind UMD 3.33 document-star hierarchy.",
    )
    parser.add_argument(
        "--physics-v334", action="store_true",
        help="Enable UMD 3.34 query fission and episodic semantic periapsis.",
    )
    parser.add_argument(
        "--physics-v335", action="store_true",
        help="Enable UMD 3.35 provenance-safe background ghost matter.",
    )
    parser.add_argument(
        "--physics-v336", action="store_true",
        help="Enable UMD 3.36 bounded ghost constellations.",
    )
    parser.add_argument(
        "--run-id",
        help=(
            "Use isolated checkpoint files prefixed by this identifier. "
            "This prevents a rerun from silently reusing historical rankings."
        ),
    )
    args = parser.parse_args()
    if args.run_id and any(
        not (character.isalnum() or character in "-_")
        for character in args.run_id
    ):
        parser.error("--run-id may contain only letters, digits, '-' and '_'")
    encoder = FastEmbedEncoder(
        cache_dir=MODEL_CACHE, batch_size=128, cache_size=32768, threads=16,
    )
    runners = {
        "memoryagentbench": run_memoryagentbench,
        "evermembench": run_evermembench,
        "memorybench": run_memorybench,
        "memorybench_temporal": run_memorybench_temporal,
        "memoryarena": run_memoryarena,
        "longmemeval_v2": run_longmemeval_v2,
    }
    if args.benchmark == "memoryagentbench":
        result = run_memoryagentbench(
            encoder, max_chunks=None if args.full else 600,
            physics_v331=args.physics_v331 or args.physics_v332 or args.physics_v333 or args.physics_v334 or args.physics_v335 or args.physics_v336,
            physics_v332=args.physics_v332 or args.physics_v333 or args.physics_v334 or args.physics_v335 or args.physics_v336,
            physics_v333=args.physics_v333 or args.physics_v334 or args.physics_v335 or args.physics_v336,
            physics_v334=args.physics_v334 or args.physics_v335 or args.physics_v336,
            physics_v335=args.physics_v335 or args.physics_v336,
            physics_v336=args.physics_v336,
            checkpoint_name=(
                f"{args.run_id}_memoryagentbench_checkpoint.json"
                if args.run_id else None
            ),
        )
    elif args.benchmark == "evermembench":
        result = run_evermembench(
            encoder, questions_per_dataset=None if args.full else 50,
            checkpoint_name=(
                f"{args.run_id}_evermembench_checkpoint.json"
                if args.run_id else None
            ),
        )
    elif args.benchmark == "longmemeval_v2":
        result = run_longmemeval_v2(
            encoder, questions_per_haystack=None if args.full else 50,
        )
    else:
        result = runners[args.benchmark](encoder)
    payload = {
        args.benchmark: result,
        "metadata": {
            "adapter": (
                "UMD 3.36 bounded ghost constellations / "
                "UMD 3.35 provenance-safe background ghost matter / "
                "UMD 3.34 query fission and episodic semantic periapsis / "
                "UMD 3.33.1 binary document stars and absorbing relation boundary / "
                "UMD 3.32.1 relation superposition and version shadows / "
                "UMD 3.31 evidence closure / UMD 3.30 slingshot / "
                "UMD 3.28.2 conserved control laws"
                if args.physics_v336
                else
                "UMD 3.35 provenance-safe background ghost matter / "
                "UMD 3.34 query fission and episodic semantic periapsis / "
                "UMD 3.33.1 binary document stars and absorbing relation boundary / "
                "UMD 3.32.1 relation superposition and version shadows / "
                "UMD 3.31 evidence closure / UMD 3.30 slingshot / "
                "UMD 3.28.2 conserved control laws"
                if args.physics_v335
                else
                "UMD 3.34 query fission and episodic semantic periapsis / "
                "UMD 3.33.1 binary document stars and absorbing relation boundary / "
                "UMD 3.32.1 relation superposition and version shadows / "
                "UMD 3.31 evidence closure / UMD 3.30 slingshot / "
                "UMD 3.28.2 conserved control laws"
                if args.physics_v334
                else
                "UMD 3.33.1 binary document stars and absorbing relation boundary / "
                "UMD 3.32.1 relation superposition and version shadows / "
                "UMD 3.31 evidence closure / UMD 3.30 slingshot / "
                "UMD 3.28.2 conserved control laws"
                if args.physics_v333
                else
                "UMD 3.32.1 relation superposition and version shadows / "
                "UMD 3.31 evidence closure / "
                "UMD 3.30 slingshot / UMD 3.28.2 conserved control laws"
                if args.physics_v332
                else
                "UMD 3.31 evidence closure / UMD 3.30 slingshot / "
                "UMD 3.28.2 conserved control laws"
                if args.physics_v331
                else "UMD 3.30 slingshot / UMD 3.28.2 conserved control laws"
            ),
            "full_run": args.full,
            "run_id": args.run_id,
            "gold_read_after_retrieval": True,
            "gold_used_for_ranking": False,
            "parameters_frozen": True,
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
