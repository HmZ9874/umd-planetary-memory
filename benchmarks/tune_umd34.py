"""Tune UMD 3.4 fusion weights on deterministic development splits.

The gold evidence is used only to choose weights on the development split.
The holdout split is reported once with the chosen configuration. Ranking never
reads answers, evidence ids, has_answer labels, or question_type.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import itertools
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from public_benchmarks import (
    BM25,
    DATA,
    RESULTS,
    _locomo_sessions,
    _turn_text,
    char_tokenize,
    ranking,
    reciprocal_ranks,
)


ASSISTANT_PATTERNS = re.compile(
    r"\b(you|your)\b.*\b(recommend|suggest|tell|told|provide|gave|give|answer|said|mention|advise)",
    re.IGNORECASE,
)
USER_PATTERNS = re.compile(r"\b(i|me|my|mine|we|our)\b", re.IGNORECASE)
RECENCY_PATTERNS = re.compile(r"\b(latest|current|currently|now|recent|recently|still|most recent|newest)\b", re.IGNORECASE)


@dataclass
class Record:
    key: str
    split: str
    gold: set[int]
    features: dict[str, np.ndarray]
    session_of: np.ndarray | None = None
    protected_order: np.ndarray | None = None


def split_for(key: str, dev_percent: int = 20) -> str:
    bucket = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) % 100
    return "dev" if bucket < dev_percent else "holdout"


def metrics(records: list[Record], config: dict[str, float], locomo: bool) -> dict[str, float]:
    count = 0
    hit1 = hit5 = hit10 = 0
    reciprocal = 0.0
    for record in records:
        score = record.features["primary"].copy()
        for name, weight in config.items():
            if name in {"neighbor", "protect_top10", "protect_top1", "keep_protected"}:
                continue
            score += weight * record.features[name]
        if locomo and config.get("neighbor", 0.0):
            neighbor = np.zeros_like(score)
            session_of = record.session_of
            for offset in (-1, 1):
                source = np.arange(len(score)) + offset
                valid = (source >= 0) & (source < len(score))
                valid &= session_of[np.clip(source, 0, len(score) - 1)] == session_of
                neighbor[valid] = np.maximum(neighbor[valid], score[source[valid]])
            score += config["neighbor"] * neighbor
        order = np.argsort(-score, kind="stable")
        if config.get("keep_protected", 0.0) and record.protected_order is not None:
            order = record.protected_order
        elif config.get("protect_top10", 0.0) and record.protected_order is not None:
            protected = record.protected_order[:10]
            protected_set = set(map(int, protected))
            if config.get("protect_top1", 0.0):
                first = protected[:1]
                remainder = protected[1:]
                head = np.concatenate([first, remainder[np.argsort(-score[remainder], kind="stable")]])
            else:
                head = protected[np.argsort(-score[protected], kind="stable")]
            tail = np.asarray([item for item in record.protected_order if int(item) not in protected_set])
            order = np.concatenate([head, tail])
        positions = [rank + 1 for rank, item in enumerate(order) if int(item) in record.gold]
        if not positions:
            continue
        count += 1
        first = min(positions)
        reciprocal += 1.0 / first
        hit1 += first <= 1
        hit5 += first <= 5
        hit10 += first <= 10
    return {
        "queries": count,
        "hit@1": hit1 / max(1, count),
        "hit@5": hit5 / max(1, count),
        "hit@10": hit10 / max(1, count),
        "mrr": reciprocal / max(1, count),
    }


def objective(metric: dict[str, float]) -> float:
    return 0.30 * metric["mrr"] + 0.20 * metric["hit@1"] + 0.25 * metric["hit@5"] + 0.25 * metric["hit@10"]


def locomo_records(path: Path) -> list[Record]:
    dataset = json.loads(path.read_text(encoding="utf-8"))
    records: list[Record] = []
    for sample in dataset:
        _, sessions, dates = _locomo_sessions(sample)
        turns = [turn for session in sessions for turn in session]
        ids = [turn["dia_id"] for turn in turns]
        id_to_index = {item: i for i, item in enumerate(ids)}
        texts = [f'{turn.get("speaker", "")}: {turn.get("text", "")} {turn.get("blip_caption", "")}' for turn in turns]
        session_of: list[int] = []
        session_indices: list[list[int]] = []
        cursor = 0
        for session_number, session in enumerate(sessions):
            indices = list(range(cursor, cursor + len(session)))
            session_indices.append(indices)
            session_of.extend([session_number] * len(session))
            cursor += len(session)
        session_texts = [f"{dates[i]} " + " ".join(texts[j] for j in indices) for i, indices in enumerate(session_indices)]
        word_index = BM25(texts)
        char_index = BM25(texts, tokenizer=char_tokenize)
        session_index = BM25(session_texts)
        speaker_names = {str(sample["conversation"].get("speaker_a", "")).lower(), str(sample["conversation"].get("speaker_b", "")).lower()}

        for qa_number, qa in enumerate(sample["qa"]):
            gold = {id_to_index[item] for item in qa.get("evidence", []) if item in id_to_index}
            if not gold:
                continue
            query = qa["question"]
            word_order = ranking(word_index.scores(query))
            char_order = ranking(char_index.scores(query))
            session_order = ranking(session_index.scores(query))
            mentioned = {name for name in speaker_names if name and re.search(rf"\b{re.escape(name)}\b", query.lower())}
            speaker = np.array([
                1.0 / 61.0 if str(turn.get("speaker", "")).lower() in mentioned else 0.0
                for turn in turns
            ])
            session_rr = reciprocal_ranks(session_order)
            primary_rr = np.asarray(reciprocal_ranks(word_order))
            propagated_session = np.asarray([session_rr[session_of[i]] for i in range(len(turns))])
            v33_score = primary_rr + 0.72 * propagated_session
            v33_seeds = np.argsort(-v33_score, kind="stable")
            v33_order: list[int] = []
            seen: set[int] = set()
            session_of_array = np.asarray(session_of)
            for seed in v33_seeds:
                for candidate in (int(seed), int(seed) - 1, int(seed) + 1):
                    if (
                        0 <= candidate < len(turns)
                        and candidate not in seen
                        and session_of_array[candidate] == session_of_array[int(seed)]
                    ):
                        seen.add(candidate)
                        v33_order.append(candidate)
            records.append(Record(
                key=f'{sample["sample_id"]}:{qa_number}',
                split=split_for(str(sample["sample_id"]), 40),
                gold=gold,
                features={
                    "primary": primary_rr,
                    "session": propagated_session,
                    "char": np.asarray(reciprocal_ranks(char_order)),
                    "speaker": speaker,
                },
                session_of=session_of_array,
                protected_order=np.asarray(v33_order),
            ))
    return records


def longmemeval_records(path: Path) -> list[Record]:
    dataset = json.loads(path.read_text(encoding="utf-8"))
    records: list[Record] = []
    for item_number, item in enumerate(dataset, 1):
        session_ids = item["haystack_session_ids"]
        id_to_index = {item_id: i for i, item_id in enumerate(session_ids)}
        sessions = item["haystack_sessions"]
        dates = item["haystack_dates"]
        session_texts = [f"{dates[i]} " + " ".join(_turn_text(turn) for turn in session) for i, session in enumerate(sessions)]
        turn_texts: list[str] = []
        turn_roles: list[str] = []
        turn_to_session: list[int] = []
        for session_number, session in enumerate(sessions):
            for turn in session:
                turn_texts.append(_turn_text(turn))
                turn_roles.append(turn.get("role", ""))
                turn_to_session.append(session_number)
        query = item["question"]
        session_index = BM25(session_texts)
        turn_index = BM25(turn_texts)
        aggregate_order = ranking(session_index.scores(query))
        turn_scores = turn_index.scores(query)
        best_all = [0.0] * len(sessions)
        best_user = [0.0] * len(sessions)
        best_assistant = [0.0] * len(sessions)
        for turn_number, score in enumerate(turn_scores):
            session_number = turn_to_session[turn_number]
            best_all[session_number] = max(best_all[session_number], score)
            if turn_roles[turn_number] == "user":
                best_user[session_number] = max(best_user[session_number], score)
            elif turn_roles[turn_number] == "assistant":
                best_assistant[session_number] = max(best_assistant[session_number], score)
        moon_order = ranking(best_all)
        if ASSISTANT_PATTERNS.search(query):
            role_order = ranking(best_assistant)
        elif USER_PATTERNS.search(query):
            role_order = ranking(best_user)
        else:
            role_order = moon_order
        if RECENCY_PATTERNS.search(query):
            recent_order = sorted(range(len(dates)), key=lambda i: (dates[i], i), reverse=True)
            recent = np.asarray(reciprocal_ranks(recent_order))
        else:
            recent = np.zeros(len(sessions))
        gold = {id_to_index[item_id] for item_id in item.get("answer_session_ids", []) if item_id in id_to_index}
        primary_rr = np.asarray(reciprocal_ranks(aggregate_order))
        moon_rr = np.asarray(reciprocal_ranks(moon_order))
        baseline_order = np.argsort(-(primary_rr + 0.82 * moon_rr), kind="stable")
        records.append(Record(
            key=item["question_id"],
            split=split_for(item["question_id"]),
            gold=gold,
            features={
                "primary": primary_rr,
                "moon": moon_rr,
                "role": np.asarray(reciprocal_ranks(role_order)),
                "recent": recent,
            },
            protected_order=baseline_order,
        ))
        if item_number % 25 == 0:
            print(f"feature extraction: {item_number}/{len(dataset)}", flush=True)
    return records


def tune(
    records: list[Record],
    configs: list[dict[str, float]],
    locomo: bool,
    baseline_config: dict[str, float],
) -> dict:
    dev = [record for record in records if record.split == "dev"]
    holdout = [record for record in records if record.split == "holdout"]
    candidates = []
    for config in configs:
        metric = metrics(dev, config, locomo)
        candidates.append((objective(metric), config, metric))
    candidates.sort(key=lambda item: item[0], reverse=True)
    _, selected, dev_metric = candidates[0]
    return {
        "baseline": {
            "development": metrics(dev, baseline_config, locomo),
            "holdout": metrics(holdout, baseline_config, locomo),
            "full": metrics(records, baseline_config, locomo),
        },
        "selected": selected,
        "development": dev_metric,
        "holdout": metrics(holdout, selected, locomo),
        "full": metrics(records, selected, locomo),
        "top_development_candidates": [
            {"config": config, "objective": score, "metrics": metric}
            for score, config, metric in candidates[:5]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=("both", "locomo", "longmemeval"), default="both")
    args = parser.parse_args()
    output: dict[str, dict] = {"methodology": {"gold_used_for_ranking": False, "split": "sha256 id", "dev_only_parameter_selection": True}}
    started = time.perf_counter()
    if args.benchmark in {"both", "locomo"}:
        records = locomo_records(DATA / "locomo10.json")
        configs = [
            {"session": session, "char": char, "speaker": speaker, "protect_top10": 1.0}
            for session, char, speaker in itertools.product(
                (0.35, 0.55, 0.75, 0.95),
                (0.0, 0.12, 0.25),
                (0.0, 0.04, 0.08, 0.12),
            )
        ]
        output["locomo"] = tune(records, configs, True, {"keep_protected": 1.0})
    if args.benchmark in {"both", "longmemeval"}:
        records = longmemeval_records(DATA / "longmemeval_s_cleaned.json")
        configs = [
            {"moon": moon, "role": role, "recent": recent, "protect_top10": 1.0, "protect_top1": 1.0}
            for moon, role, recent in itertools.product(
                (0.20, 0.40, 0.60, 0.80, 1.00),
                (0.0, 0.15, 0.30, 0.50),
                (0.0, 0.10, 0.25),
            )
        ]
        output["longmemeval"] = tune(records, configs, False, {"moon": 0.82, "role": 0.0, "recent": 0.0})
    output["runtime_seconds"] = time.perf_counter() - started
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / "umd34_tuning.json"
    if args.benchmark != "both" and path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        for name in ("locomo", "longmemeval"):
            if name not in output and name in previous:
                output[name] = previous[name]
    path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
