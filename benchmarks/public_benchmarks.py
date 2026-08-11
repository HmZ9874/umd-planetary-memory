"""Reproducible retrieval evaluation on LoCoMo and LongMemEval.

This is a raw-text adapter for the UMD 3.3 hierarchy, not an LLM answerer.
It compares a flat BM25 baseline with a hierarchical rank-fusion retriever:

* star: one long-lived user/conversation;
* planet: a dated conversation session;
* moon/asteroid: an individual turn;
* relation: neighboring turns around a retrieved turn.

No answer text, evidence labels, or question-type labels are used for ranking.
Gold evidence is read only after retrieval to calculate recall and MRR.
"""

from __future__ import annotations

import argparse
import bisect
import collections
import ctypes
from ctypes import wintypes
import hashlib
import json
import math
import re
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"


_LOCOMO_EVIDENCE_ID_RE = re.compile(r"D\d+:\d+")


def normalize_locomo_evidence_ids(value: Any) -> set[str]:
    """Return canonical LoCoMo dialogue IDs from heterogeneous gold labels.

    Some public category-3 rows store several IDs in one whitespace-separated
    string (for example ``"D9:1 D4:4 D4:6"``).  Treating that scalar as one
    source silently creates an impossible positive.  The recursive handling
    also preserves compatibility with ordinary lists and nested exports.
    """
    output: set[str] = set()
    if isinstance(value, dict):
        for child in value.values():
            output.update(normalize_locomo_evidence_ids(child))
    elif isinstance(value, (list, tuple, set)):
        for child in value:
            output.update(normalize_locomo_evidence_ids(child))
    elif value is not None:
        text = str(value).strip()
        matches = _LOCOMO_EVIDENCE_ID_RE.findall(text)
        if matches:
            output.update(matches)
        elif text:
            output.add(text)
    return output
WORD_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by",
    "did", "do", "does", "for", "from", "had", "has", "have", "he",
    "her", "hers", "him", "his", "how", "i", "in", "is", "it", "its",
    "me", "my", "of", "on", "or", "our", "she", "that", "the", "their",
    "them", "they", "this", "to", "was", "we", "were", "what", "when",
    "where", "which", "who", "why", "will", "with", "would", "you", "your",
}


def _stem(word: str) -> str:
    """A deliberately small, deterministic English stemmer."""
    for suffix in ("ingly", "edly", "ation", "ments", "ment", "ness", "ing", "ied", "ies", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            if suffix in {"ied", "ies"}:
                return word[: -len(suffix)] + "y"
            return word[: -len(suffix)]
    return word


def tokenize(text: str) -> list[str]:
    words = [_stem(w.lower()) for w in WORD_RE.findall(text)]
    content = [w for w in words if w not in STOPWORDS and len(w) > 1]
    # Bigrams add phrase precision without a learned embedding dependency.
    return content + [f"{a}_{b}" for a, b in zip(content, content[1:])]


def char_tokenize(text: str) -> list[str]:
    """Within-word character features for typos and light morphology."""
    features: list[str] = []
    for word in WORD_RE.findall(text.lower()):
        if word in STOPWORDS or len(word) < 4:
            continue
        padded = f"^{word}$"
        features.extend(f"c3:{padded[i:i + 3]}" for i in range(len(padded) - 2))
    return features


class BM25:
    def __init__(
        self,
        documents: Sequence[str],
        k1: float = 1.35,
        b: float = 0.72,
        tokenizer: Callable[[str], list[str]] = tokenize,
    ):
        self.k1 = k1
        self.b = b
        self.n = len(documents)
        self.lengths: list[int] = []
        self.postings: dict[str, list[tuple[int, int]]] = collections.defaultdict(list)
        for doc_id, document in enumerate(documents):
            counts = collections.Counter(tokenizer(document))
            self.lengths.append(sum(counts.values()))
            for term, count in counts.items():
                self.postings[term].append((doc_id, count))
        self.length_prefix = [0]
        for length in self.lengths:
            self.length_prefix.append(self.length_prefix[-1] + length)
        self.avg_length = sum(self.lengths) / max(1, self.n)
        self.tokenizer = tokenizer

    def scores(self, query: str) -> list[float]:
        scores = [0.0] * self.n
        query_counts = collections.Counter(self.tokenizer(query))
        for term, qtf in query_counts.items():
            posting = self.postings.get(term)
            if not posting:
                continue
            df = len(posting)
            idf = math.log(1.0 + (self.n - df + 0.5) / (df + 0.5))
            for doc_id, tf in posting:
                norm = 1.0 - self.b + self.b * self.lengths[doc_id] / max(1e-9, self.avg_length)
                scores[doc_id] += qtf * idf * tf * (self.k1 + 1.0) / (tf + self.k1 * norm)
        return scores

    def scores_prefix(self, query: str, size: int) -> list[float]:
        """Score an immutable document prefix without rebuilding postings.

        IDF and average document length are recomputed for the visible prefix,
        so future documents neither contribute terms nor change statistics.
        """
        visible = min(max(0, int(size)), self.n)
        if visible <= 0:
            return []
        scores = [0.0] * visible
        avg_length = self.length_prefix[visible] / visible
        query_counts = collections.Counter(self.tokenizer(query))
        for term, qtf in query_counts.items():
            full_posting = self.postings.get(term)
            if not full_posting:
                continue
            # Postings are appended in document order. Binary search prevents
            # every historical-prefix query from rescanning future postings.
            stop = bisect.bisect_left(full_posting, (visible, -1))
            if stop <= 0:
                continue
            df = stop
            idf = math.log(1.0 + (visible - df + 0.5) / (df + 0.5))
            for doc_id, tf in full_posting[:stop]:
                norm = 1.0 - self.b + self.b * self.lengths[doc_id] / max(1e-9, avg_length)
                scores[doc_id] += qtf * idf * tf * (self.k1 + 1.0) / (tf + self.k1 * norm)
        return scores


def ranking(scores: Sequence[float]) -> list[int]:
    return sorted(range(len(scores)), key=lambda i: (-scores[i], i))


def reciprocal_ranks(order: Sequence[int], constant: float = 60.0) -> list[float]:
    result = [0.0] * len(order)
    for rank, item in enumerate(order, 1):
        result[item] = 1.0 / (constant + rank)
    return result


def unit_scores(values: Sequence[float]) -> list[float]:
    maximum = max(values, default=0.0)
    return [value / maximum if maximum > 0 else 0.0 for value in values]


def planetary_query_weights(query: str) -> tuple[float, float]:
    """Return temporal-phase and exact-identifier resonance multipliers."""
    words = set(tokenize(query))
    temporal = 1.8 if words & {"when", "before", "after", "current", "latest", "previous", "history"} else 1.0
    exact = 1.7 if re.search(r"[A-Za-z]+[-_.:/]\d|\b\d+(?:\.\d+)+\b", query) else 1.0
    return temporal, exact


def normalized(text: object) -> str:
    return " ".join(WORD_RE.findall(str(text).lower()))


def peak_working_set_mib() -> float | None:
    """Return process peak RAM on Windows without allocation tracing overhead."""
    if not hasattr(ctypes, "windll"):
        return None

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(ProcessMemoryCounters),
        wintypes.DWORD,
    ]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    process = kernel32.GetCurrentProcess()
    ok = psapi.GetProcessMemoryInfo(process, ctypes.byref(counters), counters.cb)
    return counters.PeakWorkingSetSize / (1024 * 1024) if ok else None


@dataclass
class RecallAccumulator:
    ks: tuple[int, ...]
    queries: int = 0
    gold_items: int = 0
    reciprocal_rank_sum: float = 0.0
    any_hits: dict[int, int] | None = None
    full_hits: dict[int, int] | None = None
    item_hits: dict[int, int] | None = None

    def __post_init__(self) -> None:
        self.any_hits = {k: 0 for k in self.ks}
        self.full_hits = {k: 0 for k in self.ks}
        self.item_hits = {k: 0 for k in self.ks}

    def add(self, ranked_ids: Sequence[str], gold_ids: Iterable[str]) -> None:
        gold = set(gold_ids)
        if not gold:
            return
        self.queries += 1
        self.gold_items += len(gold)
        positions = [i + 1 for i, item in enumerate(ranked_ids) if item in gold]
        if positions:
            self.reciprocal_rank_sum += 1.0 / min(positions)
        for k in self.ks:
            found = set(ranked_ids[:k]) & gold
            self.any_hits[k] += bool(found)
            self.full_hits[k] += found == gold
            self.item_hits[k] += len(found)

    def result(self) -> dict:
        return {
            "evaluated_queries": self.queries,
            "gold_evidence_items": self.gold_items,
            "mrr": self.reciprocal_rank_sum / max(1, self.queries),
            "any_evidence_recall": {str(k): self.any_hits[k] / max(1, self.queries) for k in self.ks},
            "full_evidence_recall": {str(k): self.full_hits[k] / max(1, self.queries) for k in self.ks},
            "micro_evidence_recall": {str(k): self.item_hits[k] / max(1, self.gold_items) for k in self.ks},
        }


def _locomo_sessions(sample: dict) -> tuple[list[str], list[list[dict]], list[str]]:
    conversation = sample["conversation"]
    session_numbers = sorted(
        int(key.split("_")[1])
        for key, value in conversation.items()
        if re.fullmatch(r"session_\d+", key) and isinstance(value, list)
    )
    session_ids = [f"D{number}" for number in session_numbers]
    sessions = [conversation[f"session_{number}"] for number in session_numbers]
    dates = [conversation.get(f"session_{number}_date_time", "") for number in session_numbers]
    return session_ids, sessions, dates


def run_locomo(path: Path) -> dict:
    started = time.perf_counter()
    dataset = json.loads(path.read_text(encoding="utf-8"))
    ks = (1, 5, 10, 20)
    flat = RecallAccumulator(ks)
    umd = RecallAccumulator(ks)
    umd34 = RecallAccumulator(ks)
    umd37 = RecallAccumulator(ks)
    total_qa = 0
    no_evidence = 0
    context_chars = {"flat@10": [], "umd@10": []}

    for sample in dataset:
        session_ids, sessions, dates = _locomo_sessions(sample)
        turns = [turn for session in sessions for turn in session]
        turn_ids = [turn["dia_id"] for turn in turns]
        turn_texts = [f'{turn.get("speaker", "")}: {turn.get("text", "")} {turn.get("blip_caption", "")}' for turn in turns]
        turn_to_session: list[int] = []
        session_turn_indices: list[list[int]] = []
        cursor = 0
        for session_index, session in enumerate(sessions):
            indices = list(range(cursor, cursor + len(session)))
            session_turn_indices.append(indices)
            turn_to_session.extend([session_index] * len(session))
            cursor += len(session)
        session_texts = [
            f"{dates[i]} " + " ".join(turn_texts[j] for j in session_turn_indices[i])
            for i in range(len(sessions))
        ]
        turn_index = BM25(turn_texts)
        char_index = BM25(turn_texts, tokenizer=char_tokenize)
        session_index = BM25(session_texts)

        for qa in sample["qa"]:
            total_qa += 1
            gold = normalize_locomo_evidence_ids(qa.get("evidence"))
            if not gold:
                no_evidence += 1
                continue
            query = qa["question"]
            turn_scores = turn_index.scores(query)
            flat_order = ranking(turn_scores)
            flat_ids = [turn_ids[i] for i in flat_order]
            flat.add(flat_ids, gold)

            session_scores = session_index.scores(query)
            session_order = ranking(session_scores)
            flat_rr = reciprocal_ranks(flat_order)
            session_rr = reciprocal_ranks(session_order)
            fused = [
                1.00 * flat_rr[i] + 0.72 * session_rr[turn_to_session[i]]
                for i in range(len(turns))
            ]
            seed_order = ranking(fused)

            # Relation-aware orbital expansion: preserve the seed, then its two
            # nearest dialogue neighbors. This yields a complete ranked list,
            # so every k is comparable with the flat baseline.
            umd_order: list[int] = []
            seen: set[int] = set()
            for seed in seed_order:
                candidates = [seed, seed - 1, seed + 1]
                for candidate in candidates:
                    if (
                        0 <= candidate < len(turns)
                        and candidate not in seen
                        and turn_to_session[candidate] == turn_to_session[seed]
                    ):
                        seen.add(candidate)
                        umd_order.append(candidate)
            umd_ids = [turn_ids[i] for i in umd_order]
            umd.add(umd_ids, gold)

            # Protect the high-recall UMD 3.3 top-10 set and only improve its
            # internal ordering with a typo-tolerant character orbit.
            char_order = ranking(char_index.scores(query))
            char_rr = reciprocal_ranks(char_order)
            precision_score = [
                flat_rr[i] + 0.95 * session_rr[turn_to_session[i]] + 0.12 * char_rr[i]
                for i in range(len(turns))
            ]
            protected = umd_order[:10]
            protected_set = set(protected)
            head = sorted(protected, key=lambda i: (-precision_score[i], i))
            umd34_order = head + [i for i in umd_order if i not in protected_set]
            umd34.add([turn_ids[i] for i in umd34_order], gold)

            # UMD 3.7 query-comet force. Keep the validated UMD top-10
            # candidate orbit fixed, then combine lexical resonance, session
            # gravity, typo tolerance and adaptive temporal/identifier phase.
            temporal_weight, exact_weight = planetary_query_weights(query)
            turn_unit = unit_scores(turn_scores)
            session_unit = unit_scores(session_scores)
            planetary_force = [
                0.25 * flat_rr[i]
                + 0.22 * session_rr[turn_to_session[i]]
                + 0.10 * char_rr[i]
                + 0.25 * exact_weight * turn_unit[i]
                + 0.18 * temporal_weight * session_unit[turn_to_session[i]]
                for i in range(len(turns))
            ]
            umd37_head = sorted(protected, key=lambda i: (-planetary_force[i], i))
            umd37_order = umd37_head + [i for i in umd_order if i not in protected_set]
            umd37.add([turn_ids[i] for i in umd37_order], gold)
            context_chars["flat@10"].append(sum(len(turn_texts[i]) for i in flat_order[:10]))
            context_chars["umd@10"].append(sum(len(turn_texts[i]) for i in umd_order[:10]))

    return {
        "benchmark": "LoCoMo",
        "data_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "conversations": len(dataset),
        "qa_total": total_qa,
        "qa_with_evidence": total_qa - no_evidence,
        "qa_without_evidence": no_evidence,
        "unit": "dialogue turn",
        "flat_bm25": flat.result(),
        "umd33_hierarchy": umd.result(),
        "umd34_protected_rerank": umd34.result(),
        "umd37_planetary_orbit": umd37.result(),
        "umd34_config": {"session_rr": 0.95, "char_rr": 0.12, "protected_top_k": 10},
        "mean_retrieved_characters_at_10": {key: statistics.mean(values) for key, values in context_chars.items()},
        "runtime_seconds": time.perf_counter() - started,
    }


def _turn_text(turn: dict) -> str:
    return f'{turn.get("role", "")}: {turn.get("content", "")}'


def run_longmemeval(path: Path) -> dict:
    started = time.perf_counter()
    dataset = json.loads(path.read_text(encoding="utf-8"))
    ks = (1, 5, 10, 20)
    flat = RecallAccumulator(ks)
    umd = RecallAccumulator(ks)
    flat_all = RecallAccumulator(ks)
    umd_all = RecallAccumulator(ks)
    umd34 = RecallAccumulator(ks)
    umd34_all = RecallAccumulator(ks)
    umd37 = RecallAccumulator(ks)
    umd37_all = RecallAccumulator(ks)
    abstention = 0
    missing_evidence = 0
    answer_coverage = {
        "flat@5": 0, "flat@10": 0,
        "umd@5": 0, "umd@10": 0,
        "umd34@5": 0, "umd34@10": 0,
        "umd37@5": 0, "umd37@10": 0,
    }
    context_chars = {"flat@10": [], "umd@10": []}

    for item_number, item in enumerate(dataset, 1):
        session_ids = item["haystack_session_ids"]
        sessions = item["haystack_sessions"]
        dates = item["haystack_dates"]
        session_texts = [
            f"{dates[i]} " + " ".join(_turn_text(turn) for turn in session)
            for i, session in enumerate(sessions)
        ]
        session_index = BM25(session_texts)
        turn_texts: list[str] = []
        turn_to_session: list[int] = []
        for session_number, session in enumerate(sessions):
            for turn in session:
                turn_texts.append(_turn_text(turn))
                turn_to_session.append(session_number)
        turn_index = BM25(turn_texts)

        query = item["question"]
        is_abstention = item["question_id"].endswith("_abs")
        abstention += is_abstention
        aggregate_scores = session_index.scores(query)
        flat_order = ranking(aggregate_scores)
        flat_ids = [session_ids[i] for i in flat_order]
        gold = item.get("answer_session_ids") or []
        if not gold:
            missing_evidence += 1
            continue
        flat_all.add(flat_ids, gold)
        if not is_abstention:
            flat.add(flat_ids, gold)

        turn_scores = turn_index.scores(query)
        best_turn_score = [0.0] * len(sessions)
        for turn_number, score in enumerate(turn_scores):
            session_number = turn_to_session[turn_number]
            best_turn_score[session_number] = max(best_turn_score[session_number], score)
        aggregate_order = ranking(aggregate_scores)
        moon_order = ranking(best_turn_score)
        aggregate_rr = reciprocal_ranks(aggregate_order)
        moon_rr = reciprocal_ranks(moon_order)
        fused_scores = [
            1.00 * aggregate_rr[i] + 0.82 * moon_rr[i]
            for i in range(len(sessions))
        ]
        umd_order = ranking(fused_scores)
        umd_ids = [session_ids[i] for i in umd_order]
        umd_all.add(umd_ids, gold)
        if not is_abstention:
            umd.add(umd_ids, gold)

        # Freeze the proven top-1 and top-10 set. Only ranks 2..10 use the
        # lower moon weight selected on the development split.
        conservative_scores = [
            aggregate_rr[i] + 0.40 * moon_rr[i]
            for i in range(len(sessions))
        ]
        first = umd_order[:1]
        middle = sorted(umd_order[1:10], key=lambda i: (-conservative_scores[i], i))
        umd34_order = first + middle + umd_order[10:]
        umd34_ids = [session_ids[i] for i in umd34_order]
        umd34_all.add(umd34_ids, gold)
        if not is_abstention:
            umd34.add(umd34_ids, gold)

        temporal_weight, exact_weight = planetary_query_weights(query)
        aggregate_unit = unit_scores(aggregate_scores)
        moon_unit = unit_scores(best_turn_score)
        planetary_force = [
            0.24 * aggregate_rr[i]
            + 0.14 * moon_rr[i]
            + 0.38 * temporal_weight * aggregate_unit[i]
            + 0.24 * exact_weight * moon_unit[i]
            for i in range(len(sessions))
        ]
        # Preserve UMD's high-confidence first result and top-10 recall set.
        planetary_middle = sorted(umd_order[1:10], key=lambda i: (-planetary_force[i], i))
        umd37_order = umd_order[:1] + planetary_middle + umd_order[10:]
        umd37_ids = [session_ids[i] for i in umd37_order]
        umd37_all.add(umd37_ids, gold)
        if not is_abstention:
            umd37.add(umd37_ids, gold)

        if not is_abstention:
            answer = normalized(item["answer"])
            for label, order, k in (
                ("flat@5", flat_order, 5), ("flat@10", flat_order, 10),
                ("umd@5", umd_order, 5), ("umd@10", umd_order, 10),
                ("umd34@5", umd34_order, 5), ("umd34@10", umd34_order, 10),
                ("umd37@5", umd37_order, 5), ("umd37@10", umd37_order, 10),
            ):
                retrieved = normalized(" ".join(session_texts[i] for i in order[:k]))
                answer_coverage[label] += bool(answer and answer in retrieved)
        context_chars["flat@10"].append(sum(len(session_texts[i]) for i in flat_order[:10]))
        context_chars["umd@10"].append(sum(len(session_texts[i]) for i in umd_order[:10]))
        if item_number % 10 == 0:
            print(f"LongMemEval progress: {item_number}/{len(dataset)}", flush=True)

    answerable = len(dataset) - abstention
    return {
        "benchmark": "LongMemEval_S_cleaned",
        "data_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "questions": len(dataset),
        "answerable_questions": answerable,
        "abstention_questions": abstention,
        "questions_without_evidence_labels": missing_evidence,
        "unit": "session",
        "flat_bm25": flat.result(),
        "umd33_hierarchy": umd.result(),
        "umd34_protected_rerank": umd34.result(),
        "umd37_planetary_orbit": umd37.result(),
        "umd34_config": {"aggregate_rr": 1.0, "moon_rr_middle": 0.40, "protected_top_1": True, "protected_top_k": 10},
        "all_500_diagnostic_including_abstention": {
            "flat_bm25": flat_all.result(),
            "umd33_hierarchy": umd_all.result(),
            "umd34_protected_rerank": umd34_all.result(),
            "umd37_planetary_orbit": umd37_all.result(),
        },
        "exact_answer_string_coverage": {key: value / max(1, answerable) for key, value in answer_coverage.items()},
        "mean_retrieved_characters_at_10": {key: statistics.mean(values) for key, values in context_chars.items()},
        "runtime_seconds": time.perf_counter() - started,
    }


def markdown_report(results: dict) -> str:
    lines = [
        "# UMD 3.7 Public Benchmark Results",
        "",
        "This run uses official raw data and no paid model API. It measures evidence retrieval, not end-to-end QA accuracy.",
        "Gold answers and evidence labels are never used for ranking.",
        "",
    ]
    for name in ("locomo", "longmemeval"):
        if name not in results:
            continue
        item = results[name]
        lines += [f"## {item['benchmark']}", ""]
        if name == "locomo":
            lines.append(f"- Data: {item['conversations']} conversations, {item['qa_total']} QA, {item['qa_with_evidence']} with evidence")
        else:
            lines.append(f"- Data: {item['questions']} questions, {item['answerable_questions']} answerable, {item['abstention_questions']} abstention")
        lines += [
            f"- Retrieval unit: {item['unit']}",
            f"- Runtime: {item['runtime_seconds']:.2f} seconds",
            "",
            "| Retriever | MRR | Any R@1 | Any R@5 | Any R@10 | Full R@10 | Micro R@10 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for label, key in (
            ("Flat BM25", "flat_bm25"),
            ("UMD 3.3 hierarchy", "umd33_hierarchy"),
            ("UMD 3.4 protected rerank", "umd34_protected_rerank"),
            ("UMD 3.7 planetary orbit", "umd37_planetary_orbit"),
        ):
            metric = item[key]
            lines.append(
                f"| {label} | {metric['mrr']:.4f} | {metric['any_evidence_recall']['1']:.4f} | "
                f"{metric['any_evidence_recall']['5']:.4f} | {metric['any_evidence_recall']['10']:.4f} | "
                f"{metric['full_evidence_recall']['10']:.4f} | {metric['micro_evidence_recall']['10']:.4f} |"
            )
        lines.append("")
    peak = results["run_metadata"]["peak_working_set_mib"]
    lines += [
        "## Interpretation limits",
        "",
        "- UMD 3.7 adds adaptive identifier and temporal resonance while preserving the validated top-10 orbit.",
        "- Provenance is uniform in these public datasets; contradiction classification and learned neural semantics are not yet active.",
        "- Official QA accuracy still requires an answer-generating model and, for LongMemEval, the benchmark's evaluator.",
        f"- Peak process working set during the run: {peak:.2f} MiB." if peak is not None else "- Peak process working set: unavailable on this platform.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=("both", "locomo", "longmemeval"), default="both")
    parser.add_argument("--locomo", type=Path, default=DATA / "locomo10.json")
    parser.add_argument("--longmemeval", type=Path, default=DATA / "longmemeval_s_cleaned.json")
    args = parser.parse_args()

    started = time.perf_counter()
    RESULTS.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS / "public_benchmarks.json"
    results: dict[str, dict] = {}
    if args.benchmark != "both" and json_path.exists():
        previous = json.loads(json_path.read_text(encoding="utf-8"))
        results.update({
            key: value
            for key, value in previous.items()
            if key in {"locomo", "longmemeval"} and "umd37_planetary_orbit" in value
        })
    if args.benchmark in {"both", "locomo"}:
        results["locomo"] = run_locomo(args.locomo)
    if args.benchmark in {"both", "longmemeval"}:
        results["longmemeval"] = run_longmemeval(args.longmemeval)
    peak = peak_working_set_mib()
    gates: dict[str, bool] = {}
    if "locomo" in results:
        old = results["locomo"]["umd33_hierarchy"]
        new = results["locomo"]["umd34_protected_rerank"]
        planetary = results["locomo"]["umd37_planetary_orbit"]
        gates.update({
            "locomo_top10_preserved": new["any_evidence_recall"]["10"] == old["any_evidence_recall"]["10"],
            "locomo_full_top10_preserved": new["full_evidence_recall"]["10"] == old["full_evidence_recall"]["10"],
            "locomo_mrr_improved": new["mrr"] > old["mrr"],
            "locomo_hit1_improved": new["any_evidence_recall"]["1"] > old["any_evidence_recall"]["1"],
            "locomo_planetary_top10_preserved": planetary["any_evidence_recall"]["10"] == old["any_evidence_recall"]["10"],
        })
    if "longmemeval" in results:
        old = results["longmemeval"]["umd33_hierarchy"]
        new = results["longmemeval"]["umd34_protected_rerank"]
        planetary = results["longmemeval"]["umd37_planetary_orbit"]
        gates.update({
            "longmemeval_top1_preserved": new["any_evidence_recall"]["1"] == old["any_evidence_recall"]["1"],
            "longmemeval_top10_preserved": new["any_evidence_recall"]["10"] == old["any_evidence_recall"]["10"],
            "longmemeval_full_top10_preserved": new["full_evidence_recall"]["10"] == old["full_evidence_recall"]["10"],
            "longmemeval_hit5_improved": new["any_evidence_recall"]["5"] > old["any_evidence_recall"]["5"],
            "longmemeval_mrr_improved": new["mrr"] > old["mrr"],
            "longmemeval_planetary_top1_preserved": planetary["any_evidence_recall"]["1"] == old["any_evidence_recall"]["1"],
            "longmemeval_planetary_top10_preserved": planetary["any_evidence_recall"]["10"] == old["any_evidence_recall"]["10"],
        })
    results["run_metadata"] = {
        "adapter": "UMD 3.7 planetary protected orbit",
        "paid_api_calls": 0,
        "answer_or_evidence_used_for_ranking": False,
        "regression_gates": gates,
        "total_runtime_seconds": time.perf_counter() - started,
        "peak_working_set_mib": peak,
    }
    md_path = RESULTS / "PUBLIC_BENCHMARKS.md"
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(markdown_report(results), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
