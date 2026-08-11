"""Focused integration checks for the real ONNX neural encoder."""

from __future__ import annotations

import json
from pathlib import Path

from umd35_core import UMD35Config, UMD35Memory, cosine
from umd35_neural import FastEmbedEncoder, create_neural_memory


def run() -> dict:
    root = Path(__file__).resolve().parent
    encoder = FastEmbedEncoder(cache_dir=root / ".model-cache", cache_size=128)

    semantic_pairs = {
        "cross_language": ("火星计划的主数据库采用 PostgreSQL。", "Mars project uses PostgreSQL as its main database."),
        "paraphrase": ("用户喜欢简洁的周报。", "周度报告应该保持简短。"),
        "unrelated": ("火星计划的主数据库采用 PostgreSQL。", "今天的海浪适合冲浪。"),
    }
    flat = [text for pair in semantic_pairs.values() for text in pair]
    vectors = encoder.encode_many(flat)
    similarities = {}
    for index, name in enumerate(semantic_pairs):
        similarities[name] = cosine(vectors[index * 2], vectors[index * 2 + 1])

    memory = UMD35Memory(
        UMD35Config(duplicate_threshold=0.985),
        encoder=encoder,
    )
    target = memory.write(
        "火星计划的主数据库采用 PostgreSQL。",
        scope="mars-project",
        explicit_importance=1.0,
    )
    memory.write(
        "Edward 偏好简短的中文周报。",
        scope="communication",
        explicit_importance=1.0,
    )
    version_16 = memory.write(
        "Atlas uses PostgreSQL version 16.",
        scope="neural-conflict",
        explicit_importance=1.0,
    )
    version_17 = memory.write(
        "Atlas uses PostgreSQL version 17.",
        source="user_correction",
        scope="neural-conflict",
        explicit_importance=1.0,
    )
    retrieved = memory.retrieve(
        "Which database powers the Mars project?",
        scope="mars-project",
        top_k=2,
        reinforce=False,
    )
    # The same query should hit the in-process LRU cache on the second call.
    before_hits = encoder.cache_hits
    encoder.encode("Which database powers the Mars project?")
    cache_hit_added = encoder.cache_hits > before_hits
    extractor_marker = object()
    combined = create_neural_memory(
        cache_dir=root / ".model-cache",
        extractor=extractor_marker,
    )

    checks = {
        "real_neural_backend_loaded": bool(encoder.metadata()["loaded"]),
        "no_hash_fallback_used": encoder.metadata()["fallback_calls"] == 0,
        "cross_language_beats_unrelated": similarities["cross_language"] > similarities["unrelated"] + 0.15,
        "paraphrase_beats_unrelated": similarities["paraphrase"] > similarities["unrelated"] + 0.10,
        "cross_language_retrieval_finds_target": bool(retrieved) and retrieved[0].memory_id == target.memory_id,
        "embedding_cache_is_active": cache_hit_added,
        "neural_and_llm_layers_can_be_composed": (
            isinstance(combined.encoder, FastEmbedEncoder)
            and combined.extractor is extractor_marker
        ),
        "semantic_similarity_cannot_merge_conflicting_values": (
            version_17.memory_id != version_16.memory_id
            and version_17.superseded == version_16.memory_id
        ),
        "all_memory_invariants_hold": all(memory.check_invariants().values()),
    }
    return {
        "version": "UMD 3.5 neural integration",
        "checks": checks,
        "similarities": similarities,
        "retrieved": [
            {"text": item.text, "score": item.score, "explanation": item.explanation}
            for item in retrieved
        ],
        "encoder": encoder.metadata(),
        "snapshot": memory.snapshot(),
    }


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not all(result["checks"].values()):
        raise SystemExit(1)
