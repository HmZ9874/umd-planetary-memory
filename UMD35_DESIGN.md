# UMD 3.5 Core Algorithm

UMD 3.5 stops optimizing around public benchmark scores and implements the
memory lifecycle itself. It operates on raw text and keeps the planetary model:

- star: a durable theme or explicit scope;
- planet: an episode/session within the theme;
- memory node: an atomic fact, preference, event, observation, or instruction;
- life/relations: bounded entity links between active memory nodes;
- version orbit: immutable correction and supersession chain.

The core now includes a real local ONNX neural backend through
`FastEmbedEncoder`. The default multilingual MiniLM model outputs 384 dimensions,
so active vector memory is unchanged. `TextEncoder`/`CallableEncoder` remain the
generic extension interface, while `HashEncoder` is the zero-model fallback.

Neural similarity alone is not sufficient for duplicate consolidation. A
candidate must also be an exact normalized match or exceed the lexical overlap
threshold. This prevents close neural vectors such as “version 16” and
“version 17” from reinforcing each other before contradiction handling.

## Write dynamics

The online write gate uses nine observable features:

```text
p(write) = sigmoid(b + w · [explicit, novelty, future value, correction,
                            source trust, relation density, actionability,
                            redundancy, injection risk])
```

The gate is initialized with conservative engineering priors and learns only
from explicit useful/not-useful feedback. Every weight and the bias are clipped
to `[-4, 4]`.

State transitions:

```text
raw -> stable       high write probability, trusted, safe
raw -> provisional  uncertain; may join but cannot create a hierarchy
raw -> floating     low value, detached and compressed
raw -> quarantined  injection risk, detached and compressed
provisional -> stable    explicit confirmation
provisional/floating/quarantined -> archived    explicit rejection
stable -> invalidated -> compressed audit record    trusted correction
stable/provisional -> archived    capacity, decay, or consolidation
```

Security classification runs before duplicate handling and contradiction
versioning. Therefore dangerous near-duplicates cannot reinforce trusted facts,
and quarantined records cannot be promoted through the correction path.

## Retrieval dynamics

Retrieval first selects stars, then planets, then active nodes. Node attraction:

```text
A = state_factor * (
      0.46 semantic + 0.06 lexical + 0.12 source_trust + 0.10 bounded_mass
    + 0.08 recency + 0.10 active_relation + 0.05 learned_utility
    + 0.03 scope_match)
```

The lexical channel preserves exact identifiers when a neural encoder is used.
The relation signal follows at most two bounded graph hops and decays each hop;
an unrelated neighbor no longer produces a positive relation score. Final
context uses MMR with a hard character budget. Normal current retrieval excludes
invalidated, quarantined, floating, and archived nodes. Explicit historical
queries may reconstruct compressed, superseded facts after applying both valid
time and knowledge time. Explanations report weighted contributions rather than
raw feature magnitudes.

## Structured and temporal facts

Each memory may carry multiple `FactFrame` records and typed `EntityRelation`
edges in addition to its primary `subject`, `predicate`, `object_value`, and
polarity. Lightweight English/Chinese extraction is the local fallback. The
optional LLM extractor proposes complex entities, relations, and temporal
intervals through a strict schema; its output is enrichment, not authority.
`valid_from`/`valid_to` describe when a claim is true, while `created_at` records
when the system learned it.

```text
visible(memory, valid_at, known_at) =
    created_at <= known_at
    and valid_from <= valid_at < effective_valid_to(known_at)
```

This lets the engine distinguish “what was true then” from “what the agent knew
then” without allowing obsolete facts into ordinary current retrieval.

The extraction trust boundary is:

```text
raw text -> local injection score -> deterministic write gate/state
         -> (safe stable/provisional only) strict LLM schema
         -> evidence-span, size, confidence, and interval validation
         -> structured facts/relations + auditable provider metadata
```

An extracted item is accepted only when its evidence is a substring of the raw
text after whitespace/case normalization. This is necessary grounding, though
not proof that the claim is true. LLM output cannot alter source trust, write
probability, importance, permissions, or lifecycle state. High-risk text is
quarantined before any external model call; API failures degrade to the local
extractor by default.

## Bounded growth

- active memories per star are capped;
- total active memories are capped;
- planets per star are capped;
- relation degree is capped in both directions;
- nearest-neighbor search scans only the bounded active index;
- inactive vectors are released completely;
- inactive text is reversibly compressed and restored through `get_text()`;
- compressed payloads can be moved out of RAM through `SQLiteArchiveStore`;
- raw lineage and version IDs remain addressable.

The UMD 3.5 SQLite backend uses atomic archive upserts and WAL journaling, but
its node metadata index still grows with historical record count. UMD 3.6's
`UMD36TenantMemory` supplies the stronger boundary: encrypted cold metadata is
evicted from RAM and the bounded active hierarchy is restartable from a chained
state transaction log.

## Internal validation

Lifecycle tests cover duplicate reinforcement, correction/versioning,
quarantine, dangerous near-duplicates, provisional confirmation, centroid
rebuild, retrieval exclusion, explanation correctness, capacity archival,
compressed recovery, SQLite reopen recovery, bitemporal visibility, bounded
multi-hop propagation, encoder injection, Chinese cue handling, feedback
learning, decay, and all structural invariants.

The real ONNX integration test additionally covers cross-language similarity,
paraphrase separation, cross-language retrieval, LRU embedding reuse, conflict
protection, and confirmation that no hash fallback was used.

The structured LLM adapter test uses the same Pydantic schema path as the live
Responses API and covers multi-fact extraction, typed relations, UTC time
normalization, fabricated-evidence rejection, token/model audit metadata,
provider-neutral local adapters, failure fallback, and pre-call injection
isolation.

The 5,000-write stress run used 20 themes with these results:

| Property | Result |
|---|---:|
| Historical records | 5,000 |
| Active records | 128 |
| Archived records | 4,852 |
| Active vector bytes | 196,608 |
| Inactive vector bytes | 0 |
| Inactive plaintext characters | 0 |
| Compressed archive bytes | 229,187 |
| Active relations | about 308 |
| First 500 mean write latency | about 0.32 ms |
| Last 500 mean write latency | about 0.38 ms |
| Latency growth | about 1.17x |

These are deterministic internal lifecycle/stress measurements, not public
benchmark claims.
