# UMD 3.7 Planetary Interaction Algorithm

UMD 3.7 turns the earlier planetary metaphor into the actual retrieval,
reasoning, learning, security, and distribution formulas. A query is a comet;
active memories are planets; canonical entities are celestial bodies; temporal
relations are orbits; contradictions create repulsion; permissions are local
magnetosphere barriers; replicas form a rendezvous constellation.

## Query-comet force

Candidate generation is the union of random-hyperplane ANN buckets, one-bit
neighbor buckets, BM25 postings, exact entity postings, and bounded active
scope candidates. The force on memory `i` is:

```text
w'_k(q) = w_k * a_k(q) / Σ_j(w_j * a_j(q))

F_i(q) = S_i * [
    w'_s cos(q, i)              semantic gravity
  + w'_b BM25(q, i)             lexical resonance
  + w'_e Jaccard(E_q, E_i)      entity resonance
  + w'_t phase(q, i)            temporal alignment
  + w'_g flux_h(q, i)           bounded graph propagation
  + w'_a trust(i)               source authority
  + w'_u utility(i)             learned usefulness
  + w'_r reranker(q, i)         optional neural cross-encoder
  - w_c contradiction(q, i)     contradiction repulsion
]

r_i = 1 / (0.05 + F_i)
E_i = -mass_i * F_i
```

Temporal questions increase `w_t`; relationship questions increase `w_g` and
`w_e`; version numbers and exact identifiers increase `w_b`. Explicit ranking
feedback applies a bounded pairwise update using the feature difference between
the selected memory and rejected memories. Every learned weight stays in
`[0.01, 0.60]` and is encrypted in planetary state.

## Entity and time orbits

Names are normalized and may be merged by embedding similarity. Every relation
stores subject, predicate, entity/literal object, valid interval, knowledge
time, confidence, and source memory. A new value for a functional predicate
closes the old orbit instead of deleting it:

```text
old.valid_to = new.valid_from
old.invalidated_at = new.learned_at
```

Graph reasoning performs cycle-free, temporally filtered beam search for up to
eight hops. Path strength is the product of relation confidence and hop decay.
Repeated relation patterns accumulate subject/object type statistics and are
promoted into the learned ontology after a configurable observation count.

## Evidence-conserving reflection

An LLM or deterministic proposer may create claims, but it must return source
memory IDs. A claim is accepted only when:

```text
grounding(c) = |tokens(c) ∩ tokens(evidence)| / |tokens(c)| >= τ
and contradiction_count(evidence) = 0
```

Accepted summaries remain `assistant_inference`, pass through the ordinary
write gate, retain evidence IDs, and never inherit user trust. Unsupported or
internally contradictory claims are rejected and audited. Reflection may run
automatically after a bounded number of writes.

## Multiple memory species

- active/archival declarative memories remain planets and asteroids;
- shared `MemoryBlock` satellites provide bounded always-visible agent state;
- `ProcedureOrbit` stores triggers, preconditions, steps, and Beta-smoothed
  success confidence, with UCB exploration during selection;
- `ModelMemoryOrbit` stores encrypted prompt, KV-cache, or adapter payloads and
  requires an exact model fingerprint before selection;
- model-memory and retrieval feedback update bounded quality/force parameters.

## Security and retention

UMD 3.6 RBAC remains the base barrier. UMD 3.7 ABAC additionally filters by
action, scope, tag, and classification. Per-tenant AES-256-GCM encryption,
HKDF keys, HMAC event chains, stale-writer rejection, key rotation, encrypted
planetary state, automatic expiry, physical record deletion, and content-event
redaction are implemented. Full database key rotation re-encrypts memory,
materialized state, transaction events, and planetary sidecars in one SQLite
transaction.

## Incremental durability and constellations

Transaction state now contains only bounded hierarchy/index structure and
active IDs. Node payloads live once in `memory_records`; SHA-256 plaintext
digests prevent unchanged nodes from being encrypted and rewritten. Recovery
loads active node records plus the latest structure log. Missing materialized
state is reconstructed from the chained event log.

`RendezvousConstellation` provides deterministic rendezvous placement, majority
write quorum, health-aware reads, failover, and revision reconciliation. It is
the transport-neutral sharding algorithm; production network replication still
requires adapters for the selected database/service infrastructure.

## Public retrieval results

Official raw LoCoMo and LongMemEval data were ranked without answer text,
evidence labels, or paid model calls:

| Dataset | Metric | UMD 3.4 | UMD 3.7 |
|---|---:|---:|---:|
| LoCoMo | MRR | 0.4932 | 0.4994 |
| LoCoMo | Any Recall@5 | 0.6266 | 0.6736 |
| LoCoMo | Full Recall@5 | 0.5394 | 0.5898 |
| LoCoMo | Any Recall@10 | 0.7381 | 0.7381 |
| LongMemEval | MRR | 0.92107 | 0.92131 |
| LongMemEval | Any Recall@5 | 0.9745 | 0.9723 |
| LongMemEval | Full Recall@5 | 0.8468 | 0.8532 |
| LongMemEval | Any Recall@10 | 0.9894 | 0.9894 |

These are evidence-retrieval results, not end-to-end answer accuracy. The
planetary reranker preserves the validated top-10 candidate orbit; on
LongMemEval it also preserves top-1. The small Any Recall@5 regression is
reported rather than hidden.

