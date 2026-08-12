# UMD Planetary Memory Algorithm White Paper

## 1. Objective

UMD is not a prompt-stuffing scheme. It is an experimental long-term memory system designed to decide what to store, preserve provenance, resolve versions, forget selectively, enforce access control, recover after restart, and retrieve both a single precise fact and a complete evidence set.

The system addresses write gating, entity/relation/time extraction, contradictory updates, multi-tenant encryption, bounded RAM, temporal and graph reasoning, large-set recall, memory poisoning, noisy tools, and long-term error propagation.

## 2. The computational universe

| Physical metaphor | Memory primitive |
|---|---|
| Universe | one tenant/user memory space |
| Galaxy | project, domain or long-lived identity |
| Star | stable topic, person or task |
| Planet | event, fact, preference or procedure |
| Moon | local detail, source or adjacent evidence |
| Asteroid belt | weak/noisy candidate evidence |
| Gravitational edge | semantic, entity, temporal or causal relation |
| Antimatter | deletion, cancellation, invalidation or negation |
| Event horizon | bounded context compression boundary |
| Slingshot | multi-hop traversal through intermediate entities |

Every metaphor maps to a bounded data structure, score term or state transition; it is not merely presentation language.

## 3. Write path and atomic facts

Input is split into immutable source records and atomic facts. Deterministic local code controls trust, source spans, field limits, time validity, injection risk and write policy. An optional LLM may propose entities, relations and valid-time intervals, but it cannot bypass those checks.

An atomic fact carries at least:

`(subject, predicate, object, valid_time, knowledge_time, source_id, tenant, state)`

Updates create new version edges. They do not silently rewrite immutable provenance.

## 4. Lifecycle and forgetting

Core states are provisional, stable, superseded, tombstone and archived.

1. New facts enter provisional state.
2. Repetition, explicit confirmation and trusted sources increase mass.
3. Conflicts form a version chain ordered by valid time, knowledge time, authority and update ordinal.
4. Delete/cancel operations create tombstones or negative state mass.
5. Inactive low-utility nodes move to durable storage.
6. Audit orbits retain history; current-answer orbits exclude invalid facts.

Forgetting therefore updates answer state, audit state, edges, indexes, transaction logs and encrypted persistence rather than merely deleting a vector.

## 5. Hierarchical retrieval

A query combines:

1. BM25 lexical resonance;
2. character n-gram typo/morphology resonance;
3. neural embedding similarity;
4. galaxy/session/topic mass;
5. entity and identifier resonance;
6. temporal phase;
7. adjacency and graph flux;
8. state mass, authority and antimatter repulsion;
9. event-satellite coverage and capsule compression;
10. a bounded slingshot graph when multi-hop structured reasoning is required.

Document embeddings are lazy and candidate-bounded. Historical metadata, encrypted facts and the transaction log can remain outside RAM; the runtime keeps only an active working set and small indexes.

## 6. Capsules and complete recall

Ordinary top-k binds one rank to one source. UMD uses provenance-preserving capsules:

`C_k = (anchor, adjacent evidence, relation satellites, echo sources)`

Ten stable anchors cannot be deleted by later optimizations. New evidence is striped across capsules, giving the prefix conservation invariant:

`Sources_old(@k) ⊆ Sources_new(@k)`

A strict atomic channel is reported in parallel. Every atomic rank contains exactly one immutable source, preventing an inflated R@1 obtained only by widening a capsule.

## 7. State fields and antimatter

Current-state and historical questions use different fields. Current questions downweight deletion, completion and cancellation facts in the answer orbit. Historical questions keep both positive and negative versions neutral so that transitions remain reconstructable.

Explicit invalidations are compiled once into a small immutable source-ID set shared by answer, atomic, striped and state-sector channels. This preserves audit history while avoiding four full text scans per query.

## 8. Galactic census

Aggregate, recommendation and “list all” queries require coverage rather than ten near duplicates. The census orbit detects breadth, ranks the full horizon with domain contact and state, separates current/superseded/tombstone facts, emits a compact structured payload, and can deterministically compute counts, sums, temporal windows and dimension aggregates with fact-level provenance.

## 9. UMD 3.30 gravitational slingshot

### Activation

The graph activates only when the context contains a sufficiently dense universe of numbered subject-relation-object statements. Ordinary dialogue remains on the frozen control path.

### Versioned edges

Each statement compiles to:

`e = (subject, relation, object, ordinal, source_ids)`

For a visible prefix `t`, the highest-ordinal edge for `(subject, relation)` exerts current-state gravity. Older edges remain auditable.

### Bounded traversal

The longest subject mentioned in the query becomes the launch star. Query wording identifies the requested relation, such as `continent`, `official_language`, `head_state` or `works_field`. A breadth-first traversal follows at most three current edges.

Example:

`Phoenix Inferno → sport → basketball → created_country → Italy → head_state → Sergio Mattarella`

The final answer-fact provenance is promoted to atomic rank one instead of stopping at the lexically obvious first hop.

### Echo asteroid belt

The graph object is a model prediction, not the benchmark answer. Other visible sources containing that predicted object form an echo set of at most 96 satellites. Echoes append to capsules and never remove protected anchors, increasing full-set recall while preserving the old prefix.

## 9.5 UMD 3.31 output-slot evidence closure

The main 3.30 failure was not the absence of a graph. It could mistake the longest relation cue in a question for the requested output: “where did the spouse die?” might stop at `married` instead of continuing to `died_city`. UMD 3.31 assigns terminal mass `Ω(r|q)` from an explicit answer-slot grammar and then searches a bounded four-hop closure.

A terminal hit produces three provenance layers: the answer fact (`Primary`), all visible occurrences of the predicted answer (`Echo`), and intermediate path sources (`Dependency`). The strict channel deduplicates `Primary → Echo → Dependency → conserved 3.30 order`. Every rank remains exactly one source, so full-recall gains do not come from widening the ranking unit.

This path still activates only for dense numbered fact universes. A local scan of all ten LoCoMo conversations activated neither the 3.30 nor 3.31 graph (0/10).

## 9.6 UMD 3.32.1 relation superposition and version shadows

UMD 3.32.1 represents the terminal relation as a superposition of answer-head, output-slot, and explicit relation cues rather than a single label. A five-hop search permits a necessary implicit bridge with small negative vacuum energy. Terminal mass, relation coverage, and bounded multi-path consensus jointly select the predicted object. Equal-ordinal fact fragments are resolved deterministically in favor of the more complete object and provenance set.

The current answer still enters `Primary → Echo` first. Historical values of the same subject–relation pair then enter `Shadow`, followed by path dependencies. This supports both current-state and evolution questions without allowing stale values to occupy rank one. Final closure carries at most 256 satellite sources. Strict always retains one source per rank and column-unfolds ordinary capsules to improve large-set complete recall. Capacity and ranking changes are reported separately.

A rare-entity lexical bridge and an MS MARCO cross-encoder both reduced R@1 in local ablations and are not enabled by default. The relation-superposition graph remains gated to dense numbered fact universes; ordinary LoCoMo dialogue stays on the frozen path.

## 9.7 UMD 3.33.1 binary document stars and terminal absorption

UMD 3.33.1 losslessly reconstructs `Document N:` boundaries from a regular fixed-window stream, treating complete documents as stars and overlapping windows as source moons. The first Final capsule conserves old sources and appends moons from the top two stars. Strict remains one source per rank and promotes a document-local source only at a `1.13×` first/second document-score ratio.

The structured graph absorbs a path after complete query-supported terminal coverage, rejects entity cycles, and subtracts `1.25` energy for every repeated relation. This repairs direct hits that previously wandered through loops such as `citizen → head_government → citizen`. Ordinary dialogue, irregular windows, and contexts with fewer than 20 document boundaries disable document stars.

## 10. Persistence, security and tenancy

UMD 3.6+ includes SQLite durability, bounded active RAM, per-tenant AES-256-GCM ciphertext, an HMAC-chained transaction log, atomic rollback/recovery, tenant isolation, ACL/ABAC, capability tokens, key rotation, wrong-key rejection and embedding-encoder identity checks.

The master key must live outside the database in a KMS, HSM or secret manager.

## 11. Version lineage

| Release | Main contribution |
|---|---|
| PMD / 3.1–3.4 | write gating, hierarchy, adversarial tests, protected reranking |
| 3.5 | online raw-text engine, neural backend, guarded LLM extraction |
| 3.6 | persistence, WAL/recovery, encryption, tenancy and deep graph reasoning |
| 3.7–3.9 | planetary runtime, temporal entity orbits, platform, SDK and console |
| 3.10–3.14 | cognitive routing, late interaction, relative gravity and transient bridges |
| 3.15–3.18 | event horizon, state field, fact planets and Roche first capsule |
| 3.19–3.24 | learned gravity, new matter, meson field, security repair and open-domain escape velocity |
| 3.25–3.28 | antimatter forgetting, census, versioned facts, structured ledger and multi-orbit retrieval |
| 3.29 | incremental coverage and query/state caching without ranking changes |
| 3.30 | versioned fact graph, three-hop slingshot and echo asteroid belt |
| 3.31 | output-slot relations, four-hop evidence closure, dependency provenance and strict atomic closure order |
| 3.32.1 | relation superposition, five-hop coverage collapse, answer-path consensus, version shadows and strict atomic column unfolding |
| 3.33.1 | overlap reconstruction, binary document stars, confidence gate, terminal absorption and cycle energy |

The repository retains the implementation, tests and design records for the complete lineage.

## 12. Limitations

- Most public numbers are provenance retrieval proxies, not answer accuracy.
- MemoryAgentBench Test-Time Learning and Long-Range Understanding are not yet scored.
- The deterministic relation parser has limited coverage outside templated structured facts.
- Common predicted objects require a hard echo budget to prevent context explosion.
- There is no seven-day real-user failure, cost and quality-drift report yet.
- The work is not peer reviewed and has no official leaderboard certification.
