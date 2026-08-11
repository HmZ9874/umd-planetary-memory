# UMD 3.6 Persistent Multi-Tenant Runtime

UMD 3.6 wraps the UMD 3.5 algorithm with a durable security boundary. The
in-process engine contains only the bounded active working set. Inactive nodes,
their text, structured facts, relations, temporal metadata, provenance, and
lineage are serialized into encrypted SQLite records and removed from all RAM
node, fact, entity, relation, planet, star, and staging-archive collections.

## Durable transaction boundary

Every mutating operation commits these items in one SQLite transaction:

1. an append-only event with the complete encrypted operation payload;
2. the resulting encrypted bounded structural state inside that event;
3. only encrypted memory records whose plaintext digest changed;
4. the latest materialized structural state;
5. an ACL update when the event is an authorization change.

Events form an HMAC-SHA256 chain. SQLite uses WAL, foreign keys, and
`synchronous=FULL`. Restart normally restores the materialized state. If that
state is missing or damaged, the newest decryptable transaction-log state is
used instead. Optimistic revision checks reject stale concurrent writers rather
than silently replacing another user's changes.

Node vectors/text are no longer copied into each event. The transaction log
contains the complete hierarchy/index structure and active IDs, while canonical
node payloads live once in `memory_records`. This preserves deterministic
recovery while avoiding the original full-active-universe write amplification.

## Tenant security

- every SQL lookup includes `tenant_id`;
- roles are `reader`, `writer`, and `admin`;
- unknown principals and cross-tenant memory IDs are rejected;
- each tenant receives an independent AES-256-GCM key derived with HKDF-SHA256;
- ciphertext is authenticated with tenant- and purpose-specific AAD;
- the 256-bit master key is never stored in SQLite;
- a wrong key, wrong tenant, or modified ciphertext fails authentication;
- stale multi-user sessions fail with `ConcurrencyError` and must reopen.

Memory content and full metadata are encrypted. SQLite necessarily retains
minimal routing/index columns such as tenant ID, random memory ID, active flag,
state, and timestamps. Production deployments should keep the database and
master key in separate access domains and obtain the key from a KMS/HSM.

## Deep graph reasoning and reflection

`graph_paths()` performs bounded beam search up to eight hops, rejects cycles,
multiplies edge strength and hop decay, and returns every memory in the evidence
path. `link_memories()` creates explicit tenant-local edges and audits them.

`reflect()` selects the strongest active memories and writes a low-trust,
auditable `assistant_inference` summary. A supplied callable may use an LLM;
without one, a deterministic fact summary is used. `reflection_interval`
enables automatic reflection after a bounded number of writes. Reflections do
not receive user trust and pass through the normal write gate.

## Long-running validation

Normal tenant sessions record real write/retrieval latency and success samples.
The runtime labels evidence as long-running only after all of these are true:

- at least 1,000 real samples;
- at least 168 hours between the first and latest real sample;
- at least 99% real-operation success.

`umd36_soak.py` supplies restartable synthetic load and crash/restart pressure.
Synthetic samples are stored separately and can never satisfy the real-user
qualification. Time cannot be compressed by a unit test; the current automated
result validates the collector and its anti-mislabeling rule, not seven days of
actual user behavior.
