# UMD 3.12 evidence constellation design

## Representation

- Moon: one immutable source memory.
- Planet: one session or event sequence.
- Star: a long-lived subject.
- Evidence capsule: one to four source IDs with a stable content-derived capsule ID.

Capsules never store another copy of source plaintext or embeddings. Membership
is built before a query from adjacency, bounded episode chunks, version chains,
and trusted directive/execution pairs. Rebuilding after restart streams at most
512 cold version records and retains IDs only.

## Retrieval

Coarse UMD retrieval supplies at most 160 candidate capsule IDs. Capsule
relevance is:

`R=.26S+.18L+.14E+.12T+.10C+.10D+.10P`

Candidates must pass `max(0.18, 0.72*R_best)`. Greedy selection then applies:

`R+.18*new_subgoal+.14*new_entity+.12*new_time+.10*contradiction+.08*new_source-.16*redundancy-.12*uncertainty`

The benchmark adapter uses an additional regression invariant: capsule number
`i` contains atomic UMD 3.9 source number `i`. Consequently expanded source
R@K cannot be below the atomic R@K.

## Safety and anti-gaming constraints

- Membership is query-independent and gold-label blind.
- A capsule contains at most four source IDs.
- Every output exposes exact immutable provenance.
- Scope, tenant, classification, quarantine, and redaction checks are applied
  to every source; one forbidden source rejects the entire capsule.
- Atomic and capsule metrics are never presented as the same retrieval unit.
- Raw text stays in encrypted durable storage and is materialized only for the
  selected answer context.

## Cost control

The default candidate orbit is 160 capsule IDs and the response is packed under
`budget_chars`. Long session capsules can approximately double expanded context,
so applications should use a smaller budget or atomic retrieval for already
large session-level memories.
