# UMD 3.15 Event-Lagrange Memory

UMD 3.15 addresses the LoCoMo failure mode where one relevant turn enters the
top-ten constellation but complementary evidence remains in the secondary
orbit. It preserves every UMD 3.14 source and attaches a bounded set of
query-time event satellites.

## Formula

For a secondary candidate `m` at atomic rank `r`:

`H15(m,q) = 0.46*field + 0.30*semantic + 0.14*lexical + 0.06/(1+r) + 0.04*new_planet`

The top sixteen candidates from a pool of 64 are striped across ten stable
capsules:

`C'_i = C_i union {event satellites assigned to i}`

Consequently, every stable top-ten provenance set is a subset of the UMD 3.15
set. No query changes star, planet, tenant, version or database membership.

## Safety and cost bounds

- Ranking receives query text, embeddings, force values and immutable metadata;
  it never receives answers, evidence IDs or LoCoMo categories.
- Quarantined, invalidated-current and out-of-scope memories are filtered by
  the inherited UMD 3.14 eligibility policy.
- Production retrieval respects the existing total character budget and emits
  explicit `[event-satellite:<source-id>]` provenance.
- Benchmark retrieval uses exactly ten capsules and at most sixteen additional
  unique sources. Retrieved character cost is reported beside quality.

## Validation protocol

The first five LoCoMo conversations were the development split for the original
eight-satellite model. Parameters were frozen before evaluating conversations
5-9, where R@10 reached `0.8629`. The later sixteen-satellite target
configuration was selected as the first point reaching R@10 >= 0.90 on a
full-data 4-20 source cost curve, so it is explicitly an internal optimization
and not a second blind-test claim.
