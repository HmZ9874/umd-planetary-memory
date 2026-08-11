# UMD 3.10 Cognitive Orbit Design

UMD 3.10 keeps the UMD 3.9 platform contract and replaces generic-only
retrieval with a role-aware cognitive layer. The change addresses failures in
instruction following, preference following, event ordering, and multi-session
evidence coverage without increasing embedding dimensionality.

## Memory structure

The existing hierarchy remains unchanged:

`tenant universe -> topic star -> session planet -> memory/moon`

Three sparse derived structures are added:

1. **Directive satellites** store IDs of trusted user instructions and
   preferences, grouped by scope and semantic family. Normal retrieval uses
   only the latest active item in each family. Contradiction queries may use a
   bounded four-version history.
2. **Episode chains** store `(created_at, memory_id)` pairs by star and planet.
   They connect neighboring events and make one anchor per relevant planet
   available for broad timeline or summary questions.
3. **Coverage scheduler** selects from the candidate orbit using new-planet,
   new-time-bucket, and new-entity rewards with a semantic redundancy penalty.

These structures copy no raw text or vectors. They are reconstructed from the
encrypted durable store after restart. Active memories contribute at most two
episode IDs; directive history is capped per family.

## Formulas

An injected directive or episode receives cognitive force:

`Fc = .30*S + .18*L + .24*D + .12*E + .08*T + .08*A`

Where `S` is semantic gravity, `L` lexical overlap, `D` directive-role
resonance, `E` episode resonance, `T` recency phase, and `A` source authority.

The final multi-evidence selector maximizes:

`Select(m) = F(m) + .16*new_planet + .10*new_time + .08*new_entity - .12*redundancy`

The original highest-confidence fact remains an anchor. Up to three directive
satellites are then reserved, so a preference cannot replace the task subject
but also cannot disappear merely because its wording differs from the query.

## Safety and temporal behavior

- Only `user`, `user_correction`, and `verified_tool` sources can enter a
  directive satellite orbit.
- Injection-risk records are quarantined before cognitive indexing.
- Scope and ABAC checks are applied to both ordinary and injected candidates.
- Normal preference retrieval selects the latest active member of a family.
- Explicit contradiction queries may surface bounded history and version-chain
  evidence instead of silently returning only the newest statement.
- `chronological_context()` returns selected evidence ordered by valid/event
  time for answer synthesis.

## Complexity

For `A` active memories and `D` directive families:

- additional ID memory: `O(A + D*h)`, with `h <= 4` by default;
- directive routing: `O(D)` over a bounded scan of at most 64 candidates;
- episode expansion: bounded neighbors plus one anchor per relevant planet;
- coverage selection: `O(k*C)` for output size `k` and bounded candidate set
  `C <= candidate_limit`.

No additional embedding model, embedding dimensions, plaintext cache, or
database transaction is introduced.

## Validation status

The UMD 3.10 adversarial suite covers atomic writes, distant instruction
retrieval, latest preference routing, bounded contradiction history, scope
isolation, prompt-injection quarantine, untrusted-source rejection,
multi-planet episode coverage, chronological output, exact-fact regression,
restart reconstruction, event-chain integrity, and bounded active-ID indexes.

This implementation improves the algorithmic mechanisms exposed by the
retrieval benchmarks. It is not yet a new official LoCoMo, LongMemEval, or
BEAM end-to-end score; those require a frozen benchmark rerun and an answerer
and judge model for formal QA metrics.
