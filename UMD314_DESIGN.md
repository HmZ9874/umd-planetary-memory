# UMD 3.14 adaptive tidal Lagrange optimizer

UMD 3.14 keeps UMD 3.13's immutable home orbits and adds three bounded
optimization stages.

## 1. Multi-attractor satellites

A memory is no longer forced into exactly one query subgoal. If several
relative fields are within the configured potential window, the source may
cover all of them. Each subgoal still owns a reserved candidate quota.

## 2. Coverage-driven sparse tidal propagation

Directly relevant seeds propagate a field only through shared entities or
extracted relations. Semantic similarity can strengthen an existing structural
link but cannot create one by itself.

`F_g=Delta_g+sum_h(.34*.58^(h-1)*max_j(link(i,j)*wave_h(j))), h<=4`

Propagation starts with two hops and can deepen to four through a bounded
structural frontier when coverage is incomplete. Only sparse shared-entity or
extracted-relation neighbors are visited. Entities occurring in more than 25%
of the local population are excluded from propagation, and link force is
multiplied by absolute inverse-degree mass so a common hub cannot obtain full
force merely because it is the only shared entity.

## 3. Adaptive capture and Lagrange bridge search

Every attractor receives a threshold from the current candidate distribution:
`median + 0.35*MAD`. A minimum potential and per-goal quotas prevent both field
collapse and domination by one easy subgoal.

Bridge membership uses a bounded beam search. Its objective is:

`L=field+.20adaptive_goal+.18cohesion+.12cross_planet+.07time+.05source-.18unexplained_redundancy-.14budget-.08ambiguity`

The uncovered-goal multiplier grows when evidence coverage is incomplete; the
budget multiplier grows with context consumption. The search remains bounded
by candidate, branch, beam, source and character limits.

## 4. Cold and historical recall

Authorized archived memories may enter a bounded 512-record cold scan and a
64-record lexical/entity preselection. Invalidated versions are considered
only for explicit historical queries such as "before" or "previously". Cold
records remain durable and are not silently promoted back into the active set.

## 5. Budget and conflict safety

If the only correct source is larger than the character budget, retrieval emits
an atomic prefix while retaining the complete immutable source ID. Named
single-value facts such as `city is Lisbon` versus `city is Oslo` now form a
version chain even when values contain no numbers and the second write omitted
an explicit correction cue.

## Safety and storage invariants

- Query fields and bridges are transient.
- Durable star/planet assignments never change.
- Every output retains immutable source IDs.
- Authorization and source eligibility are applied before propagation.
- No bridge plaintext, vector or graph is persisted in RAM or the database.
- Unicode NFKC and format-character removal precede compact injection-cue checks.
