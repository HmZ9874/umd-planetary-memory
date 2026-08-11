# UMD 3.14 measured engineering results

## Regression suite

The repaired UMD 3.12–3.14 stack passes 21 deterministic unit tests. The UMD
3.5 lifecycle self-test also reports all 27 checks and all storage, hierarchy,
versioning, safety and capacity invariants as true.

New permanent regression gates cover:

- atomic excerpt fallback when the only source exceeds the character budget;
- NFKC/format-character/compact-cue injection normalization;
- named single-value conflict versioning;
- cross-session and multi-attractor evidence;
- immutable durable home orbits and provenance;
- dense distractors, scope isolation and restart recovery.

## Adversarial repair result

The initial 15-case red-team run passed 8 and failed 7. After repair, all 15
cases pass. Three final complete reruns also produced 15/15 with runtimes of
2.427, 2.487 and 2.581 seconds.

| Previously failing attack | Repaired result |
|---|---|
| Single oversized correct source | atomic excerpt returned at rank 1 |
| Precise query for capacity-evicted memory | cold source returned at rank 1 |
| Space-split prompt injection | quarantined and excluded |
| Lisbon/Oslo equal-trust conflict | old value invalidated and version-linked |
| Historical "before Porto" query | invalidated Lisbon source returned |
| Common-entity hub crosstalk | answer retained; hub noise reduced to 2/4 |
| Crowded three-edge chain | all four provenance sources returned |

## Local scale diagnostic

A synthetic run with 152 active memories (two relevant and 150 unrelated)
used the deterministic local encoder and requested three bridges under a
5,000-character budget. Across 30 untraced warm queries:

| Check | Result |
|---|---:|
| Complete two-source chain | 30/30 |
| Median latency | 5.518 ms |
| p95 latency | 6.332 ms |
| Maximum latency | 9.127 ms |
| Median Python traced query peak | about 1.05 MiB |
| Persistent bridge/cache created | no |
| Durable orbit changed | no |

The additional memory is transient query workspace for candidates, sparse
neighbors and beam states; it is not retained after retrieval.

## Remaining limits

- The cold scan is bounded to the 512 most recent inactive records.
- Structural propagation is capped at four hops.
- Injection normalization is not a substitute for a trained security model.
- Named-value conflict inference covers configured single-value predicates;
  extraction errors can still produce a wrong fact key.
- These are synthetic engineering tests, not LoCoMo, LongMemEval, production
  latency, or real-user security certification.
