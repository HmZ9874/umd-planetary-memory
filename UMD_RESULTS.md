# UMD 3.1 Optimization Results

> Historical report: the quarantine, lifecycle, and bitemporal tests were later
> strengthened to remove circular checks. See `UMD33_RESULTS.md` for the current
> validation status.

UMD 3.1 adds universe/star/planet routing and graph-like "biospheres" on top
of the bounded PMD 2.2 memory controller. Results below are deterministic
synthetic stress tests, not production benchmark claims.

## 1. Multi-orbit routing

```text
pi(memory, star) = softmax(
    semantic + relation + scope - conflict,
    temperature=0.8
)
```

- Retain at most three candidate orbits.
- A candidate becomes a stable orbit only when probability is at least 0.145.
- Lower-confidence candidates remain provisional asteroids.
- Single-parent query coverage: 28.69%.
- Three-orbit provisional coverage: 62.33%.
- Stable-orbit coverage: 43.02%.
- Stable-orbit contamination: 19.38%.
- Mean stable orbit count: 1.61.

The remaining contamination shows that orbit extraction, not the hierarchy
formula, is now a primary weakness. Stable attachment must remain reversible.

## 2. Uncertainty-aware confidence

Use a Beta posterior and a lower confidence bound:

```text
confidence = beta_mean - z*beta_std
```

| Mode | z | Unsafe accept | Useful recall |
|---|---:|---:|---:|
| mean only | 0 | 0.920% | 77.72% |
| ordinary | 0.025 | 0.827% | 76.61% |
| high stakes | 0.625 | 0.163% | 57.86% |

Risk level must choose the confidence policy. One universal confidence score is
not suitable for both casual conversation and consequential actions.

## 3. Bitemporal facts

```text
eligible = event_time in [valid_from, valid_to)
           AND ingested_at <= knowledge_cutoff
```

Ingestion-recency retrieval reached 72.47% on historical queries. Bitemporal
filtering reached 100% in the property test because it enforces the definition
used to construct the valid historical state. This is a correctness invariant,
not a learned accuracy result.

## 4. Activation conservation

```text
activation_i = B * softmax(logit_i / temperature)
```

Use `B=12` and retain at most 64 active memories after routing. Raw independent
sigmoid activation grew from 32.1 at 100 candidates to 31,452 at 100,000
candidates; budgeted activation remained exactly 12.

## 5. Typed relationship propagation

Normalize each relation channel independently:

```text
A(k+1) = (1-eta)*A0 + eta*sum_r(theta_r * P_r * A(k))
```

Prototype parameters:

- `eta = 0.45`;
- maximum three propagation steps;
- causal weight `1.0`;
- semantic weight `0.12`;
- contradiction weight `-0.55`.

Typed causal-chain recall@6 reached 91.9%, versus 2.93% when all edge types
were flattened. Independent per-type normalization prevents many weak semantic
edges from drowning out one strong causal edge.

## 6. Quarantine

Untrusted memories are capped at activation 0.45 until confirmed by at least
two independent sources or explicit user confirmation.

- Poison top-1 rate without quarantine: 90.91%.
- Poison top-1 rate with quarantine: 0%.
- A naive repeat-count release rule would falsely release 39.89%.
- Same-source repetition under the independent-source rule releases 0%.

## 7. Lifecycle

The guarded state machine supports:

```text
floating -> provisional -> stable -> structured -> consolidated -> archived
```

Quarantine, invalidation, deletion, and restoration use explicit guarded
transitions. Across 100,000 random transition attempts, no illegal transition
was accepted.

## 8. Star homeostasis

```text
star_activation = local_activation - 0.28*log(1 + child_count)
```

Routing accuracy improved from 48.64% to 65.91%, while selection of the largest
star fell to 6.39%. Topic entropy should trigger star splitting, but its
threshold still needs real-data calibration.

## 9. Procedural memory

```text
skill_score = beta_mean * (0.35 + 0.65*context_match)
```

The expected-success optimum used `z=0`; consequential skill execution can
reuse the high-stakes lower-bound policy. Outcome-and-context scoring identified
the best skill 74.58% of the time, versus 12.82% for usage count alone.

## Regression status

All nine UMD 3.1 synthetic regression gates pass. The strongest improvements
are typed relations, quarantine, activation conservation, and bitemporal state.
The weakest remaining component is reliable extraction of stable multi-orbit
memberships from raw language.
