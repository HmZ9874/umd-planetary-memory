# UMD 3.2 Optimization Results

> Historical report: several validation weaknesses in this version were found
> and corrected in `UMD33_RESULTS.md`. Do not treat the original UMD 3.2 safety
> gates as current evidence.

UMD 3.2 focuses on robustly attaching raw memories to multiple star systems,
rejecting open-set inputs, evolving star topology, and preserving reversible
evidence lineage. All results are deterministic synthetic tests.

## 1. Learned multi-orbit extractor

The stable-orbit classifier uses:

```text
probability = sigmoid((
    2.4253*semantic
  + 3.7580*relation
  + 3.7097*scope
  - 1.2868*contradiction
  - 4.7368
) / temperature)
```

Calibrated parameters:

- temperature: `0.35`;
- stable threshold: `0.98`;
- provisional threshold: `0.90`;
- maximum orbit candidates: `3`.

Training uses hard-negative weighting for unrelated stars that nevertheless
have high semantic similarity. Training and calibration data also include a
shifted attack distribution.

### Stable orbit under shifted attack

| Metric | Result |
|---|---:|
| membership recall | 46.47% |
| memory-level query coverage | 60.89% |
| orbit contamination | 10.63% |
| open-set false attachment | 7.01% |
| open-set abstention | 92.99% |
| mean stable orbits | 0.68 |

Stable orbits may participate in normal routing. The intentionally conservative
threshold leaves many memories unattached rather than forcing a wrong home.

### Provisional orbit under shifted attack

| Metric | Result |
|---|---:|
| membership recall | 68.02% |
| memory-level query coverage | 79.93% |
| orbit contamination | 21.97% |
| open-set false attachment | 24.55% |

Provisional orbits are exploration hints only. They receive capped activation
and cannot trigger consolidation, parent-mass reinforcement, permissions, or
destructive updates.

The test confirms a useful separation:

```text
stable attachment = precision path
provisional attachment = recall path
```

## 2. Open-world behavior

When no score reaches 0.98, the memory remains a floating asteroid. A new star
may be proposed only after multiple related floating memories accumulate; the
classifier never creates a permanent star from one uncertain observation.

Semantic similarity alone could not satisfy the joint contamination and
open-set constraints. This supports using relation, scope, and contradiction as
first-class routing inputs.

## 3. Automatic star splitting

Split score:

```text
split_score = cluster_balance * centroid_separation / within_cluster_scatter
```

- threshold: `1.0048`;
- holdout precision: 99.78%;
- holdout recall: 88.45%;
- false-positive rate: 0.22%.

A split is a reversible proposal. The original star ID and child links remain
addressable until post-split retrieval quality is verified.

## 4. Automatic star merging

Merge score:

```text
merge_score =
    0.50*centroid_similarity
  + 0.35*membership_overlap
  - 0.65*conflict
```

- threshold: `0.4208`;
- holdout precision: 99.07%;
- holdout recall: 94.60%;
- false-positive rate: 0.73%.

Conflict has the largest effective veto contribution. High semantic similarity
is insufficient when two stars contain incompatible facts or permissions.

## 5. Evidence lineage

Every derived memory stores immutable source IDs and a transformation version:

```text
derived_memory = source_ids + transform_version + reversible materialization
```

In the test:

- 12,000 raw memories produced 4,000 derived memories;
- invalidating 900 raw memories marked 1,071 derived memories stale;
- orphaned derivations: 0;
- incorrect rebuilds without valid evidence: 0;
- all consolidations remained reversible.

## 6. Remaining limitation

Stable recall falls to 46.47% under the shifted attack distribution. This is a
deliberate safety tradeoff, but it means a production system needs:

1. a stronger real-language relation and scope extractor;
2. delayed promotion from provisional to stable using user feedback and
   independent evidence;
3. benchmark calibration by domain rather than one universal threshold.

All seven UMD 3.2 regression gates pass. These numbers remain prototype priors
until evaluated on real multi-session conversation benchmarks.
