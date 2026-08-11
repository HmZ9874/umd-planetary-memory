# PMD 2.2 Formula Optimization Results

## Scope

These results come from deterministic synthetic adversarial and property tests
with separate training, validation, and shifted-attack seeds. They establish
useful prototype priors and reject unstable formulas; they are not production
accuracy claims.

## 1. Formation / write gate

Use a hard policy-risk veto followed by a bounded learned score:

```text
write = risk_veto AND sigmoid(w*x - 1.8937)
```

| Feature | Weight |
|---|---:|
| utility | 1.6318 |
| novelty | 0.7951 |
| explicit request | 1.7105 |
| task impact | 1.3840 |
| confidence | 0.7369 |
| redundancy | -0.7812 |
| risk | -1.6919 |

Holdout: precision 95.29%, recall 88.17%, F1 91.59%, unsafe-write rate
0%. The objective intentionally favors fewer false writes because they pollute
future retrievals.

## 2. Distance and attraction

Reject inverse-distance gravity. Its tested dynamic range was 125,722,786x,
allowing a nearly identical poisoned memory to dominate every other feature.
Use a bounded logit followed by sigmoid:

```text
attraction = hard_gate * sigmoid(weighted_features / temperature)
```

| Feature | Weight |
|---|---:|
| semantic similarity | 0.6193 |
| log mass | 0.1774 |
| confidence | 0.2610 |
| source trust | 0.8047 |
| validity | 4.1974 |
| scope match | 4.2480 |
| predicted utility | 4.6751 |
| independent evidence | 2.1071 |
| risk | -3.2207 |
| duplication | -1.9338 |

Shifted-attack top-1 accuracy was 99.32%, versus 70.58% for the previous soft
formula. If utility is unavailable, use a neutral value and zero reliability;
neutral utility retained 96.79%, while random trusted utility fell to 82.36%.

## 3. Mass update

```text
m' = clip(m*decay
          + 0.10*log(1 + new_independent_sources)
          + 0.11*counterfactual_utility
          - 0.50*conflict,
          0, 1)
```

Repeated claims from an already-counted source add no mass. Five independent
sources plus moderate utility changed mass from 0.25 to 0.4622. A full conflict
changed 0.80 to 0.30.

## 4. Orbit mapping

Reject:

```text
r = r_min + k/(mass + epsilon)
```

Use only for cache tiers or visualization:

```text
r = 1 + 9*(1 - mass)^1.25
```

The bounded radius has a 10x range rather than about 1,000,001x. Both radius
functions produce exactly the same top-k ordering as mass, proving that radius
must not be added back into retrieval scoring.

## 5. Decay

| Type | Prototype half-life |
|---|---:|
| core rule | no automatic decay |
| preference | 450 days |
| event | 152 days |
| temporary memory | 5.8 days |

Expiry and replacement are explicit state transitions, never decay. These
half-lives optimize a simulated 4:1 cost ratio between dropping useful memory
and retaining stale memory; product-specific calibration is required.

## 6. Collision / merge

```text
merge iff similarity >= 0.6607
       AND compatible
       AND same_scope
```

Conflicting memories are versioned, not merged. Guarded merge achieved 98.96%
precision and a 0.78% false-positive rate. Similarity alone produced a 27.11%
false-positive rate. The numeric threshold is embedding-model-specific.

## 7. Consolidation

```text
consolidate iff evidence_count >= 3
            AND coherence >= 0.61
            AND conflict <= 0.22
            AND source_diversity >= 0.35
```

Holdout precision was 97.71%, recall 53.73%, and false-positive rate 3.14%.
The low recall is intentional: uncertain episodes remain unconsolidated. A
summary is always derived data and never overwrites its raw episodes.

## 8. Context selection

```text
maximize attraction - 0.10*redundancy + coverage
subject to total_tokens <= budget
```

The diversity-aware selector improved simulated context utility by 3.78% over
plain relevance top-k.

## 9. Memory overhead

For one million 768-dimensional FP16 vectors, raw vectors occupy about 1.536
GB. PMD state plus four 24-byte edges per record adds approximately 0.192 GB,
or 12.5% over raw vectors, excluding text, ANN indexes, allocator overhead, and
model weights.

## Verdict

All automated regression gates pass. The viable model is a gated, bounded,
evidence-preserving memory controller. The celestial terms remain an organizing
metaphor; inverse-square attraction, singular orbit radii, repetition-based
mass, and destructive consolidation are rejected.
