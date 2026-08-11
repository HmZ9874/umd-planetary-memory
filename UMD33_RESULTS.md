# UMD 3.3 Bug Fix and Robustness Results

UMD 3.3 fixes the circular tests identified in UMD 3.1/3.2 and introduces an
orbit model evaluated on attack regimes not present in training or calibration.
All results remain synthetic and do not replace raw-language benchmarks.

## Fixed test defects

### Quarantine promotion

The previous test contained a contradictory condition and therefore always
reported zero false releases. The corrected test executes a promotion function
over actual source-ID lists:

```text
promote iff explicit_confirmation OR unique_source_count >= 2
```

Repeated observations from one source never promote a memory. A naive
repeat-count rule falsely promoted 99.9% of the generated repeated-source cases.

### Lifecycle

The previous illegal-transition counter was never incremented. The corrected
test executes guarded transitions and checks:

- graph-level transition permission;
- two independent sources or explicit confirmation for provisional -> stable;
- at least three sources and conflict <= 0.22 for structured -> consolidated;
- equivalent evidence guards for release from quarantine.

Across 100,000 attempted transitions, illegal transitions and unsafe provisional
promotions were both zero.

### Bitemporal state

The vectorized bitemporal implementation is now checked against an independent
row-wise oracle rather than defining both prediction and truth from one array.
The corrected implementation had zero oracle disagreements; ingestion-recency
retrieval remained at 72.47%.

### Evidence lineage

The lineage test now injects 100 malformed derivations: invalid source IDs and
insufficient source lists. All 100 were detected, none survived validation, and
no derived memory was rebuilt from invalidated evidence.

### Baseline reporting

Threshold tuning now reports `constraints_met: false` when no operating point
satisfies safety constraints. It selects the least-violating fallback instead
of silently returning 0.98 and presenting it as a safe comparator.

## Robust orbit model

Training uses only `normal` and `hard` regimes. Final tests use two regimes that
do not appear in training or calibration:

- `coherent_open`: unknown topics mimic known topics in semantic, relationship,
  and scope features;
- `calibration_drift`: coherent unknowns plus systematic +0.08 relation/scope
  bias and +0.04 source-trust bias on negatives.

The model adds three features that were absent from UMD 3.2:

- source trust;
- neighborhood density inside the proposed star;
- novelty relative to the known star.

Learned prototype weights:

| Feature | Weight |
|---|---:|
| semantic | 1.9003 |
| relation | 2.2152 |
| scope | 2.2334 |
| contradiction | -1.1090 |
| source trust | 2.9338 |
| neighborhood density | 3.0403 |
| known-star novelty | -4.4716 |

Calibration uses a class-balanced Brier score. Temperature is 0.35 and the
stable threshold is 0.6166; thresholds are meaningful only together with the
calibrated temperature and feature extractors.

### Results

| Dataset | Recall | Contamination | Open-set false attachment | Query coverage |
|---|---:|---:|---:|---:|
| hard validation | 97.88% | 2.50% | 3.99% | 98.84% |
| unseen coherent-open | 98.08% | 2.85% | 6.11% | 99.00% |
| unseen calibration drift | 98.14% | 5.69% | 12.70% | 99.09% |

The model is substantially more robust than UMD 3.2 under coherent camouflage,
but calibration drift remains the leading synthetic failure mode.

## Executable state permissions

UMD 3.3 implements an action whitelist:

```text
stable:
  retrieve, propagate, propose_consolidation,
  propose_fact_update, update_parent_mass

provisional:
  retrieve_capped, request_confirmation,
  collect_independent_evidence

quarantined:
  inspect_metadata, request_confirmation

floating:
  raw_search, request_classification
```

Provisional and quarantined memories had zero dangerous-action leaks. Promotion
requires probability >= 0.98, two independent sources, confirmed scope, or an
explicit confirmation. Randomized policy testing found zero unsafe promotions.

## Remaining limitations

1. The feature extractor still receives synthetic scores rather than raw text.
2. Source trust and neighborhood density can themselves be poisoned.
3. Split/merge tests still use simplified synthetic geometry.
4. There is no persistent store, concurrent writer, deletion propagation, or
   public long-conversation benchmark yet.
5. Calibration drift false attachment is 12.70%, so high-consequence actions
   must still use quarantine and explicit confirmation.

All eight UMD 3.3 regression gates pass under the current synthetic suite.
