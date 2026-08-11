# UMD 3.4 Results

UMD 3.4 adds protected reranking to the UMD 3.3 public-data adapter. Parameters
were selected only on deterministic development splits and then evaluated on
holdout data. Gold answers and evidence labels are never ranking inputs.

## Frozen formulas

LoCoMo precision score inside the protected UMD 3.3 top-10:

```text
P(i) = RRF_word(i) + 0.95 * RRF_session(i) + 0.12 * RRF_char3(i)
```

LongMemEval keeps UMD 3.3 rank 1 and its complete top-10 candidate set. Only
ranks 2 through 10 are reordered with:

```text
P(i) = RRF_session(i) + 0.40 * RRF_best_turn(i)
```

Speaker boosts, role heuristics, and recency heuristics were tested but rejected
because the development split did not select them or holdout performance did
not justify them.

## Holdout validation

| Benchmark | Metric | UMD 3.3 | UMD 3.4 | Delta |
|---|---|---:|---:|---:|
| LoCoMo | Hit@1 | 35.08% | 36.90% | +1.83 pp |
| LoCoMo | Hit@5 | 62.22% | 62.91% | +0.69 pp |
| LoCoMo | Hit@10 | 73.87% | 73.87% | unchanged |
| LoCoMo | MRR | 46.34% | 49.06% | +2.72 pp |
| LongMemEval | Hit@1 | 88.19% | 88.19% | unchanged |
| LongMemEval | Hit@5 | 96.23% | 96.98% | +0.75 pp |
| LongMemEval | Hit@10 | 98.74% | 98.74% | unchanged |
| LongMemEval | MRR | 91.60% | 91.75% | +0.14 pp |

## Full official-data run

| Benchmark | Metric | UMD 3.3 | UMD 3.4 | Delta |
|---|---|---:|---:|---:|
| LoCoMo | Hit@1 | 35.87% | 37.54% | +1.67 pp |
| LoCoMo | Hit@5 | 62.51% | 62.66% | +0.15 pp |
| LoCoMo | Hit@10 | 73.81% | 73.81% | unchanged |
| LoCoMo | MRR | 46.87% | 49.32% | +2.44 pp |
| LongMemEval (official 470) | Hit@1 | 88.72% | 88.72% | unchanged |
| LongMemEval (official 470) | Hit@5 | 96.60% | 97.45% | +0.85 pp |
| LongMemEval (official 470) | Hit@10 | 98.94% | 98.94% | unchanged |
| LongMemEval (official 470) | MRR | 91.99% | 92.11% | +0.12 pp |

The protected design intentionally preserves top-10 recall. The tradeoff is a
small decrease in full multi-evidence recall at k=5 (about 0.4--0.6 percentage
points), while full and micro evidence recall at k=10 remain unchanged.
