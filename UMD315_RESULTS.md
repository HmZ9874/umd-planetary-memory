# UMD 3.15 Results

These are gold-blind evidence-retrieval results, not end-to-end answer or judge
scores.

| Scope | Model | MRR | R@1 | R@5 | R@10 | Full R@10 | Micro R@10 |
|---|---|---:|---:|---:|---:|---:|---:|
| Full, 1,982 queries | UMD 3.14 | 0.6102 | 0.5061 | 0.7588 | 0.8073 | 0.7170 | 0.6297 |
| Full, 1,982 queries | UMD 3.15, 16 satellites | **0.6589** | **0.5353** | **0.8385** | **0.9001** | **0.8047** | **0.7505** |

Official LoCoMo categories 1-4 improve from R@10 `0.7839` to `0.8841`.
Mean retrieved characters rise from `2,415` to `4,683` (`+93.9%`).

| Event satellites | R@10 | Full R@10 | Mean characters | Increase |
|---:|---:|---:|---:|---:|
| 4 | 0.8567 | 0.7614 | 2,982 | +23.5% |
| 5 | 0.8648 | 0.7689 | 3,127 | +29.5% |
| 6 | 0.8683 | 0.7725 | 3,270 | +35.4% |
| 7 | 0.8718 | 0.7745 | 3,414 | +41.4% |
| 8 | 0.8734 | 0.7785 | 3,556 | +47.3% |
| 10 | 0.8819 | 0.7871 | 3,836 | +58.9% |
| 12 | 0.8885 | 0.7941 | 4,121 | +70.7% |
| 14 | 0.8961 | 0.8022 | 4,402 | +82.3% |
| 15 | 0.8981 | 0.8037 | 4,542 | +88.1% |
| 16 | **0.9001** | **0.8047** | 4,683 | +93.9% |

The original eight-satellite model was frozen before a 985-query held-out run
and scored R@10 `0.8629` there. The later 4-20 satellite cost curve uses the
already-seen full dataset, so the sixteen-satellite target configuration is an
internal optimization result rather than a new blind-test claim.

Final regression status: 31/31 discovered unit tests passed, including UMD 3.15
budget, provenance, hierarchy immutability, quarantine and scope isolation.
The inherited three adversarial rounds also passed 29/29.
