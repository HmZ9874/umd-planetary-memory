# UMD Public Results and Limitations

## 1. Integrity statement

The numbers below are **source/evidence retrieval proxies on official data**. They are not official end-to-end answer accuracy and are not official leaderboard submissions. Ranking completes before answers or evidence labels are read; published formal JSON files record `gold_used_for_ranking=false`.

Final means provenance-capsule retrieval. Strict means one immutable source per rank. They must not be presented as the same metric.

## 2. Formal snapshot

Snapshot date: 2026-08-11.

The table standardizes reporting fields, not task difficulty; raw scores from different benchmarks are not a common Elo ranking.

| Benchmark | Evaluated scope | Final Any R@1 | Final Any R@10 | Final Full R@10 | Strict Any R@1 | Strict Any R@10 | Strict Full R@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| LoCoMo | 1,982 | 0.5605 | 0.9082 | 0.8219 | 0.2861 | 0.7089 | 0.6060 |
| LongMemEval_S 3.28.2 frozen | 470 | 0.9468 | 1.0000 | 0.9936 | 0.8979 | 0.9745 | 0.8021 |
| LongMemEval_S 3.34 frozen holdout | 376 | 0.9415 | 1.0000 | 0.9920 | 0.9096 | 0.9840 | 0.9069 |
| LongMemEval_S 3.35 post-evaluation regression | 470 | 0.9745 | 1.0000 | 0.9936 | 0.9298 | 0.9872 | 0.9085 |
| LongMemEval_S 3.36 post-evaluation regression | 470 | **0.9979** | 1.0000 | 0.9936 | 0.9298 | 0.9872 | 0.9085 |
| MemBench stratified proxy | 550 | 0.8200 | 0.9982 | 0.9418 | 0.5782 | 0.9509 | 0.6000 |
| Memora | 398 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.5905 |
| MemoryBench DialSim | 34 | 0.4118 | 0.8824 | 0.2647 | 0.2353 | 0.3235 | 0.0000 |
| MemoryArena reuse | 1,714 | 0.9994 | 1.0000 | 1.0000 | 0.9708 | 1.0000 | 1.0000 |
| MemoryAgentBench UMD 3.33.1 | 1,152 | 0.8681 | 0.9705 | 0.7813 | 0.7639 | 0.8976 | 0.5200 |
| LongMemEval-V2 small | 230 | 0.9304 | 0.9870 | 0.5565 | 0.7652 | 0.9087 | 0.1174 |
| EverMemBench-Dynamic | 202 | 0.8465 | 0.9802 | 0.4604 | 0.4406 | 0.9010 | 0.2327 |

Memora additionally reports state-sector exact-set rate `0.9925`, old-value contamination `0.0`, and `107/107` correct deterministic aggregate-answer validations.

## 3. UMD 3.30 ablation

MemoryAgentBench uses all 30 selected contexts and 2,800 questions; 1,152 questions have answer-bearing sources under the frozen chunker.

| Metric | UMD 3.29 | UMD 3.30 | Absolute delta |
|---|---:|---:|---:|
| Final MRR | 0.6357 | 0.7549 | +0.1192 |
| Final Any R@1 | 0.5078 | 0.6727 | +0.1649 |
| Final Full R@10 | 0.3429 | 0.5712 | +0.2283 |
| Strict Any R@1 | 0.3767 | 0.5556 | +0.1788 |
| Strict Full R@10 | 0.1484 | 0.2595 | +0.1111 |

Accurate Retrieval is unchanged metric-for-metric. The improvement comes from multi-hop and version-conflict cases in Conflict Resolution.

## 3.1 UMD 3.30 → 3.31 evidence closure

| Metric | UMD 3.30 | UMD 3.31 | Absolute delta |
|---|---:|---:|---:|
| Final Any R@1 | 0.6727 | 0.7352 | +0.0625 |
| Final Full R@10 | 0.5712 | 0.6189 | +0.0477 |
| Strict Any R@1 | 0.5556 | 0.6398 | +0.0842 |
| Strict Full R@10 | 0.2595 | **0.4141** | **+0.1545** |

Strict Full R@10 improves by `59.5%` relative. Accurate Retrieval is unchanged; Conflict Resolution Strict Full R@10 rises from `0.2737` to `0.4963`. Full runtime increases by `6.5%`.

## 3.2 UMD 3.31 → 3.32.1 relation superposition

| Metric | UMD 3.31 | UMD 3.32.1 | Absolute delta |
|---|---:|---:|---:|
| Final Any R@1 | 0.7352 | **0.8281** | **+0.0929** |
| Final Full R@10 | 0.6189 | **0.7526** | **+0.1337** |
| Strict Any R@1 | 0.6398 | **0.7378** | **+0.0981** |
| Strict Full R@10 | 0.4141 | **0.5095** | **+0.0955** |

The same-capacity 600-chunk run reaches `0.9370 / 0.9701 / 0.9104 / 0.7065`, but does not replace full-run scores. Strict can return only one source per rank, giving a raw full-dataset Strict Full R@10 ceiling of `794/1152 = 0.6892`; all four raw metrics cannot simultaneously reach 0.99 under the current definitions. On the 600-question Conflict Resolution capacity subset, ceiling-normalized Strict Full R@10 is `0.7050 / 0.7450 = 0.9463`. See `benchmarks/results/UMD3321_RELATION_SUPERPOSITION_REPORT.md`.

## 3.3 UMD 3.32.1 → 3.33.1 binary document stars

The four core metrics rise from `0.8281 / 0.7526 / 0.7378 / 0.5095` to `0.8681 / 0.7813 / 0.7639 / 0.5200`. The subset-budget Final Full R@10 implementation ceiling is about `0.9531`, of which the result reaches `81.97%`; Strict Full reaches `75.44%` of its fixed ceiling. See `benchmarks/results/UMD3331_BINARY_STAR_REPORT.md`.

### 3.4 UMD 3.34 query fission on LongMemEval

Questions 0–99 are an explicitly post-dataset development replay (94 answerable). Parameters were then frozen before scoring questions 100–499 as a 376-answerable holdout. On that holdout, Strict Any R@1 rises `0.8936 → 0.9096`, Strict MRR `0.9138 → 0.9371`, and Strict Full R@10 `0.8218 → 0.9069`; Final Any R@1 and Full R@10 remain `0.9415 / 0.9920`. The combined 500-question replay is not blind and is reported separately. See `benchmarks/results/UMD334_QUERY_FISSION_REPORT.md`.

### 3.5 UMD 3.35 background ghost matter

The 500-question result is a post-evaluation regression, not a blind test. Relative to 3.34, Final Any R@1 moves `0.9468 → 0.9745`, Final Full R@1 `0.7383 → 0.8426`, and Strict Any R@1 `0.9213 → 0.9298`, while Final/Strict Full R@10 remain conserved at `0.9936 / 0.9085`. Per-query comparison yields 13 Final R@1 improvements with zero regressions and four Strict R@1 improvements with zero regressions. See `benchmarks/results/UMD335_GHOST_MATTER_REPORT.md`.

### 3.6 UMD 3.36 bounded ghost constellations

Relative to 3.35, Final Any R@1 moves `0.9745 → 0.9979` and Final Full R@1 `0.8426 → 0.9851`. Across 470 answerable queries, Final Full@1 improves on 67 with zero regressions; Final Full@10 and every Strict metric have zero per-query changes, with identical mean Final@10 characters. This remains a post-evaluation regression. See `benchmarks/results/UMD336_GHOST_CONSTELLATION_REPORT.md`.

## 4. Scope and bias

- LoCoMo: all 10 conversations; 1,982 of 1,986 questions have evidence labels.
- LongMemEval: retrieval scores cover 470 answerable questions; 30 abstention questions require an answer model.
- MemBench: the first 10 trajectories from each of 55 official groups (550 total), not a full 26,637-trajectory end-to-end run.
- MemoryAgentBench: Accurate Retrieval and Conflict Resolution only; Test-Time Learning and Long-Range Understanding are unscored.
- MemoryBench: DialSim answer-bearing session retrieval only; official continual-learning/response-quality metrics are not run.
- MemoryArena: prior-subtask answer-component reuse, not environment success. Group travel planner contributes 1,599 of 1,714 eligible questions.
- EverMemBench: only 202 of 3,121 questions can be mapped to answer-bearing day groups by the frozen matcher; `0.8465` is not all-question answer accuracy.
- LongMemEval-V2: small tier only; official reader accuracy/LAFS is not run.
- EvolMem: the audited upstream checkout was still marked under construction.

## 5. Data and auditability

Formal JSON includes counts, evidence totals, runtime, dataset SHA-256 where applicable, encoder metadata and gold-use declarations. Benchmark data is not redistributed; obtain it from upstream and follow its license.

Upstream examples:

- LoCoMo: `snap-research/locomo`
- LongMemEval cleaned: `xiaowu0162/longmemeval-cleaned`
- MemoryAgentBench: <https://github.com/HUST-AI-HYZ/MemoryAgentBench>
- EverMemBench: <https://github.com/EverMind-AI/EverMemBench>
- MemoryArena: <https://github.com/ZexueHe/MemoryArena>
- MemoryBench: <https://github.com/THUIR/MemoryBench>
- MemBench: <https://github.com/import-myself/Membench>
- Memora: <https://github.com/geniesinc/Memora>

## 6. Not established

- official end-to-end answer accuracy;
- an apples-to-apples comparison with commercial memory services on identical hardware, LLM and cost;
- a seven-day real-user failure/cost/quality-drift report;
- peer-review acceptance;
- production security certification.
