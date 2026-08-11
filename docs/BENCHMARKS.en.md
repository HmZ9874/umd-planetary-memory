# UMD Public Results and Limitations

## 1. Integrity statement

The numbers below are **source/evidence retrieval proxies on official data**. They are not official end-to-end answer accuracy and are not official leaderboard submissions. Ranking completes before answers or evidence labels are read; published formal JSON files record `gold_used_for_ranking=false`.

Final means provenance-capsule retrieval. Strict means one immutable source per rank. They must not be presented as the same metric.

## 2. Formal snapshot

Snapshot date: 2026-08-11.

| Benchmark | Evaluated scope | Final Any R@1 | Final Any R@10 | Final Full R@10 | Strict Any R@1 | Strict Any R@10 | Strict Full R@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| LoCoMo | 1,982 | 0.5605 | 0.9082 | 0.8219 | 0.2861 | 0.7089 | 0.6060 |
| LongMemEval_S | 470 | 0.9468 | 1.0000 | 0.9936 | 0.8979 | 0.9745 | 0.8021 |
| MemBench stratified proxy | 550 | 0.8200 | 0.9982 | 0.9418 | 0.5782 | 0.9509 | 0.6000 |
| Memora | 398 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.5905 |
| MemoryBench DialSim | 34 | 0.4118 | 0.8824 | 0.2647 | 0.2353 | 0.3235 | 0.0000 |
| MemoryArena reuse | 1,714 | 0.9994 | 1.0000 | 1.0000 | 0.9708 | 1.0000 | 1.0000 |
| MemoryAgentBench UMD 3.30 | 1,152 | 0.6727 | 0.9271 | 0.5712 | 0.5556 | 0.7812 | 0.2595 |
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

