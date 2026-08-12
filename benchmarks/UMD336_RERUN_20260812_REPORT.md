# UMD 3.36 isolated cold rerun / UMD 3.36 隔离冷启动复测

Date / 日期: 2026-08-12

## 中文结论

截图中的 12 行均已重新运行。评测器使用全新 `run_id` 和独立 checkpoint，未复用历史排名；所有结果均记录 `gold_used_for_ranking=false`。六项展示指标与先前公开表格一致。MemoryAgentBench `Final Full R@10` 的精确值为 `0.78125`，按常规四舍五入显示为 `0.7813`。

这仍是证据检索代理复测，不是官方端到端答案准确率、LLM judge 分数或排行榜提交。LongMemEval 3.35/3.36 是评估后回归；3.34 行是预声明的参数冻结保留集；3.28.2 行是冻结控制版本。

## English conclusion

All 12 rows from the referenced table were rerun. The harness used fresh `run_id` values and isolated checkpoints, so no historical ranking was reused. Every result records `gold_used_for_ranking=false`. All six displayed metrics reproduce the previously published table. The exact MemoryAgentBench `Final Full R@10` value is `0.78125`, conventionally displayed as `0.7813`.

These remain evidence-retrieval proxy reruns, not official end-to-end answer accuracy, LLM-judge scores, or leaderboard submissions. LongMemEval 3.35/3.36 are post-evaluation regressions; the 3.34 row is the predeclared parameter-frozen holdout; 3.28.2 is the frozen control.

## Reproduced metrics / 复现指标

| Benchmark / 基准 | Evaluated / 可评估 | Final Any R@1 | Final Any R@10 | Final Full R@10 | Strict Any R@1 | Strict Any R@10 | Strict Full R@10 | Cold runtime / 冷启动秒数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LongMemEval_S 3.36 post-evaluation regression | 470 | 0.9979 | 1.0000 | 0.9936 | 0.9298 | 0.9872 | 0.9085 | 849.4 |
| LoCoMo | 1,982 | 0.5605 | 0.9082 | 0.8219 | 0.2861 | 0.7089 | 0.6060 | 78.0 |
| LongMemEval_S 3.28.2 frozen | 470 | 0.9468 | 1.0000 | 0.9936 | 0.8979 | 0.9745 | 0.8021 | 834.3 |
| LongMemEval_S 3.34 frozen holdout | 376 | 0.9415 | 1.0000 | 0.9920 | 0.9096 | 0.9840 | 0.9069 | 918.0 |
| LongMemEval_S 3.35 post-evaluation regression | 470 | 0.9745 | 1.0000 | 0.9936 | 0.9298 | 0.9872 | 0.9085 | 1,077.5 |
| MemBench stratified proxy | 550 | 0.8200 | 0.9982 | 0.9418 | 0.5782 | 0.9509 | 0.6000 | 118.0 |
| Memora | 398 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.5905 | 265.2 |
| MemoryBench DialSim | 34 | 0.4118 | 0.8824 | 0.2647 | 0.2353 | 0.3235 | 0.0000 | 33.2 |
| MemoryArena reuse | 1,714 | 0.9994 | 1.0000 | 1.0000 | 0.9708 | 1.0000 | 1.0000 | 165.1 |
| MemoryAgentBench UMD 3.36 | 1,152 | 0.8681 | 0.9705 | 0.7813 | 0.7639 | 0.8976 | 0.5200 | 2,558.8 |
| LongMemEval-V2 small | 230 | 0.9304 | 0.9870 | 0.5565 | 0.7652 | 0.9087 | 0.1174 | 101.2 |
| EverMemBench-Dynamic | 202 | 0.8465 | 0.9802 | 0.4604 | 0.4406 | 0.9010 | 0.2327 | 2,395.4 |

Summed evaluator runtime was `9,394.3 s` (`156.6 min`, `2.61 h`). This is the sum of per-process evaluator timers, not a standardized hardware benchmark.

## Additional observations / 补充观察

- LoCoMo performed 7,823 neural encoder calls and reproduced all six metrics exactly.
- LongMemEval_S 3.36 performed 43,696 neural encoder calls over 500 questions and reproduced the published 470-answerable result.
- MemoryAgentBench rebuilt all 30 contexts (22 Accurate Retrieval and 8 Conflict Resolution), covering 2,800 questions and 1,152 answer-bearing questions. Test-Time Learning and Long-Range Understanding remain unscored.
- EverMemBench processed all five English users and 3,121 questions, but only 202 questions had answer-bearing day groups under the current fixed matcher. Its score is not whole-suite answer accuracy.
- MemoryBench remains the weakest complete-evidence case: `Final Full R@10=0.2647` and `Strict Full R@10=0.0000` on 34 answer-bearing queries.
- Cold-start scalability remains a material engineering weakness. MemoryAgentBench took about 42.6 minutes; EverMemBench took about 39.9 minutes and its observed working set reached roughly 2.56 GB.

## Isolation and validation / 隔离与验证

- Added `--run-id` to both exam entry points so reruns cannot silently reuse historical checkpoint rankings.
- Fresh checkpoints were used for every checkpointed benchmark and LongMemEval version.
- Official local data hashes recorded in the result JSON remained unchanged.
- The tracked UMD test scope passed: `148 passed`.
- Both modified exam modules passed `py_compile`; `git diff --check` passed.
- The 12 final rerun JSON files are published under `benchmarks/results/`; bulky intermediate checkpoints remain local and ignored by Git.

## Raw result files / 原始结果文件

- [LongMemEval_S UMD 3.36](results/rerun_20260812_umd336_longmemeval.json)
- [LoCoMo](results/rerun_20260812_umd336_locomo.json)
- [LongMemEval_S UMD 3.28.2](results/rerun_20260812_umd3282_longmemeval.json)
- [LongMemEval_S UMD 3.34 holdout](results/rerun_20260812_umd334_holdout.json)
- [LongMemEval_S UMD 3.35](results/rerun_20260812_umd335_longmemeval.json)
- [MemBench](results/rerun_20260812_umd336_membench.json)
- [Memora](results/rerun_20260812_umd336_memora.json)
- [MemoryBench](results/rerun_20260812_umd336_memorybench.json)
- [MemoryArena](results/rerun_20260812_umd336_memoryarena.json)
- [MemoryAgentBench](results/rerun_20260812_umd336_memoryagentbench.json)
- [LongMemEval-V2](results/rerun_20260812_umd336_longmemeval_v2.json)
- [EverMemBench](results/rerun_20260812_umd336_evermembench.json)
