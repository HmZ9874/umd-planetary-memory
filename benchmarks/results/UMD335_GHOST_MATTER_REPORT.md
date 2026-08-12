# UMD 3.35 Background Ghost Matter / 后台幽灵物质

## 中文

UMD 3.35 增加一个可丢弃、可重建、来源安全的后台整理层。它从当前可见的角色化对话中提取用户事实子句，把绝对日期投影为相对时间别名，并在偏好或时间问题出现时提供导航。幽灵不是记忆事实，也不是答案证据；所有命中必须回锚到不可变原始 `source_id`。

安全边界：

- 无事实的请求产生“静默幽灵”，不生成内容；
- 普通非角色文档完全不激活幽灵；
- 偏好幽灵只补充 Final 首胶囊，不改变 Strict 排序；
- Strict 默认逐名次守恒；只有明确时间计算/顺序问题通过词法接触、分数比和边际三重门时，才允许把原始来源牵引到首位；
- 助手输出回忆问题和简单相对日期筛选不能触发 Strict 提升。

LongMemEval_S 500 题结果是**评估后回归**，不是盲测，也不是官方端到端答案分数。470 题可回答；排序时不读取 gold，`gold_used_for_ranking=false`。

| 指标 | UMD 3.34 | UMD 3.35 | 变化 |
|---|---:|---:|---:|
| Final MRR | 0.9671 | **0.9834** | +0.0163 |
| Final Any R@1 | 0.9468 | **0.9745** | +0.0277 |
| Final Full R@1 | 0.7383 | **0.8426** | +0.1043 |
| Final Full R@10 | 0.9936 | **0.9936** | 0 |
| Strict MRR | 0.9465 | **0.9511** | +0.0046 |
| Strict Any R@1 | 0.9213 | **0.9298** | +0.0085 |
| Strict Full R@10 | 0.9085 | **0.9085** | 0 |

逐题守恒审计：Final Any R@1 改善 13 题、回退 0 题；Strict Any R@1 改善 4 题、回退 0 题；Strict Any/Full R@10 均为改善 0、回退 0。Final 与 Strict 的平均 R@10 字符数与 3.34 相同，因此提升没有通过扩大检索输出获得。

## English

UMD 3.35 adds a disposable, rebuildable, provenance-safe background organization layer. It distills user fact clauses from visible role-marked dialogue, projects absolute dates into relative-time aliases, and supplies navigation only for preference or temporal queries. A ghost is neither a memory fact nor answer evidence: every hit must re-anchor to an immutable original `source_id`.

Safety boundaries:

- requests without facts produce silent ghosts rather than invented content;
- ordinary unmarked documents never activate ghost matter;
- preference ghosts may augment Final rank one but cannot change Strict order;
- Strict ranks are conserved by default; only explicit temporal computation/order queries may promote an original source after lexical-contact, ratio, and margin gates;
- assistant-output recollection and simple relative-date filters cannot trigger Strict promotion.

The 500-question LongMemEval_S run is a **post-evaluation regression**, not a blind test or official end-to-end answer score. There are 470 answerable questions. Ranking does not read gold evidence: `gold_used_for_ranking=false`.

| Metric | UMD 3.34 | UMD 3.35 | Delta |
|---|---:|---:|---:|
| Final MRR | 0.9671 | **0.9834** | +0.0163 |
| Final Any R@1 | 0.9468 | **0.9745** | +0.0277 |
| Final Full R@1 | 0.7383 | **0.8426** | +0.1043 |
| Final Full R@10 | 0.9936 | **0.9936** | 0 |
| Strict MRR | 0.9465 | **0.9511** | +0.0046 |
| Strict Any R@1 | 0.9213 | **0.9298** | +0.0085 |
| Strict Full R@10 | 0.9085 | **0.9085** | 0 |

The per-query conservation audit found 13 Final Any R@1 improvements and zero regressions, plus four Strict Any R@1 improvements and zero regressions. Strict Any/Full R@10 each had zero improvements and zero regressions. Mean retrieved characters at rank 10 are identical to 3.34 in both channels, so the gain does not come from widening retrieval output.

Raw manifest: `umd335_longmemeval_full_regression.json`. Implementation: `benchmarks/umd335_adapter.py`. Deterministic tests: `benchmarks/umd335_adapter_test.py`.
