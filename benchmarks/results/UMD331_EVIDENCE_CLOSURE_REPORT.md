# UMD 3.31 Evidence-Closure Optimization / UMD 3.31 证据闭包优化

## 中文

### 目标与失败诊断

UMD 3.30 已能在 MemoryAgentBench 找到部分正确来源，但 `Any R@10` 与 `Full R@10` 的差距说明主要瓶颈位于：问题输出关系识别、候选扩展、依赖链重建与完整证据聚合。具体错误包括把 `spouse`、`citizenship`、`country of origin` 等中间关系误当成问题最终要求的关系。

UMD 3.31 不使用答案或证据标签参与排序。它只接收问题和当前可见来源，并加入四个算法关系：

1. 输出槽关系质量：`Ω(r|q) = 1 − 0.08 × position(r)`，旧版回退线索为 `0.50`；
2. 四跳闭包能量：`E₃.₃₁(p|q) = Σ[0.20 + min(3, contact(q,r)/5)] + 8 + Ω(r_last|q)`；
3. 证据闭包：`Closure = Primary ∪ Echo ∪ Dependency`；
4. 严格原子顺序：`unique(Primary ⧺ Echo ⧺ Dependency ⧺ Strict₃.₃₀)`。

严格通道的每个名次仍然只含一个来源，因此提升不是通过扩大排名单位获得的。新路径只在密集编号事实宇宙中激活；LoCoMo 全部 10 段会话扫描均未激活 3.30 或 3.31 图路径。

### 30/30 上下文全量结果

全量运行覆盖 2,800 个问题，其中 1,152 个问题能由冻结切块定位答案来源。两版使用同一数据、切块、编码器和检索口径。

| 指标 | UMD 3.30 | UMD 3.31 | 绝对变化 | 相对变化 |
|---|---:|---:|---:|---:|
| Final Any R@1 | 0.6727 | 0.7352 | +0.0625 | +9.3% |
| Final Any R@10 | 0.9271 | 0.9306 | +0.0035 | +0.4% |
| Final Full R@10 | 0.5712 | 0.6189 | +0.0477 | +8.4% |
| Strict Any R@1 | 0.5556 | 0.6398 | +0.0842 | +15.2% |
| Strict Any R@10 | 0.7812 | 0.8290 | +0.0477 | +6.1% |
| Strict Full R@10 | 0.2595 | **0.4141** | **+0.1545** | **+59.5%** |

Accurate Retrieval 的 Final/Strict Full R@10 均逐项不变；Conflict Resolution 的 Final Full R@10 从 `0.6400` 提升到 `0.7087`，Strict Full R@10 从 `0.2737` 提升到 `0.4963`。这与算法只修复结构化依赖闭包的设计相符。全量运行时间从 `1144.80 s` 增至 `1218.85 s`（`+6.5%`）。

Strict Full R@10 的 gold-size-only 理论上限约为 `0.6892`，因为 1,152 个可评估问题中有大量问题包含超过 10 个 gold 来源。3.31 的 `0.4141` 达到该容量上限的约 `60.1%`；这不是重新归一化后的官方指标，只用于解释原始分数。

### 同容量 A/B

以下两次运行使用相同官方数据、相同 600-chunk 上限、相同编码器设置和相同 603 个可评估问题。`gold_used_for_ranking=false`。

| 指标 | UMD 3.30 | UMD 3.31 | 绝对变化 |
|---|---:|---:|---:|
| Final Any R@1 | 0.6965 | 0.8226 | +0.1260 |
| Final Any R@10 | 0.9486 | 0.9652 | +0.0166 |
| Final Full R@10 | 0.7347 | 0.8192 | +0.0846 |
| Strict Any R@1 | 0.6136 | 0.7761 | +0.1625 |
| Strict Any R@10 | 0.8292 | 0.9221 | +0.0929 |
| Strict Full R@10 | 0.3184 | 0.5954 | **+0.2769** |

Conflict Resolution 单项的 Strict Full R@10 从 `0.3183` 提升到 `0.5967`，绝对提升 `0.2783`，相对提升约 `87.4%`。运行时间为 3.30 的 `362.70 s` 对 3.31 的 `361.29 s`，差异约 `−0.4%`，可视为同一量级。

### 验证与边界

- 新旧适配器与集成测试合计 `114 passed`；
- 专项覆盖配偶去世地、公民身份到洲、工作城市三跳、原产国到首都、普通对话禁用；
- 这是来源/证据检索代理，不是官方端到端回答准确率或 leaderboard 提交；
- MemoryAgentBench 只计 Accurate Retrieval 与 Conflict Resolution，未计 Test-Time Learning 和 Long-Range Understanding；
- gold 来源由答案字符串在冻结切块中的出现位置定义；gold 来源数超过 10 的问题在 Strict Full R@10 下理论上不可完成，因此该指标的未归一化上限小于 1；
- 不把不同 benchmark 的 raw score 当作统一 Elo 排名。

## English

### Objective and failure diagnosis

UMD 3.30 often retrieved some correct MemoryAgentBench sources, but the `Any R@10`–`Full R@10` gap localized the bottleneck to output-relation parsing, candidate expansion, dependency reconstruction, and evidence aggregation. Typical errors treated intermediate relations such as `spouse`, `citizenship`, or `country of origin` as the requested endpoint.

UMD 3.31 never accepts an answer or evidence label as a ranking input. Given only the query and visible sources, it adds four algorithm relations:

1. output-slot mass: `Ω(r|q) = 1 − 0.08 × position(r)`, with a `0.50` legacy fallback;
2. four-hop closure energy: `E₃.₃₁(p|q) = Σ[0.20 + min(3, contact(q,r)/5)] + 8 + Ω(r_last|q)`;
3. evidence closure: `Closure = Primary ∪ Echo ∪ Dependency`;
4. strict atomic order: `unique(Primary ⧺ Echo ⧺ Dependency ⧺ Strict₃.₃₀)`.

Every strict rank still contains exactly one source, so gains do not come from widening a ranking unit. The new path activates only in dense numbered-fact universes. A scan of all ten LoCoMo conversations activated neither the 3.30 nor 3.31 graph.

### Full 30/30-context result

The full run covers 2,800 questions, 1,152 of which have answer-bearing sources under the frozen chunker. Both versions use the same data, chunker, encoder, and retrieval definition.

| Metric | UMD 3.30 | UMD 3.31 | Absolute delta | Relative delta |
|---|---:|---:|---:|---:|
| Final Any R@1 | 0.6727 | 0.7352 | +0.0625 | +9.3% |
| Final Any R@10 | 0.9271 | 0.9306 | +0.0035 | +0.4% |
| Final Full R@10 | 0.5712 | 0.6189 | +0.0477 | +8.4% |
| Strict Any R@1 | 0.5556 | 0.6398 | +0.0842 | +15.2% |
| Strict Any R@10 | 0.7812 | 0.8290 | +0.0477 | +6.1% |
| Strict Full R@10 | 0.2595 | **0.4141** | **+0.1545** | **+59.5%** |

Accurate Retrieval Final/Strict Full R@10 is unchanged metric-for-metric. Conflict Resolution Final Full R@10 rises from `0.6400` to `0.7087`, while Strict Full R@10 rises from `0.2737` to `0.4963`. That localization matches the design: UMD 3.31 activates only for structured dependency closure. Full runtime rises from `1144.80 s` to `1218.85 s` (`+6.5%`).

The gold-size-only ceiling for strict source-atomic Full R@10 is about `0.6892`, because many of the 1,152 evaluable questions have more than ten gold sources. The 3.31 result reaches about `60.1%` of that capacity ceiling. This is explanatory only, not a renormalized official metric.

### Same-capacity A/B

Both runs use the same official data, 600-chunk cap, encoder configuration, and 603 evaluable questions. `gold_used_for_ranking=false`.

| Metric | UMD 3.30 | UMD 3.31 | Absolute delta |
|---|---:|---:|---:|
| Final Any R@1 | 0.6965 | 0.8226 | +0.1260 |
| Final Any R@10 | 0.9486 | 0.9652 | +0.0166 |
| Final Full R@10 | 0.7347 | 0.8192 | +0.0846 |
| Strict Any R@1 | 0.6136 | 0.7761 | +0.1625 |
| Strict Any R@10 | 0.8292 | 0.9221 | +0.0929 |
| Strict Full R@10 | 0.3184 | 0.5954 | **+0.2769** |

Conflict Resolution Strict Full R@10 rises from `0.3183` to `0.5967`: `+0.2783` absolute and about `+87.4%` relative. Runtime is `362.70 s` for 3.30 versus `361.29 s` for 3.31 (`−0.4%`), effectively the same tier.

### Validation and boundaries

- The integrated new/old suite reports `114 passed`.
- Targeted cases cover spouse-to-death-place, citizenship-to-continent, a three-hop work-city chain, origin-country-to-capital, and ordinary-dialogue deactivation.
- These are provenance-retrieval proxies, not official end-to-end answer accuracy or leaderboard submissions.
- MemoryAgentBench scores Accurate Retrieval and Conflict Resolution only; Test-Time Learning and Long-Range Understanding remain unscored.
- Gold sources are frozen chunks containing answer strings. Questions with more than ten gold sources are mathematically impossible under strict source-atomic Full R@10, so its raw ceiling is below one.
- Raw scores from different benchmarks are not treated as a common Elo ranking.
