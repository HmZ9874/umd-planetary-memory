# UMD 3.32.1 Relation Superposition / UMD 3.32.1 关系叠加场

## 中文

### 结论

UMD 3.32.1 在不读取 gold、不改变 Strict 单来源定义的条件下，继续提高 MemoryAgentBench 的问题分解、版本冲突与完整证据闭包。30/30 上下文全量结果：

| 指标 | UMD 3.31 | UMD 3.32.1 | 绝对变化 |
|---|---:|---:|---:|
| Final Any R@1 | 0.7352 | **0.8281** | +0.0929 |
| Final Full R@10 | 0.6189 | **0.7526** | +0.1337 |
| Strict Any R@1 | 0.6398 | **0.7378** | +0.0981 |
| Strict Full R@10 | 0.4141 | **0.5095** | +0.0955 |

Conflict Resolution 单项达到 Final Any R@1 `0.9050`、Final Full R@10 `0.9013`、Strict Any R@1 `0.8600`、Strict Full R@10 `0.6038`。Accurate Retrieval 的 Final 与 Strict Any R@1 保持不变，Strict Full R@10 从 `0.2273` 提升到 `0.2955`。

### 新物理关系

1. **多终点关系叠加**：问题中的 answer-head、output-slot 和普通关系线索同时产生终点质量，不再遇到第一个规则就停止。
2. **带负真空能的路径边**：问题未提及的桥仍可通过，但获得 `−0.18` 能量；被问题支持的边获得正能量。
3. **覆盖坍缩**：路径覆盖的问题关系越完整，终点能量越高，author → spouse → citizen 等依赖链不会停在中间边。
4. **答案路径共振**：预测相同对象的独立路径获得最高 `0.45` 的有界共识质量。
5. **重叠碎片修复**：同一编号、主语和关系同时出现完整/截断切块版本时，优先完整对象。
6. **版本影子轨道**：最新答案及回声保持首位；旧冲突版本只在其后进入审计闭包。
7. **自适应闭包视界**：结构化闭包卫星上限由 96 提升到 256；最坏 Final top-10 来源容量不超过约 296。
8. **严格原子列展开**：先读取每个 Final 胶囊的最强来源，再读取同一胶囊的第二来源；每个 Strict 名次仍只有一个来源。

### 同容量结果

相同 600-chunk 上限、603 个可评估问题：

| 指标 | UMD 3.31 | UMD 3.32.1 | 绝对变化 |
|---|---:|---:|---:|
| Final Any R@1 | 0.8226 | 0.9370 | +0.1144 |
| Final Full R@10 | 0.8192 | 0.9701 | +0.1509 |
| Strict Any R@1 | 0.7761 | 0.9104 | +0.1343 |
| Strict Full R@10 | 0.5954 | 0.7065 | +0.1111 |

容量档 Conflict Resolution 的 Strict Full R@10 原始上限为 `0.7450`；`0.7050` 相当于 size-only ceiling-normalized `0.9463`。该归一化值只解释容量限制，不替代原始指标。

### 全量上限与目标

全量 Strict Full R@10 的 size-only 上限是 `0.6892`，因此原始 `0.99` 在 `k=10`、每个名次一个来源的定义下数学上不可能。当前 `0.5095` 约为该上限的 `73.9%`。Final 当前硬容量约 296 个来源，按 gold 集合大小计算，其 Full R@10 上限也低于 1。

### 被否决的实验

- **稀有实体二跳桥**：Accurate Retrieval Final Any R@1 从 `0.6534` 降到 `0.6222`，Strict Any R@1 从 `0.4602` 降到 `0.4261`；默认关闭。
- **MS MARCO ONNX 胶囊内交叉重排**：Strict Any R@1 降到 `0.4006`，Strict Full R@10 降到 `0.1818`，352 题开发回放约 `897 s`，而基础路径约 `239 s`；拒绝进入正式配置。
- **句级 BM25 峰值**：首位命中 `154/352`，低于块级 BM25 的 `172/352`；拒绝进入正式配置。

### 诚信边界

- 成绩是答案承载来源检索代理，不是官方端到端回答准确率或 leaderboard 提交。
- 只计 Accurate Retrieval 与 Conflict Resolution；Test-Time Learning、Long-Range Understanding 尚未计分。
- 排序返回后才读取答案并构造评测 gold；正式 JSON 记录 `gold_used_for_ranking=false`。
- 本轮是在已经查看 MemoryAgentBench 后进行的开发优化，不是统计独立的新盲测。
- 全量运行时间从 3.31 的 `1218.85 s` 变为 `942.26 s`；这是同机观测值，可能包含模型/文件系统热缓存差异，不能解释为稳定的 22.7% 加速保证。

## English

### Result

UMD 3.32.1 improves query decomposition, version conflict handling, and complete evidence closure without reading gold or widening a Strict rank beyond one source.

| Metric | UMD 3.31 | UMD 3.32.1 | Absolute delta |
|---|---:|---:|---:|
| Final Any R@1 | 0.7352 | **0.8281** | +0.0929 |
| Final Full R@10 | 0.6189 | **0.7526** | +0.1337 |
| Strict Any R@1 | 0.6398 | **0.7378** | +0.0981 |
| Strict Full R@10 | 0.4141 | **0.5095** | +0.0955 |

Conflict Resolution reaches `0.9050 / 0.9013 / 0.8600 / 0.6038` in the same order. Accurate Retrieval Final and Strict Any R@1 are conserved; Strict Full R@10 rises from `0.2273` to `0.2955`.

### New physical relations

1. Multi-terminal relation superposition retains answer-head, output-slot, and ordinary relation cues simultaneously.
2. Unsupported bridge edges carry `−0.18` vacuum energy; query-supported edges carry positive energy.
3. Coverage collapse rewards paths that reconstruct the complete mentioned dependency chain.
4. Independent paths predicting the same object receive at most `0.45` bounded consensus mass.
5. Equal-ordinal overlap fragments prefer the more complete object.
6. Version shadows follow the current answer and echoes; they never displace the newest fact from rank one.
7. The structured closure horizon grows from 96 to 256 satellites, with an approximate worst-case Final top-10 capacity of 296 sources.
8. Strict atomic column expansion takes the strongest source from each Final capsule before its second source; every Strict rank remains source-atomic.

### Same-capacity result

With the same 600-chunk cap and 603 evaluable questions:

| Metric | UMD 3.31 | UMD 3.32.1 | Absolute delta |
|---|---:|---:|---:|
| Final Any R@1 | 0.8226 | 0.9370 | +0.1144 |
| Final Full R@10 | 0.8192 | 0.9701 | +0.1509 |
| Strict Any R@1 | 0.7761 | 0.9104 | +0.1343 |
| Strict Full R@10 | 0.5954 | 0.7065 | +0.1111 |

Capacity-tier Conflict Resolution has a raw Strict Full R@10 size ceiling of `0.7450`; the observed `0.7050` is `0.9463` of that ceiling. This explanatory normalization does not replace the raw score.

### Full-run ceiling and target

The full-run size-only ceiling for Strict Full R@10 is `0.6892`, so a raw `0.99` is mathematically impossible when `k=10` and every rank contains one source. The current `0.5095` reaches about `73.9%` of that ceiling. Final also has a hard capacity of about 296 sources and therefore a gold-size ceiling below one.

### Rejected experiments

- The rare-entity two-hop bridge reduced Accurate Retrieval Final Any R@1 from `0.6534` to `0.6222` and Strict Any R@1 from `0.4602` to `0.4261`; it is disabled by default.
- MS MARCO ONNX within-capsule reranking reduced Strict Any R@1 to `0.4006` and Strict Full R@10 to `0.1818`; its 352-query development replay took about `897 s` versus `239 s` for the base path, so it is rejected.
- Sentence-peak BM25 hit `154/352` at rank one versus `172/352` for chunk BM25 and is rejected.

### Integrity boundaries

- Scores are answer-bearing provenance retrieval proxies, not official end-to-end answer accuracy or leaderboard submissions.
- Accurate Retrieval and Conflict Resolution are scored; Test-Time Learning and Long-Range Understanding remain unscored.
- Answers are read only after ranking returns; formal JSON records `gold_used_for_ranking=false`.
- This is post-dataset development, not a statistically independent fresh blind test.
- Full runtime changed from `1218.85 s` to `942.26 s`. The observation may include warm model/filesystem caches and is not a guaranteed 22.7% speedup.
