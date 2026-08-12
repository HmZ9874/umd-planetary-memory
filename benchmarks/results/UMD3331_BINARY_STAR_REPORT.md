# UMD 3.33.1 Binary Document Stars / UMD 3.33.1 双文档恒星

## 中文

### 全量结果

UMD 3.33.1 在 MemoryAgentBench 相同 30/30 上下文、2,800 问题与 1,152 个可定位来源问题上运行。排序只接收查询和可见文本；正式 JSON 记录 `gold_used_for_ranking=false`。

| 指标 | UMD 3.32.1 | UMD 3.33.1 | 绝对变化 |
|---|---:|---:|---:|
| Final Any R@1 | 0.8281 | **0.8681** | +0.0399 |
| Final Full R@10 | 0.7526 | **0.7813** | +0.0286 |
| Strict Any R@1 | 0.7378 | **0.7639** | +0.0260 |
| Strict Full R@10 | 0.5095 | **0.5200** | +0.0104 |

Conflict Resolution 达到 Final Any R@1 `0.9225`、Final Full R@10 `0.9213`、Strict Any R@1 `0.8863`、Strict Full R@10 `0.6175`。Accurate Retrieval 达到 `0.7443 / 0.4631 / 0.4858 / 0.2983`。

### 新机制

1. **重叠窗口重建**：只在大量窗口具有一致精确重叠且检测到至少 20 个 `Document N:` 边界时激活。
2. **文档恒星**：完整文档作为恒星参与 BM25，覆盖它的固定窗口作为不可变来源卫星。
3. **双恒星焦点**：Final 首胶囊保留原 3.32.1 来源，并追加排名前两篇文档的少量卫星；不删除旧命中。
4. **严格置信门**：只有第一文档分数至少为第二文档的 `1.13×` 时，才把第一文档内最强原子来源提升到 Strict 首位。
5. **终点吸收边界**：路径覆盖全部查询支持的终点关系后停止扩展，避免 `citizen → head_government → citizen` 等越过正确答案的环路。
6. **循环与重复关系能量**：拒绝实体回环，并对路径中重复关系每次减 `1.25`。

### 上限距离

UMD 3.33.1 普通文档路径前十个 Final 胶囊最多约 88 个来源，结构化闭包路径最多约 296 个来源。按各子集的实际预算计算，Final Full R@10 的 size-only 实现上限约为 `1098/1152 = 0.9531`，当前达到该上限的 `81.97%`。Strict Full R@10 的上限仍为 `794/1152 = 0.6892`，当前达到 `75.44%`。Any R@1 的来源数量上限均为 1。

### 被否决实验

把每个候选文档的最强来源提前组成 Strict“恒星梯”，使 RULER 200 题 Strict Any R@10 从 `0.850` 微升到 `0.855`，但 Strict Full R@10 从 `0.355` 降至 `0.300`，因此未进入正式算法。

### 成本与边界

- 全量任务由检查点分两段完成，总墙钟约 `1301.1 s`；JSON 同时保留最后恢复段 `100.86 s`，后者不能单独视为全量运行时间。
- 3.32.1 同机观测为 `942.26 s`，因此本轮精度提升付出了约 `38%` 的墙钟代价；缓存和后台负载使该比较不是稳定性能保证。
- 本轮属于已经查看数据后的开发优化，不是统计独立盲测，也不是官方端到端答案准确率或 leaderboard 提交。

## English

### Full result

UMD 3.33.1 runs the same 30/30 MemoryAgentBench contexts, 2,800 questions, and 1,152 answer-bearing-source questions. Ranking receives only query and visible text; the formal JSON records `gold_used_for_ranking=false`.

| Metric | UMD 3.32.1 | UMD 3.33.1 | Absolute delta |
|---|---:|---:|---:|
| Final Any R@1 | 0.8281 | **0.8681** | +0.0399 |
| Final Full R@10 | 0.7526 | **0.7813** | +0.0286 |
| Strict Any R@1 | 0.7378 | **0.7639** | +0.0260 |
| Strict Full R@10 | 0.5095 | **0.5200** | +0.0104 |

Conflict Resolution reaches `0.9225 / 0.9213 / 0.8863 / 0.6175`; Accurate Retrieval reaches `0.7443 / 0.4631 / 0.4858 / 0.2983` in the same metric order.

### New mechanisms

1. Overlap reconstruction activates only for a regular exact-overlap stream with at least 20 `Document N:` boundaries.
2. Complete documents become stars; immutable fixed windows overlapping them are source moons.
3. The Final first capsule conserves the 3.32.1 sources and appends the small moon sets of the top two document stars.
4. Strict promotes the best atomic moon of the first star only when its document score is at least `1.13×` the runner-up.
5. Complete-terminal absorption stops expansion after all query-supported terminal relations are covered.
6. Entity cycles are rejected and each repeated path relation loses `1.25` energy.

### Distance to ceilings

The ordinary document path carries at most about 88 sources in Final top ten; the structured closure path carries about 296. The subset-aware Final Full R@10 size ceiling is approximately `1098/1152 = 0.9531`, of which the observed result reaches `81.97%`. The Strict Full R@10 ceiling remains `794/1152 = 0.6892`; the result reaches `75.44%`. Both Any R@1 source-count ceilings are one.

### Rejected experiment

A Strict star ladder moved the best source from every candidate document forward. It raised RULER-200 Strict Any R@10 from `0.850` to `0.855` but reduced Strict Full R@10 from `0.355` to `0.300`, so it is excluded.

### Cost and boundaries

- The checkpointed full run used approximately `1301.1 s` total wall clock. The JSON also retains the `100.86 s` final-resume segment, which is not a standalone full runtime.
- The same-machine 3.32.1 observation was `942.26 s`, so this accuracy gain cost roughly `38%` more wall time; cache and background-load differences prevent a stable speed claim.
- This is post-dataset development, not an independent blind test, official end-to-end answer accuracy, or a leaderboard submission.
