# UMD 公开成绩与限制（中文）

## 1. 诚实声明

以下数字是官方数据上的**来源/证据检索代理**。它们不是官方端到端答案准确率，也不是 benchmark 官方 leaderboard 提交。运行协议保证先检索、后读取答案或证据标签，正式 JSON 均记录 `gold_used_for_ranking=false`。

Final 指胶囊检索；Strict 指每个排名位置只包含一个来源的原子检索。二者不可混为同一个指标。

## 2. 正式结果快照

日期：2026-08-11。

下表只统一报告字段，不统一任务难度；不同 benchmark 的 raw score 不能解释为共同 Elo 排名。

| 基准 | 范围/可评估问题 | Final Any R@1 | Final Any R@10 | Final Full R@10 | Strict Any R@1 | Strict Any R@10 | Strict Full R@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| LoCoMo | 1,982 | 0.5605 | 0.9082 | 0.8219 | 0.2861 | 0.7089 | 0.6060 |
| LongMemEval_S 3.28.2 冻结 | 470 | 0.9468 | 1.0000 | 0.9936 | 0.8979 | 0.9745 | 0.8021 |
| LongMemEval_S 3.34 参数冻结保留集 | 376 | 0.9415 | 1.0000 | 0.9920 | 0.9096 | 0.9840 | 0.9069 |
| LongMemEval_S 3.35 评估后回归 | 470 | 0.9745 | 1.0000 | 0.9936 | 0.9298 | 0.9872 | 0.9085 |
| LongMemEval_S 3.36 评估后回归 | 470 | **0.9979** | 1.0000 | 0.9936 | 0.9298 | 0.9872 | 0.9085 |
| MemBench（分层代理） | 550 | 0.8200 | 0.9982 | 0.9418 | 0.5782 | 0.9509 | 0.6000 |
| Memora | 398 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.5905 |
| MemoryBench DialSim | 34 | 0.4118 | 0.8824 | 0.2647 | 0.2353 | 0.3235 | 0.0000 |
| MemoryArena reuse | 1,714 | 0.9994 | 1.0000 | 1.0000 | 0.9708 | 1.0000 | 1.0000 |
| MemoryAgentBench UMD 3.33.1 | 1,152 | 0.8681 | 0.9705 | 0.7813 | 0.7639 | 0.8976 | 0.5200 |
| LongMemEval-V2 small | 230 | 0.9304 | 0.9870 | 0.5565 | 0.7652 | 0.9087 | 0.1174 |
| EverMemBench-Dynamic | 202 | 0.8465 | 0.9802 | 0.4604 | 0.4406 | 0.9010 | 0.2327 |

Memora 另有状态扇区 exact-set rate `0.9925`、旧值污染率 `0.0`，107 个确定性聚合答案验证为 `107/107`。

## 3. UMD 3.30 前后对照

MemoryAgentBench 30/30 上下文、2,800 问题；1,152 题能由冻结切块定位答案来源。

| 指标 | UMD 3.29 | UMD 3.30 | 绝对变化 |
|---|---:|---:|---:|
| Final MRR | 0.6357 | 0.7549 | +0.1192 |
| Final Any R@1 | 0.5078 | 0.6727 | +0.1649 |
| Final Full R@10 | 0.3429 | 0.5712 | +0.2283 |
| Strict Any R@1 | 0.3767 | 0.5556 | +0.1788 |
| Strict Full R@10 | 0.1484 | 0.2595 | +0.1111 |

Accurate Retrieval 指标逐项不变；提升来自 Conflict Resolution 的多跳与冲突事实。

## 3.1 UMD 3.30 → 3.31 证据闭包

| 指标 | UMD 3.30 | UMD 3.31 | 绝对变化 |
|---|---:|---:|---:|
| Final Any R@1 | 0.6727 | 0.7352 | +0.0625 |
| Final Full R@10 | 0.5712 | 0.6189 | +0.0477 |
| Strict Any R@1 | 0.5556 | 0.6398 | +0.0842 |
| Strict Full R@10 | 0.2595 | **0.4141** | **+0.1545** |

Strict Full R@10 相对提升 `59.5%`。Accurate Retrieval 不变；Conflict Resolution Strict Full R@10 从 `0.2737` 提升到 `0.4963`。全量运行时间增加 `6.5%`。

## 3.2 UMD 3.31 → 3.32.1 关系叠加

| 指标 | UMD 3.31 | UMD 3.32.1 | 绝对变化 |
|---|---:|---:|---:|
| Final Any R@1 | 0.7352 | **0.8281** | **+0.0929** |
| Final Full R@10 | 0.6189 | **0.7526** | **+0.1337** |
| Strict Any R@1 | 0.6398 | **0.7378** | **+0.0981** |
| Strict Full R@10 | 0.4141 | **0.5095** | **+0.0955** |

同容量 600-chunk 档达到 `0.9370 / 0.9701 / 0.9104 / 0.7065`，但不替代全量成绩。Strict 每个名次只能返回一个来源，所以全量 Strict Full R@10 的 raw 数据集上限为 `794/1152 = 0.6892`；四项 raw 指标同时达到 0.99 在当前定义下不可能。容量档的 600 题 Conflict Resolution 子集为 `0.7050 / 0.7450 = 0.9463` 上限归一化 Strict Full R@10。详见 `benchmarks/results/UMD3321_RELATION_SUPERPOSITION_REPORT.md`。

## 3.3 UMD 3.32.1 → 3.33.1 双文档恒星

四项核心指标分别从 `0.8281 / 0.7526 / 0.7378 / 0.5095` 提升到 `0.8681 / 0.7813 / 0.7639 / 0.5200`。子集预算下 Final Full R@10 实现上限约 `0.9531`，当前达到 `81.97%`；Strict Full 达到固定上限的 `75.44%`。详见 `benchmarks/results/UMD3331_BINARY_STAR_REPORT.md`。

### 3.4 UMD 3.34 LongMemEval 查询裂变

第 0–99 题是明确的后数据集开发回放（94 题可回答）；参数随后冻结，第 100–499 题作为 376 题可回答保留集。保留集 Strict Any R@1 为 `0.8936 → 0.9096`，Strict MRR 为 `0.9138 → 0.9371`，Strict Full R@10 为 `0.8218 → 0.9069`；Final Any R@1 和 Full R@10 保持 `0.9415 / 0.9920`。500 题混合回放不是盲测，单独披露。详见 `benchmarks/results/UMD334_QUERY_FISSION_REPORT.md`。

### 3.5 UMD 3.35 后台幽灵物质

500 题结果是评估后回归，不是盲测。相对 3.34，Final Any R@1 `0.9468 → 0.9745`、Final Full R@1 `0.7383 → 0.8426`、Strict Any R@1 `0.9213 → 0.9298`，而 Final/Strict Full R@10 守恒为 `0.9936 / 0.9085`。逐题比较得到 Final R@1 改善 13、回退 0，Strict R@1 改善 4、回退 0。详见 `benchmarks/results/UMD335_GHOST_MATTER_REPORT.md`。

### 3.6 UMD 3.36 有界幽灵星座

相对 3.35，Final Any R@1 为 `0.9745 → 0.9979`，Final Full R@1 为 `0.8426 → 0.9851`。470 题逐题审计得到 Final Full@1 改善 67、回退 0；Final Full@10 和所有 Strict 指标逐题零变化，Final@10 平均字符数完全相同。该结果仍属于评估后回归。详见 `benchmarks/results/UMD336_GHOST_CONSTELLATION_REPORT.md`。

## 4. 范围与偏差

- LoCoMo：完整 10 段对话，1,986 问中 4 问无证据标签；评分 1,982 问。
- LongMemEval：500 问中 470 个可回答问题进入检索分数，30 个拒答问题需要答案模型判断。
- MemBench：从 55 个官方组各取前 10 条，共 550 条；不是本地 26,637 条全量端到端运行。
- MemoryAgentBench：只计 Accurate Retrieval 与 Conflict Resolution；Test-Time Learning、Long-Range Understanding 未计分。
- MemoryBench：只报告 DialSim 答案承载会话检索；不含官方持续学习/回答质量。
- MemoryArena：报告前序子任务答案组件复用；不是环境成功率，且 1,714 个可评估问题中 1,599 个来自 group travel planner。
- EverMemBench：3,121 问中只有 202 问能由当前固定日组匹配器定位来源，因此 0.8465 不能解释为全部问题答案准确率。
- LongMemEval-V2：只运行 small tier；官方 reader accuracy/LAFS 未运行。
- EvolMem：本地审计时上游仓库仍标注 under construction，未产生分数。

## 5. 数据与可复核性

正式 JSON 包含问题数、证据数、运行时间、数据 SHA-256（适用时）、编码器元数据和 gold 使用声明。仓库不重新分发 benchmark 数据；请从上游获取并遵守其许可。

上游示例：

- LoCoMo: `snap-research/locomo`
- LongMemEval cleaned: `xiaowu0162/longmemeval-cleaned`
- MemoryAgentBench: <https://github.com/HUST-AI-HYZ/MemoryAgentBench>
- EverMemBench: <https://github.com/EverMind-AI/EverMemBench>
- MemoryArena: <https://github.com/ZexueHe/MemoryArena>
- MemoryBench: <https://github.com/THUIR/MemoryBench>
- MemBench: <https://github.com/import-myself/Membench>
- Memora: <https://github.com/geniesinc/Memora>

## 6. 尚未证明

- 官方端到端回答准确率；
- 与商业记忆服务在同一硬件、同一 LLM、同一费用下的公平比较；
- 七天以上真实用户故障率、成本和质量漂移；
- 同行评审结论；
- 生产安全认证。
