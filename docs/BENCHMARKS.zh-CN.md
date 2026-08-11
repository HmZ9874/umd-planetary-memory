# UMD 公开成绩与限制（中文）

## 1. 诚实声明

以下数字是官方数据上的**来源/证据检索代理**。它们不是官方端到端答案准确率，也不是 benchmark 官方 leaderboard 提交。运行协议保证先检索、后读取答案或证据标签，正式 JSON 均记录 `gold_used_for_ranking=false`。

Final 指胶囊检索；Strict 指每个排名位置只包含一个来源的原子检索。二者不可混为同一个指标。

## 2. 正式结果快照

日期：2026-08-11。

| 基准 | 范围/可评估问题 | Final Any R@1 | Final Any R@10 | Final Full R@10 | Strict Any R@1 | Strict Any R@10 | Strict Full R@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| LoCoMo | 1,982 | 0.5605 | 0.9082 | 0.8219 | 0.2861 | 0.7089 | 0.6060 |
| LongMemEval_S | 470 | 0.9468 | 1.0000 | 0.9936 | 0.8979 | 0.9745 | 0.8021 |
| MemBench（分层代理） | 550 | 0.8200 | 0.9982 | 0.9418 | 0.5782 | 0.9509 | 0.6000 |
| Memora | 398 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.5905 |
| MemoryBench DialSim | 34 | 0.4118 | 0.8824 | 0.2647 | 0.2353 | 0.3235 | 0.0000 |
| MemoryArena reuse | 1,714 | 0.9994 | 1.0000 | 1.0000 | 0.9708 | 1.0000 | 1.0000 |
| MemoryAgentBench UMD 3.30 | 1,152 | 0.6727 | 0.9271 | 0.5712 | 0.5556 | 0.7812 | 0.2595 |
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

