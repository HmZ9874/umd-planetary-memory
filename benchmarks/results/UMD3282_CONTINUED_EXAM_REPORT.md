# UMD 3.28.2 续测报告

日期：2026-08-11

## 本轮结论

十项列表中，目前 7 项产生了可审计的冻结检索指标；其中新增完成 MemoryArena、LongMemEval-V2 容量档和 MemoryBench DialSim 时间会话代理。MemoryAgentBench 与 EverMemBench 暴露严重吞吐瓶颈，只有部分检查点，不计算总分。EvolMem 官方仓库尚无公开数据，仅有生成与 LLM-judge 脚本，因此不能运行。

所有新增分数都是 retrieval proxy，不是官方端到端 Accuracy、agent-environment success、reader accuracy、LAFS 或 LLM-judge 分。

## 十项最新状态

| # | 基准 | 最新状态 | 结果摘要 |
|---:|---|---|---|
| 1 | LongMemEval | 已完成全量检索 | 最终 Any R@1 0.9468、Full R@10 0.9936；严格 Any R@1 0.8979、Full R@10 0.8021 |
| 2 | LoCoMo | 已完成全量检索 | 最终 Any R@1 0.5605、Full R@10 0.8219；严格 Any R@1 0.2861、Full R@10 0.6060 |
| 3 | MemoryAgentBench | 无总分，规模/查询超时 | 全量检查点 3/30；≤600 chunk 容量档 1/18；部分记录不汇总 |
| 4 | MemoryArena | 官方数据复用检索代理完成 | 4,149 顺序查询，1,714 个可验证复用查询；最终 Any R@1 0.9994、Full R@10 1.0000；严格 Any R@1 0.9708、Full R@10 1.0000 |
| 5 | MemBench | 550 条分层检索代理完成 | 最终 Any R@1 0.8200、Full R@10 0.9418；严格 Any R@1 0.5782、Full R@10 0.6000 |
| 6 | LongMemEval-V2 | small 容量档完成 | 100/451 均匀抽样，53 个 answer-bearing 查询；最终 Any R@1 0.9245、Full R@10 0.5660；严格 Any R@1 0.8113、Full R@10 0.1132 |
| 7 | MemoryBench | DialSim 时间会话代理完成 | 51/51；最终 Any R@1 0.5098、Any R@10 1.0000、Full R@10 0.7059；严格 Any R@1 0.1176、Any R@10 0.4706 |
| 8 | EverMemBench | 无总分，长会话吞吐超时 | 全量 0/5 用户；每用户均匀 50 问容量档 1/5；部分记录不汇总 |
| 9 | Memora | 本地 600 问检索代理完成 | 状态 Full R@1 1.0000；严格 Any R@1 1.0000、Full R@1 0.1055；exact-set 0.9925；旧值污染 0 |
| 10 | EvolMem | 无法运行 | 官方仓库 under construction，无数据；生成和评测均要求 API/model |

## 新增测试解释

### MemoryArena

- 使用官方 701 个任务、五类环境。
- 每完成一个子任务，将其问题和成功 observation 写入记忆；下一子任务先检索，再读取当前答案并判断哪些历史 observation 含可复用组件。
- 4,149 个顺序查询中只有 1,714 个具备可验证组件复用；其余不计零分。
- 1,599/1,714 个 eligible 查询来自 group travel planner，因此总分高度受该环境支配，不能视为官方 MemoryArena success。

### LongMemEval-V2

- 下载并核验官方 1,195,604,539-byte trajectories 文件；small tier 有 451 问、两个共享 haystack、共 200 trajectories。
- 完整 accessibility-tree 直接索引在首个 haystack 超时，因此该运行作废、无分。
- 容量档在每个 haystack 均匀抽 50 问；索引 gold-independent 的 goal/action/thought 经验摘要，完整 147,857,357 字符轨迹只在检索返回后判定 answer-bearing relevance。
- 100 问中 53 问有可逐字验证的 answer-bearing trajectory。严格 Full R@10 仅 0.1132，说明检索到至少一个相关轨迹不难，但完整覆盖相关经验很弱。

### MemoryBench

- 初始 answer-bearing 规则会把 `yes`、`okay` 等高频答案映射到大量伪 gold，会话结果作废并保留作诊断，不计成绩。
- 有效测试改用问题文本明确给出的历史日期，不读取 hidden golden answer；gold 是该日期的全部会话。
- 最终胶囊 Any R@10=1.0，但严格 Any R@1=0.1176、Full R@10=0.0196，显示行星聚合覆盖强、原子时间会话排序弱。

## 规模与性能失败

1. MemoryAgentBench 第 3 个全量 context 约 2,307 chunks，构图超过约 20 分钟窗口；全量只封存 3/30。
2. 即使限制到 ≤600 chunks，首个约 286-node、100-query context 仍接近十分钟；容量档只封存 1/18。
3. EverMemBench 每用户约 10,000 messages。聚合为约 254 个日期×群组会话后，单用户全量 626 问仍超时；50 问容量档也接近十分钟/用户。
4. LongMemEval-V2 的 100 个完整 raw-trajectory 节点包含约 74–89 MB 文本/域，首 haystack 超时；改用经验摘要后 100 问在 122.81 秒内完成。

这些结果说明下一步优先级应是索引和查询复杂度，而不是继续增加召回公式：需要稀疏近邻构图、物理特征预计算、按层惰性展开、长节点摘要缓存和批量查询。

## 防作弊审计

- 新增适配器在检索返回后才读取当前答案或派生 relevance。
- MemoryBench temporal 测试只使用 query 自身公开日期，不读取 hidden golden answer。
- 容量样本按大小阈值或均匀位置选择，不按答案和分数选择。
- 中途超时检查点不计算总分。
- 付费 API、answer model、judge model 均为 0。
- 所有有效运行 `fallback_calls=0`。
- UMD 回归：66 passed。

## 新增产物

- 协议：`benchmarks/umd3282_continued_protocol.json`
- 续测入口：`benchmarks/umd3282_continued_exam.py`
- MemoryArena：`benchmarks/results/umd3282_exam_memoryarena.json`
- LongMemEval-V2：`benchmarks/results/umd3282_exam_longmemeval_v2_capacity.json`
- MemoryBench temporal：`benchmarks/results/umd3282_exam_memorybench_temporal.json`
- MemoryAgentBench 部分检查点：`benchmarks/results/umd3282_memoryagentbench_checkpoint.json`
- EverMemBench 部分检查点：`benchmarks/results/umd3282_evermembench_capacity_50_checkpoint.json`

