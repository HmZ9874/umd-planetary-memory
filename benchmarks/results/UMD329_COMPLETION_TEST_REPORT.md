# UMD 3.29 补充全量测试报告

日期：2026-08-11

## 测试协议

- 检索器：UMD 3.29 性能实现，保持 UMD 3.28.2 冻结检索语义和参数。
- 排名发生在读取答案或证据标签之前。
- `gold_used_for_ranking=false`，没有付费 API、答案模型或裁判模型。
- 下列成绩是官方数据上的答案承载来源检索代理，不是官方端到端回答准确率。
- 只对能通过固定、无标签切块在记忆文本中定位答案的题目计算检索分数；未定位题目不进入指标分母。

## 本轮新增全量结果

| 基准 | 全部问题 | 可评估问题 | Any R@1 | Any R@10 | Full R@10 | 运行时间 |
|---|---:|---:|---:|---:|---:|---:|
| MemoryAgentBench | 2,800 | 1,152 | 0.5078 | 0.8585 | 0.3429 | 972.3 秒 |
| LongMemEval-V2 small | 451 | 230 | 0.9304 | 0.9870 | 0.5565 | 113.3 秒 |
| EverMemBench-Dynamic | 3,121 | 202 | 0.8465 | 0.9802 | 0.4604 | 2,172.6 秒 |

严格单来源原子轨道结果：

| 基准 | Any R@1 | Any R@10 | Full R@10 |
|---|---:|---:|---:|
| MemoryAgentBench | 0.3767 | 0.7222 | 0.1484 |
| LongMemEval-V2 small | 0.7652 | 0.9087 | 0.1174 |
| EverMemBench-Dynamic | 0.4406 | 0.9010 | 0.2327 |

## 已有有效结果核验

| 基准 | 状态 | Any R@1 | Any R@10 | Full R@10 |
|---|---|---:|---:|---:|
| MemoryBench DialSim | 51/51 问题检索代理完成；34 题可评估 | 0.4118 | 0.8824 | 0.2647 |
| MemoryArena | 4,149 个连续问题完成；1,714 个复用证据问题可评估 | 0.9994 | 1.0000 | 1.0000 |

MemoryArena 的 1,714 个可评估问题中有 1,599 个来自 group travel planner，分布高度集中，因此不能把总分解释为五种环境上的均衡能力。

## 十项测试当前状态

| 项目 | 当前状态 |
|---|---|
| LongMemEval | 已完成正式数据检索测试 |
| LoCoMo | 已完成正式数据检索测试 |
| MemoryAgentBench | Accurate Retrieval 与 Conflict Resolution 全量检索代理完成；Test-Time Learning、Long-Range Understanding 尚未计分 |
| MemoryArena | 官方数据复用检索代理完成；未运行代理环境成功率 |
| MemBench | 已完成 |
| LongMemEval-V2 | small tier 451/451 全量完成 |
| MemoryBench | DialSim 51/51 全量检索代理完成；未运行回答质量/持续学习官方指标 |
| EverMemBench | 5/5 用户、3,121/3,121 问题全量完成 |
| Memora | 已完成 |
| EvolMem | 官方仓库仍标注 under construction，当前没有可执行数据集 |

## 主要结论

1. UMD 在 LongMemEval-V2 与 EverMemBench 的 Any R@10 已达到约 0.98，但完整证据集回忆仍明显较低。
2. MemoryAgentBench 是当前更明确的弱项，尤其是 Full R@10=0.3429；严格原子 Full R@10=0.1484。
3. 胶囊轨道与严格原子轨道差距很大，说明当前高分相当依赖多来源胶囊压缩，而单来源首位排序仍需优化。
4. EverMemBench 只有 202/3,121 题能由当前固定日组匹配器找到答案来源，覆盖率较低；0.8465 不能代表全部 3,121 题的端到端正确率。

## 结果文件

- `umd329_exam_memoryagentbench_full.json`
- `umd329_exam_longmemeval_v2_full.json`
- `umd329_exam_evermembench_full.json`
- `umd3282_exam_memorybench.json`
- `umd3282_exam_memoryarena.json`

回归验证：69 项测试通过，Python 编译检查通过。
