# UMD 3.30 引力弹弓优化报告

日期：2026-08-11

## 结论

UMD 3.30 修复了 MemoryAgentBench 中最主要的三跳知识链缺陷。新算法只在密集编号事实宇宙中启用；LoCoMo 等普通对话继续使用冻结控制轨道。

全量 MemoryAgentBench 检索代理结果：

| 指标 | UMD 3.29 | UMD 3.30 | 绝对变化 |
|---|---:|---:|---:|
| Final MRR | 0.6357 | 0.7549 | +0.1192 |
| Final Any R@1 | 0.5078 | 0.6727 | +0.1649 |
| Final Any R@10 | 0.8585 | 0.9271 | +0.0686 |
| Final Full R@10 | 0.3429 | 0.5712 | +0.2283 |
| Strict Any R@1 | 0.3767 | 0.5556 | +0.1788 |
| Strict Any R@10 | 0.7222 | 0.7812 | +0.0590 |
| Strict Full R@10 | 0.1484 | 0.2595 | +0.1111 |

相对提升：Final Any R@1 提升 32.5%，Final Full R@10 提升 66.6%，Strict Any R@1 提升 47.5%，Strict Full R@10 提升 74.9%。

## 根因

旧检索器能找到第一跳，例如：

`Bagratuni Dynasty → religion → Christianity`

但问题要求的是第二跳或第三跳：

`Christianity → founded in city → Taipei`

类似失败还包括：

- 球队 → 运动 → 起源国家 → 洲；
- 组织 → 主席 → 职业；
- 人物 → 国籍国家 → 国家元首；
- 作品 → 表演者 → 子女 → 国籍 → 官方语言。

扩大普通 BM25 或神经候选池无法可靠跨越这些词面不相交的关系。

## UMD 3.30 物理定律

### 1. 版本化事实行星图

编号自然语言事实被编译为：

`e = (subject, relation, object, ordinal, provenance)`

同一主体和关系的当前边为可见前缀内序号最大的事实：

`e_current(s,r,t) = argmax ordinal(e), source(e) < t`

因此冲突解决不删除历史，只让最新可见事实产生更强的当前引力。

### 2. 三跳引力弹弓

问题中出现的最长图主体成为发射恒星。搜索最多三跳；每条路径只使用当前事实边。关系与问题线索接触越强，路径能量越高；到达问题要求的目标关系时增加终点势能。

`E(path) = Σ(0.20 + min(3, contact(q,r)/5)) + 6·I[r_last ∈ target(q)]`

最高能路径的答案事实来源被提升到原子首位。

### 3. 回声小行星带

弹弓得到的对象是模型根据记忆图预测的答案，不是标准答案。所有包含该预测对象的来源构成回声集合：

`Echo(a) = {source_i | casefold(a) ⊆ casefold(text_i)}`

回声来源作为卫星加入十个胶囊，预算上限为 96。它不替换原稳定来源，因此 Any Recall 单调不减，同时显著提高重复与分散证据的 Full Recall。

## 分任务结果

Accurate Retrieval 的全部指标与 UMD 3.29 逐项相同，说明非图任务没有回退。

Conflict Resolution：

| 指标 | UMD 3.29 | UMD 3.30 |
|---|---:|---:|
| Final Any R@1 | 0.4438 | 0.6813 |
| Final Any R@10 | 0.7650 | 0.9300 |
| Final Full R@10 | 0.3113 | 0.6400 |
| Strict Any R@1 | 0.3400 | 0.5975 |
| Strict Full R@10 | 0.1138 | 0.2737 |

## 性能与内存

- UMD 3.29 全量运行：972.3 秒。
- UMD 3.30 全量运行：1,144.8 秒，增加 17.7%。
- 最大已测结构化上下文：1,119 个块、17,831 条事实。
- 事实图构建：0.652 秒。
- 事实图稳态 Python 内存：5.92 MiB。
- 构建瞬时峰值：15.03 MiB。
- 没有增加全量神经向量常驻索引，也没有稠密邻接矩阵。

## 防回退与无作弊验证

- 72 项单元/回归测试通过。
- 开发上下文、未参与调参的留出上下文和原本强基线上均提升。
- Accurate Retrieval 全量指标保持不变。
- LoCoMo 样本的事实图未激活，前十问 final/atomic 排名逐项完全相同。
- 排名在读取答案前完成，`gold_used_for_ranking=false`。
- 成绩仍是答案承载来源检索代理，不是官方端到端回答准确率。

## 文件

- `umd330_exam_memoryagentbench_full.json`
- `umd330_memoryagentbench_checkpoint.json`
- `../umd330_adapter.py`
- `../umd330_adapter_test.py`
