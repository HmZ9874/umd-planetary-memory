# UMD 3.28.2 冻结考试报告

日期：2026-08-11

后续新增的 MemoryArena、LongMemEval-V2、MemoryBench 与规模超时结果见 `UMD3282_CONTINUED_EXAM_REPORT.md`。

## 结论

本轮列出的 10 个基准中，4 个具备本地数据和现有适配器并已实际执行；其余 6 个没有本地官方数据、运行框架或所需模型端点，因此标记为“未运行”，没有制造分数，也没有用相似测试代替。

所有有效结果均使用冻结的 UMD 3.28.2 参数。LoCoMo、LongMemEval 和 MemBench 的代码路径在排序返回后才读取 evidence 标签。Memora 审计时发现旧适配器虽然未把标签传给排序器，却提前把标签读入了局部变量；首次结果因此作废。修正评测顺序后使用全新检查点重跑，结果与首次运行一致。

## 十项状态

| # | 基准 | 本轮状态 | 范围 | 核心结果 |
|---:|---|---|---|---|
| 1 | LongMemEval | 已运行 | cleaned S 全 500 题；470 个可回答题计主检索分 | 最终胶囊 Any R@1 0.9468、Full R@10 0.9936；严格原子 Any R@1 0.8979、Full R@10 0.8021 |
| 2 | LoCoMo | 已运行 | 10 个完整对话，1,986 问；1,982 问有 evidence | 最终胶囊 Any R@1 0.5605、Full R@10 0.8219；严格原子 Any R@1 0.2861、Full R@10 0.6060 |
| 3 | MemoryAgentBench | 未运行 | 本地无官方任务数据、代理学习循环和评测器 | 不给分 |
| 4 | MemoryArena | 未运行 | 本地无交互环境与 agent-environment 执行框架 | 不给分 |
| 5 | MemBench | 已运行，分层检索代理 | 官方本地 55 组各取前 10 条，共 550/26,637 trajectories | 最终胶囊 Any R@1 0.8200、Full R@10 0.9418；严格原子 Any R@1 0.5782、Full R@10 0.6000 |
| 6 | LongMemEval-V2 | 未运行 | 本地无 V2 数据、reader/controller/embed/judge 端点 | 不给分 |
| 7 | MemoryBench | 未运行 | 本地无其 28 数据集套件、反馈模拟器和模型评测环境 | 不给分 |
| 8 | EverMemBench | 未运行 | 本地无官方百万 token 数据与适配器 | 不给分 |
| 9 | Memora | 已运行，本地全量检索代理 | 30 persona-period，600 问；398 个 memory-presence 查询 | 状态集合 Full R@1 1.0000；严格来源 Any R@1 1.0000、Full R@1 0.1055；状态 exact-set 0.9925；旧值污染 0.0000 |
| 10 | EvolMem | 未运行 | 本地无官方代码、数据与演化记忆评测环境 | 不给分 |

## 指标解释

- “最终胶囊”是 UMD 行星聚合后的排名单位，一个胶囊可含多个来源。因此它的 R@1 不是传统的“单来源 top-1”。
- “严格原子”每个排名单位只含一个来源，更接近普通 retriever 的 R@k，通常也更难。
- Any R@k 表示前 k 个排名单位至少命中一个 gold 来源；Full R@k 表示 gold 来源全部被前 k 个单位覆盖。
- 这些是 evidence retrieval 分，不是官方端到端问答、agent 成功率或 LLM-judge 分。没有答案模型时，不把 retrieval 分冒充官方总分。
- Memora 的状态集合 Full R@1=1.0 只测召回完整性；其额外来源由 micro precision=0.9997 和 exact-set=0.9925 单独约束。严格来源 Full R@1=0.1055 等于该数据在单来源 top-1 下的理论上限。

## 主要弱点

1. LoCoMo 是当前最明显短板。严格 Any R@1 只有 0.2861；类别 3 的最终 Any R@1 0.2500、Full R@10 0.3696，是最弱分组。
2. UMD 聚合在 LongMemEval 上效果很强，但聚合掩盖了严格来源排序不足：multi-session 的最终 Full R@10=1.0000，而严格 Full R@10=0.5289。
3. MemBench 的最终胶囊覆盖较好，但严格 Full R@1=0.0582、Full R@10=0.6000，说明多步轨迹的原子证据排序仍弱。
4. Memora 的旧值污染为零，但 session-level forgetting contamination@10=0.3889；其中不可约的混合会话下限为 0.3485，扣除该下限后仍有约 0.0404 的可优化空间。
5. 尚不能给出十基准横向总排名，因为 6 项未运行，且已运行的 4 项也不是统一的官方端到端指标。

## 防作弊与完整性审计

- 参数冻结：`neural_candidate_pool=64`；physics v3.16/v3.17/v3.25/v3.26/v3.27 开启；first-orbit v3.16 关闭。
- LoCoMo 数据 SHA-256：`79FA87E90F04081343B8C8DEBECB80A9A6842B76A7AA537DC9FDF651EA698FF4`。
- LongMemEval 数据 SHA-256：`D6F21EA9D60A0D56F34A05B609C79C88A451D2AE03597821EA3D5A9678C3A442`。
- LongMemEval 检查点 500/500；LoCoMo 检查点 10/10。
- UMD 适配器回归：66 passed。
- 本地嵌入模型：`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`，384 维；所有有效运行 fallback_calls=0。
- 付费 API 调用：0；答案模型：无；judge 模型：无；测试中参数修改：无。

## 产物

- 协议：`benchmarks/umd3282_exam_protocol.json`
- 考试入口：`benchmarks/umd3282_frozen_exam.py`
- LongMemEval：`benchmarks/results/umd3282_exam_longmemeval.json`
- LoCoMo：`benchmarks/results/umd3282_exam_locomo.json`
- MemBench：`benchmarks/results/umd3282_exam_membench.json`
- Memora：`benchmarks/results/umd3282_exam_memora.json`

## 未运行基准的官方入口

- MemoryAgentBench: https://github.com/HUST-AI-HYZ/MemoryAgentBench
- MemoryArena: https://memoryarena.github.io/
- LongMemEval-V2: https://github.com/xiaowu0162/LongMemEval-V2
- MemoryBench: https://github.com/THUIR/MemoryBench
- EverMemBench: https://arxiv.org/abs/2602.01313
- EvolMem: https://github.com/shenye7436/EvolMem
