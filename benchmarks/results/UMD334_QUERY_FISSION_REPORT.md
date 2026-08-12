# UMD 3.34 Query Fission and Episodic Nucleus / 查询裂变与用户陈述核

## 中文

UMD 3.34 针对上一轮的两个主要瓶颈：复合查询覆盖，以及 LongMemEval 风格开放文本的严格原子首位。算法只读取问题与可见来源；答案和 evidence ID 在检索完成后才由评估器读取。

### 机制

1. 完整问题作为中央恒星，最多四个真实子句作为子引力场；短名词并列不拆分。
2. 为第一人称记忆问题生成至多一个确定性声明式改写。
3. 候选发现采用原问题、子句 BM25 和用户陈述核 BM25 的有界轮转，最多 96 个来源。
4. 对带 `user:`/`assistant:` 角色的会话，仅用用户陈述构成“生物核”；助手推荐仍保留在完整证据和审计文本中，但不主导严格首位。
5. 严格首位质量为 `0.50` 用户语义、`0.30` 用户词汇、`0.12` 全文语义、`0.08` 全文词汇；只有首/次比至少 `1.08`、边际至少 `0.055` 且用户词汇不冲突时才提升。
6. 复合问题从每个子场轮转选证据；旧首胶囊来源全部保留，所以 Final Any R@1 不会因追加而下降。

### 开发/保留集协议

- 前 100 题是明确的后数据集开发回放，其中 94 题可回答；用于选择权重和门限。
- 第 100–499 题是参数冻结后的保留集，其中 376 题可回答；参数在读取该切片结果前冻结。
- 这不是官方端到端答案准确率或 leaderboard 提交。
- 完整 500 题汇总混合了开发与保留题，只用于描述总体行为，不冒充盲测。

### 参数冻结保留集（376 题）

| 指标 | UMD 3.28.2 | UMD 3.34 | 绝对变化 |
|---|---:|---:|---:|
| Final Any R@1 | 0.9415 | 0.9415 | 0.0000 |
| Final Full R@10 | 0.9920 | 0.9920 | 0.0000 |
| Strict MRR | 0.9138 | **0.9371** | **+0.0233** |
| Strict Any R@1 | 0.8936 | **0.9096** | **+0.0160** |
| Strict Any R@10 | 0.9707 | **0.9840** | **+0.0133** |
| Strict Full R@10 | 0.8218 | **0.9069** | **+0.0851** |
| Strict Micro R@10 | 0.8636 | **0.9435** | **+0.0799** |

### 500 题混合回放（470 个可回答问题）

Strict Any R@1 从 `0.8979` 提升到 `0.9213`，Strict MRR 从 `0.9186` 提升到 `0.9465`，Strict Full R@10 从 `0.8021` 提升到 `0.9085`，Strict Micro R@10 从 `0.8404` 提升到 `0.9404`。Final Any R@1、Final Any R@10 和 Final Full R@10 保持 `0.9468 / 1.0000 / 0.9936`。

### 成本和边界

每个独立会话索引额外编码用户陈述核，候选预算从 64 增至最多 96。开发 100 题冷启动约 `191.7 s`；保留集分两段运行，外部 600 秒时限前完成至第 400 题，续跑最后 100 题约 `208.4 s`。检查点续跑时间不能解释为完整保留集总时间。用户陈述核向量属于当前索引，索引销毁后释放；长期元数据仍不常驻 RAM。

## English

UMD 3.34 targets compound-query coverage and strict rank-one retrieval in LongMemEval-style open dialogue. Retrieval receives only the query and visible sources; answers and evidence IDs remain evaluator-only.

### Mechanism

The complete question is conserved as a central star and up to four genuine clauses become bounded subfields. One deterministic declarative rewrite is allowed for first-person memory questions. Candidate discovery round-robins full-query, clause, and user-nucleus BM25 fields up to 96 sources. In role-marked dialogue, assistant expansions remain in the evidence/audit text but strict periapsis is scored primarily from user statements:

`P_i = 0.50 S_user(i) + 0.30 L_user(i) + 0.12 S_full(i) + 0.08 L_full(i)`.

Rank one changes only at a `1.08` first/second ratio, `0.055` margin, and non-conflicting user-lexical contact. Compound-query moons append to the conserved first Final capsule.

### Development/holdout protocol

Questions 0–99 form an explicitly post-dataset development replay (94 answerable). Questions 100–499 form a parameter-frozen holdout (376 answerable). The combined 500-question number mixes both partitions and is not represented as a blind or official leaderboard result.

### Frozen holdout

| Metric | UMD 3.28.2 | UMD 3.34 | Absolute delta |
|---|---:|---:|---:|
| Final Any R@1 | 0.9415 | 0.9415 | 0.0000 |
| Final Full R@10 | 0.9920 | 0.9920 | 0.0000 |
| Strict MRR | 0.9138 | **0.9371** | **+0.0233** |
| Strict Any R@1 | 0.8936 | **0.9096** | **+0.0160** |
| Strict Any R@10 | 0.9707 | **0.9840** | **+0.0133** |
| Strict Full R@10 | 0.8218 | **0.9069** | **+0.0851** |
| Strict Micro R@10 | 0.8636 | **0.9435** | **+0.0799** |

Across the mixed 470-answerable replay, Strict Any R@1 reaches `0.9213`, Strict MRR `0.9465`, Strict Full R@10 `0.9085`, and Strict Micro R@10 `0.9404`. Final `Any R@1 / Any R@10 / Full R@10` remains `0.9468 / 1.0000 / 0.9936`.

The method adds a current-index user-nucleus vector cache and widens bounded discovery from 64 to at most 96 sources. It does not restore historical metadata to resident RAM. These are retrieval proxies, not official end-to-end answer accuracy.
