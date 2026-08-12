# UMD 3.36 Bounded Ghost Constellations / 有界幽灵星座

## 中文

UMD 3.36 在 3.35 的来源安全幽灵上增加“零扩容潮汐迁移”。查询先估计证据多重性，并检测同一事实在多个会话中的版本共振；幽灵轨道与 Strict 原子轨道随后形成双轨共识。只有已经存在于迁移前 Final@10 来源并集的原始 `source_id` 才能进入首胶囊，因此 Final@10 来源集合和字符成本严格守恒。

边界条件：

- 显式列表查询的幽灵预算上限为 12；时间查询 10；复合/共振 6；演化 5；偏好 2；
- 原子锚点最多 6 个，并与幽灵来源去重；
- 迁移使用 `Final₁ ← Final₁ ∪ M`，不删除旧首胶囊来源；
- Strict 从迁移前胶囊快照计算，因此 3.36 的打包不会改变任何 Strict 名次；
- 幽灵文本仍不是证据，所有结果回锚到原始来源；
- 排序不读取 gold，结果是评估后回归，不是盲测或官方端到端答案分数。

LongMemEval_S 500 题中有 470 题可回答：

| 指标 | UMD 3.35 | UMD 3.36 | 变化 |
|---|---:|---:|---:|
| Final MRR | 0.9834 | **0.9981** | +0.0147 |
| Final Any R@1 | 0.9745 | **0.9979** | +0.0234 |
| Final Full R@1 | 0.8426 | **0.9851** | +0.1426 |
| Final Micro R@1 | 0.8944 | **0.9899** | +0.0955 |
| Final Full R@10 | 0.9936 | **0.9936** | 0 |
| Strict Any R@1 | 0.9298 | **0.9298** | 0 |
| Strict Full R@10 | 0.9085 | **0.9085** | 0 |

470 题逐题审计：Final Any R@1 改善 11、回退 0；Final Full R@1 改善 67、回退 0；Final Full R@10 改善 0、回退 0；所有 Strict 指标逐题零变化。Final@10 平均字符数在两个版本中完全相同，均为 `419245.0170`；Strict@10 均为 `127540.5957`。

仍有 7 个可回答问题未达到 Final Full R@1，其中 3 个连 Final Full R@10 也缺少至少一条来源。下一步应提升时间事件抽取和候选发现，而不是继续扩大首胶囊。

## English

UMD 3.36 adds zero-expansion tidal migration above the provenance-safe 3.35 ghost layer. The query estimates evidence multiplicity and detects version resonance across repeated facts. Ghost navigation and the Strict atomic orbit then form a dual-orbit consensus. Only immutable source IDs already present in the pre-migration Final@10 union may move into rank one, exactly conserving the Final@10 source set and character cost.

Bounds:

- ghost budgets are capped at 12 for explicit lists, 10 for temporal queries, 6 for composite/resonance, 5 for evolution, and 2 for preference;
- at most six atomic anchors are added and deduplicated with ghost sources;
- migration is a conservative union and never deletes prior rank-one sources;
- Strict is computed from the pre-migration capsule snapshot, so packaging cannot alter a Strict rank;
- distilled ghost text is never evidence; every result re-anchors to original provenance;
- ranking does not read gold. This is a post-evaluation regression, not a blind test or official end-to-end answer score.

The 500-question LongMemEval_S run contains 470 answerable questions:

| Metric | UMD 3.35 | UMD 3.36 | Delta |
|---|---:|---:|---:|
| Final MRR | 0.9834 | **0.9981** | +0.0147 |
| Final Any R@1 | 0.9745 | **0.9979** | +0.0234 |
| Final Full R@1 | 0.8426 | **0.9851** | +0.1426 |
| Final Micro R@1 | 0.8944 | **0.9899** | +0.0955 |
| Final Full R@10 | 0.9936 | **0.9936** | 0 |
| Strict Any R@1 | 0.9298 | **0.9298** | 0 |
| Strict Full R@10 | 0.9085 | **0.9085** | 0 |

Per-query audit over the 470 answerable questions: 11 Final Any R@1 improvements and zero regressions; 67 Final Full R@1 improvements and zero regressions; zero Final Full R@10 changes; and zero changes in every Strict metric. Mean characters at Final@10 remain exactly `419245.0170`, while Strict@10 remains `127540.5957`.

Seven answerable questions remain below Final Full R@1, including three that still miss at least one source at Final Full R@10. The next useful work is temporal-event extraction and candidate discovery, not wider rank-one packaging.
