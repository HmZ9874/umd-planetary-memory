# UMD Planetary Memory / UMD 行星记忆

[中文](#中文) · [English](#english) · [Results](docs/BENCHMARKS.en.md) · [成绩](docs/BENCHMARKS.zh-CN.md)

> Public research release, 2026-08-11. UMD is an experimental memory and retrieval system. The reported numbers are evidence-retrieval proxies unless explicitly stated otherwise; they are **not** official end-to-end answer accuracy or leaderboard submissions.

## 中文

UMD（Universal/Universe Memory Dynamics）把长期 AI 记忆建模成一个动态宇宙：主题是恒星，事件与事实是行星，局部细节是卫星，关系形成引力边，更新与删除形成版本化状态场。系统把词法检索、神经嵌入、时间、实体、图关系、权限、遗忘和完整证据回忆组合到一个可审计的检索过程里。

### 当前版本：UMD 3.30

UMD 3.30 新增最多三跳的“引力弹弓”事实图：

1. 从问题中识别发射恒星（主体）；
2. 沿当前可见的主体—关系—客体边最多跳转三次；
3. 把到达目标关系的答案事实来源提升到原子首位；
4. 用模型预测出的答案对象形成“回声小行星带”，补全分散和重复证据；
5. 只在密集编号事实宇宙中启用，普通对话保持冻结控制路径。

MemoryAgentBench 全量检索代理从 UMD 3.29 提升到 UMD 3.30：

| 指标 | 3.29 | 3.30 |
|---|---:|---:|
| Final Any R@1 | 0.5078 | **0.6727** |
| Final Full R@10 | 0.3429 | **0.5712** |
| Strict Any R@1 | 0.3767 | **0.5556** |
| Strict Full R@10 | 0.1484 | **0.2595** |

完整材料：

- [中文算法白皮书](docs/ALGORITHM.zh-CN.md)
- [公式与状态机](docs/FORMULAS.md)
- [中文成绩与限制](docs/BENCHMARKS.zh-CN.md)
- [复现说明](docs/REPRODUCIBILITY.md)
- [UMD 3.30 优化审计](benchmarks/results/UMD330_SLINGSHOT_OPTIMIZATION_REPORT.md)

### 快速验证

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q `
  benchmarks/umd316_adapter_test.py `
  benchmarks/umd317_adapter_test.py `
  benchmarks/umd318_adapter_test.py `
  benchmarks/umd325_adapter_test.py `
  benchmarks/umd326_adapter_test.py `
  benchmarks/umd327_adapter_test.py `
  benchmarks/umd329_performance_test.py `
  benchmarks/umd330_adapter_test.py `
  benchmarks/umd39_benchmarks_test.py
```

测试数据不随仓库分发。请从各 benchmark 官方仓库获取，并遵守各自许可。

## English

UMD (Universal/Universe Memory Dynamics) models long-term AI memory as a dynamic universe: topics are stars, events and facts are planets, local details are moons, relations are gravitational edges, and updates/deletions form versioned state fields. It combines lexical retrieval, neural embeddings, time, entities, graph relations, access control, forgetting, and complete-evidence recall in an auditable pipeline.

### Current release: UMD 3.30

UMD 3.30 introduces a bounded three-hop **gravitational slingshot** fact graph:

1. identify a launch star (subject) in the query;
2. traverse current visible subject-relation-object edges for at most three hops;
3. promote the target-relation fact provenance to atomic rank one;
4. construct an echo asteroid belt from occurrences of the model-predicted object;
5. activate only in dense numbered fact universes, leaving ordinary dialogue on the frozen control path.

Full MemoryAgentBench retrieval-proxy improvement:

| Metric | 3.29 | 3.30 |
|---|---:|---:|
| Final Any R@1 | 0.5078 | **0.6727** |
| Final Full R@10 | 0.3429 | **0.5712** |
| Strict Any R@1 | 0.3767 | **0.5556** |
| Strict Full R@10 | 0.1484 | **0.2595** |

Detailed material:

- [English algorithm white paper](docs/ALGORITHM.en.md)
- [Formula and state-machine reference](docs/FORMULAS.md)
- [English results and limitations](docs/BENCHMARKS.en.md)
- [Reproduction guide](docs/REPRODUCIBILITY.md)
- [UMD 3.30 optimization audit](benchmarks/results/UMD330_SLINGSHOT_OPTIMIZATION_REPORT.md)

### Repository map

- `umd35_core.py` — online raw-text memory engine;
- `umd36_persistent.py` — encrypted persistence, tenants, WAL/recovery;
- `umd37_planetary.py` — planetary interaction runtime;
- `umd38_cosmic.py` — capabilities, policy and distributed/cosmic layer;
- `umd39_platform.py`, `umd_sdk/`, `sdk-typescript/` — platform and SDK surface;
- `benchmarks/` — public benchmark adapters, frozen protocols and UMD 3.30;
- `UMD*_DESIGN.md` — historical design records and derivations.

### Integrity statement

- Ranking runs before answers or evidence labels are read.
- Published result files declare `gold_used_for_ranking=false`.
- Raw benchmark data, third-party repositories, model weights, checkpoints and learned artifacts are intentionally excluded.
- MemoryAgentBench currently scores Accurate Retrieval and Conflict Resolution retrieval proxies; Test-Time Learning and Long-Range Understanding are not yet scored.
- MemoryArena reports a prior-subtask reuse retrieval proxy, not environment success.
- EvolMem was still marked under construction in the locally audited upstream checkout.

## License

No open-source license has been selected yet. Public source visibility does not by itself grant reuse rights. A license should be added explicitly before third-party redistribution or commercial use.
