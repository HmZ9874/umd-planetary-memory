# Reproducibility / 复现说明

## What is included / 仓库包含内容

- complete UMD algorithm evolution source code;
- deterministic unit, invariant, adversarial and benchmark-adapter tests;
- frozen benchmark protocols/configuration;
- audited formal result JSON and reports;
- Python and TypeScript SDK prototypes;
- no raw third-party dataset, model weight, secret, checkpoint or learned binary artifact.

- 完整 UMD 算法演进源码；
- 确定性单测、不变量、对抗与 benchmark 适配器测试；
- 冻结协议与配置；
- 审计后的正式结果 JSON 和报告；
- Python/TypeScript SDK 原型；
- 不包含第三方原始数据、模型权重、密钥、检查点或训练二进制。

## Environment / 环境

The published validation used Python 3.12 on Windows with the local FastEmbed ONNX backend. Core deterministic tests are platform-neutral.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-dev.txt
```

Optional backends:

```bash
python -m pip install -r requirements-neural.txt
python -m pip install -r requirements-persistence.txt
python -m pip install -r requirements-learning.txt
python -m pip install -r requirements-llm.txt
```

The LLM extractor is optional. Its security tests use a fake adapter and do not require a network call. Never commit `OPENAI_API_KEY` or any database master key.

## Deterministic release tests / 确定性发布测试

```bash
python -m pytest -q \
  --ignore=benchmarks/vendor \
  --ignore=benchmarks/vendor_exam \
  --ignore=benchmarks/data \
  --ignore=benchmarks/data_exam
```

Expected release result: `148 passed`.

Validate the published result manifests:

```bash
python scripts/verify_published_results.py
```

Expected manifest result: `validated_results=20 formula_relations=61 gold_used_for_ranking=false status=pass`.

## Benchmark data layout / Benchmark 数据目录

Data is deliberately ignored. After obtaining it from upstream, place it under these local-only paths:

```text
benchmarks/data/
  locomo10.json
  longmemeval_s_cleaned.json

benchmarks/data_exam/
  MemoryAgentBench/data/*.parquet
  EverMemBench/dataset/*
  LongMemEval-V2/{questions.jsonl,trajectories.jsonl,haystacks/...}
  MemoryArena/*/data.jsonl
  MemoryBench/{corpus,dataset}/...
```

Additional adapters document their expected local source under `benchmarks/README.md` and in their runner code. Verify upstream licenses before downloading or redistributing anything.

## Frozen LoCoMo and LongMemEval / 冻结测试

```bash
python -m benchmarks.umd3282_frozen_exam \
  --benchmark locomo \
  --output benchmarks/results/local_locomo.json

python -m benchmarks.umd3282_frozen_exam \
  --benchmark longmemeval \
  --output benchmarks/results/local_longmemeval.json
```

The formal snapshot used SHA-256:

- LoCoMo: `79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4`
- LongMemEval_S cleaned: `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`

## UMD 3.34 LongMemEval development and holdout / 开发集与保留集

The published split is positional and declared in advance: questions `0:100`
are post-dataset development; questions `100:500` are the parameter-frozen
holdout. / 公开切片按位置预先声明：`0:100` 为后数据集开发回放，`100:500`
为参数冻结保留集。

```bash
python -m benchmarks.umd3282_frozen_exam \
  --benchmark longmemeval --physics-v334 \
  --question-limit 100 \
  --output benchmarks/results/local_umd334_dev100.json

python -m benchmarks.umd3282_frozen_exam \
  --benchmark longmemeval --physics-v334 \
  --question-start 100 --question-limit 400 --frozen-holdout \
  --output benchmarks/results/local_umd334_holdout400.json
```

The full replay can be emitted after both slices complete, but it mixes tuned
development questions with holdout questions and must not be called blind. The
ignored checkpoint is resumable every ten questions.

## UMD 3.35 LongMemEval ghost regression / 幽灵物质回归

UMD 3.35 was developed after inspecting LongMemEval behavior, so this command
is a post-evaluation regression and must not be described as blind or frozen. /
UMD 3.35 在检查 LongMemEval 行为后开发，因此下面是评估后回归，不能称为盲测或冻结保留集。

```bash
python -m benchmarks.umd3282_frozen_exam \
  --benchmark longmemeval --physics-v335 \
  --output benchmarks/results/local_umd335_longmemeval_regression.json
```

The ignored checkpoint resumes every ten questions. Ghost matter uses no gold
for ranking and publishes only original source IDs; the evaluator reads gold
after retrieval.

## UMD 3.36 LongMemEval constellation regression / 星座回归

```bash
python -m benchmarks.umd3282_frozen_exam \
  --benchmark longmemeval --physics-v336 \
  --output benchmarks/results/local_umd336_longmemeval_regression.json
```

This is also post-evaluation development. The migration horizon is the
pre-migration Final@10 union; Strict is scored from the conserved capsule
snapshot. / 这同样是评估后开发；迁移视界是迁移前 Final@10 并集，Strict
使用守恒胶囊快照计分。

## UMD 3.30–3.33.1 MemoryAgentBench / UMD 3.30–3.33.1 测试

```bash
python -m benchmarks.umd3282_continued_exam \
  --benchmark memoryagentbench \
  --full \
  --output benchmarks/results/local_umd330_memoryagentbench.json

python -m benchmarks.umd3282_continued_exam \
  --benchmark memoryagentbench \
  --full \
  --physics-v331 \
  --output benchmarks/results/local_umd331_memoryagentbench.json

python -m benchmarks.umd3282_continued_exam \
  --benchmark memoryagentbench \
  --full \
  --physics-v332 \
  --output benchmarks/results/local_umd3321_memoryagentbench.json

python -m benchmarks.umd3282_continued_exam \
  --benchmark memoryagentbench \
  --full \
  --physics-v333 \
  --output benchmarks/results/local_umd3331_memoryagentbench.json
```

The runner writes a versioned local checkpoint after every context. Checkpoints are ignored and must not be presented as final results. Formal ranking always occurs before answer inspection.

For a same-capacity A/B, omit `--full` in all compared commands. This applies the same frozen 600-chunk limit; never compare a capacity result with a full result. / 同容量 A/B 应在所有对比命令中省略 `--full`，统一使用 600-chunk 上限；容量档不能与全量档直接比较。

## How metrics are computed / 指标计算

For each question:

1. call the retriever with query and visible memory only;
2. freeze returned final and strict-atomic channels;
3. read evaluator evidence/answer;
4. map gold evidence to immutable source IDs;
5. update Any, Full, Micro recall and MRR accumulators.

对每个问题：先检索并冻结结果，再读取评测答案/证据，最后累计指标。标准答案不进入检索函数。

## Reproduction caveats / 复现差异

- ONNX runtime, CPU threading and model version can change runtime and tiny floating-point tie behavior.
- Public JSON records the encoder identity and dataset hashes where available.
- Proxy mapping can exclude questions whose answer string is not found in a frozen chunk/day-group/trajectory. The denominator is always reported.
- Official answer generation, judge models, LAFS and environment-success metrics require their upstream evaluation stacks and are not reproduced here.

## Adding a new result / 新增成绩

1. freeze parameters and protocol before reading held-out labels;
2. use a versioned checkpoint and output filename;
3. record dataset hash, question counts, excluded counts, hardware/runtime and encoder identity;
4. report final capsule and strict atomic channels together;
5. state whether the number is development, held-out, full, sampled, proxy or official;
6. run `scripts/verify_published_results.py` and the release test suite.
