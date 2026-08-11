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
  benchmarks/umd316_adapter_test.py \
  benchmarks/umd317_adapter_test.py \
  benchmarks/umd318_adapter_test.py \
  benchmarks/umd325_adapter_test.py \
  benchmarks/umd326_adapter_test.py \
  benchmarks/umd327_adapter_test.py \
  benchmarks/umd329_performance_test.py \
  benchmarks/umd330_adapter_test.py \
  benchmarks/umd39_benchmarks_test.py
```

Expected release result: `72 passed`.

Validate the published result manifests:

```bash
python scripts/verify_published_results.py
```

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

## UMD 3.30 MemoryAgentBench / UMD 3.30 全量测试

```bash
python -m benchmarks.umd3282_continued_exam \
  --benchmark memoryagentbench \
  --full \
  --output benchmarks/results/local_umd330_memoryagentbench.json
```

The runner writes a versioned local checkpoint after every context. Checkpoints are ignored and must not be presented as final results. Formal ranking always occurs before answer inspection.

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

