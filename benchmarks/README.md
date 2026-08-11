# Public benchmark adapter

This directory runs the UMD 3.3 raw-text hierarchy on two official datasets:

- LoCoMo `locomo10.json` from `snap-research/locomo`;
- LongMemEval `longmemeval_s_cleaned.json` from
  `xiaowu0162/longmemeval-cleaned`.

The downloaded JSON files are intentionally ignored by Git because
LongMemEval_S is about 277 MB. Expected SHA-256 values for this run:

- LoCoMo: `79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4`
- LongMemEval_S cleaned: `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`

Run both datasets:

```powershell
& 'C:\Users\edward\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' .\pmd_formula_lab\benchmarks\public_benchmarks.py --benchmark both
```

## Metric definitions

- `Any evidence recall@k`: at least one gold evidence item appears in top-k.
- `Full evidence recall@k`: every gold evidence item appears in top-k.
- `Micro evidence recall@k`: retrieved gold items divided by all gold items.
- `MRR`: reciprocal rank of the first retrieved gold evidence item.
- `Exact answer string coverage`: diagnostic only; it checks whether the
  normalized reference answer occurs verbatim in retrieved context. It is not
  answer accuracy.

The retriever never reads the answer, evidence labels, `has_answer` turn labels,
or `question_type` when ranking. This adapter exercises lexical semantics,
session hierarchy, turn hierarchy, and neighboring-turn relations. It does not
yet exercise neural embeddings, contradiction extraction, or heterogeneous
provenance because those features are absent from the current raw-text model.

The generated evidence-retrieval scores are not the official end-to-end QA
scores. Those require an answer-generating model; LongMemEval also requires its
provided answer evaluator.
