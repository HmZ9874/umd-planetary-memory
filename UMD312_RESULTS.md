# UMD 3.12 measured results

UMD 3.12 passed capsule construction, query-independence, four-source capacity,
scope isolation, prompt-injection exclusion, version provenance, restart rebuild,
bounded container overhead, REST/Python SDK typing, TypeScript syntax, and all
previous UMD 3.5–3.10 regression suites.

## Retrieval results

| Benchmark | Metric | UMD 3.9 atomic | UMD 3.12 capsules |
|---|---|---:|---:|
| LoCoMo | MRR | 0.5182 | 0.6102 |
| LoCoMo | Any R@10 | 0.7381 | 0.8073 |
| LoCoMo | Full R@10 | 0.6418 | 0.7170 |
| LongMemEval | MRR | 0.9267 | 0.9335 |
| LongMemEval | Any R@10 | 0.9894 | 0.9915 |
| LongMemEval | Full R@10 | 0.9191 | 0.9468 |
| BEAM 100K | MRR | 0.4599 | 0.5678 |
| BEAM 100K | Any R@10 | 0.6676 | 0.7549 |
| BEAM 100K | Full R@10 | 0.3521 | 0.4282 |

These are label-blind evidence-retrieval diagnostics, not end-to-end answer or
judge scores. Capsule metrics expand immutable source provenance and therefore
use a different retrieval unit from atomic metrics.

## Cost

- LoCoMo mean context at 10: 1,528 to 2,410 characters.
- LongMemEval mean context at 10: 124,920 to 252,226 characters.
- BEAM mean context at 20: 38,978 to 79,702 characters.
- LongMemEval evaluation-process peak working set: about 2,430 MiB, including
  the dataset and ONNX embedding model; this is not capsule index RAM.

LongMemEval crosses Any R@10=0.99 but not Full R@10=0.99. BEAM's atomic
Full R@10 target remains mathematically impossible because 12 questions contain
more than ten gold turns; capsule results are reported separately and do not
rewrite that fact.

## Scale probe

With 200 active memories in one event chain, UMD 3.12 reconstructed 466
capsules and 864 source links. Measured capsule-index container overhead was
144,552 bytes, or 722.76 bytes per active memory. Ten selected capsules expanded
to 40 unique sources, retrieval took about 496 ms with the synthetic hash-based
test engine, and capsule plaintext cache remained disabled. Response rendering
now enforces `budget_chars` as a hard limit with atomic fallback.
