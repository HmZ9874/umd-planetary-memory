# UMD 3.11 bounded late-interaction results

## Honest feasibility boundary

The requested universal `Full R@10 >= 0.99` target is mathematically impossible on the local BEAM 100K labels. Twelve of 355 retrieval questions require more than ten distinct evidence turns (five require 11, one requires 12, three require 14, and three require 16). Even an oracle therefore has a maximum Full R@10 of `343/355 = 0.966197`.

The corresponding oracle ceilings are 0.997982 for all LoCoMo evidence questions, 0.997396 for LoCoMo categories 1–4, and 1.0 for LongMemEval.

## Frozen BEAM split experiment

The first six conversations (108 questions) were used as a development split. The remaining fourteen conversations (247 questions) were evaluated once with frozen weights:

`F311 = 0.60*late_interaction + 0.30*F39 + 0.10*adjacent_resonance + 0.10*base_rank`

| Split | Adapter | MRR | Any R@10 | Full R@10 |
|---|---|---:|---:|---:|
| Development | UMD 3.9 | 0.5543 | 0.7407 | 0.3519 |
| Development | UMD 3.11 | 0.5668 | 0.7500 | 0.4167 |
| Frozen unseen | UMD 3.9 | 0.4187 | 0.6356 | 0.3522 |
| Frozen unseen | UMD 3.11 | 0.4229 | 0.6316 | 0.3806 |
| Combined | UMD 3.9 | 0.4599 | 0.6676 | 0.3521 |
| Combined | UMD 3.11 | 0.4667 | 0.6676 | 0.3915 |

The unseen split validates the multi-evidence improvement (+0.0283 Full R@10), but Any R@10 loses one question. Full-history late interaction took about 18.6 minutes across both splits and is therefore not accepted as the default online path.

## Production form

`umd311_late.py` restricts token-level encoding to the top 64 coarse-orbit candidates. Query and passage token vectors are request-local and discarded after reranking; they do not create another persistent memory tier. The existing UMD 3.9 retriever remains the default until this bounded form passes full regression tests.
