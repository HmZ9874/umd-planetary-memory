# UMD Formula Reference / UMD 公式参考

This document publishes the current deterministic score derivations. Exact executable definitions remain authoritative in the linked source files. / 本文公开当前确定性评分推导；精确可执行定义以链接源码为准。

## 1. Normalization / 归一化

For a non-negative score vector `x`, UMD uses max normalization where applicable:

`unit(x_i) = x_i / max_j(x_j)` when `max(x) > 0`, otherwise `0`.

Rank fusion uses reciprocal rank:

`RR_i = 1 / (1 + rank_i)`.

The initial lexical/planet seed is:

`Seed_i = RR_lexical(i) + 0.72 · RR_planet(group_i)`.

## 2. Base gravitational force / 基础引力

The frozen UMD 3.9 public-array force is:

`F_i = 0.34S_i + 0.22L_i + 0.16P_i + 0.08C_i + 0.08E_i + 0.06T_i + 0.06G_i`

where:

- `S`: neural semantic similarity / 神经语义相似度;
- `L`: BM25 lexical score / BM25 词法分;
- `P`: planet/session hierarchy score / 行星或会话层级分;
- `C`: character n-gram score / 字符 n-gram 分;
- `E`: entity and identifier resonance / 实体和值共振;
- `T`: temporal phase / 时间相位;
- `G`: local graph flux / 局部图通量.

All weights sum to one. Authority and learned utility exist in the production runtime but are uniform in raw public arrays and therefore omitted from this frozen adapter.

Source: [`benchmarks/umd39_benchmarks.py`](../benchmarks/umd39_benchmarks.py).

## 3. State mass / 状态质量

For an explicit current-state query:

- negative-only fact: `Δstate = -0.18`;
- positive-only fact: `Δstate = +0.08`;
- mixed update fact: `Δstate = -0.04`;
- neutral fact: `0`.

`F_state(i) = max(0, F_i + Δstate_i)`.

Historical/evolution queries set every state delta to zero so that both sides of a transition remain retrievable.

## 4. Event satellites / 事件卫星

Secondary sources are ordered by:

`E_i = 0.46F_i + 0.30S_i + 0.14L_i + 0.06/(1+rank_i) + 0.04·I[new_group]`

Stable capsule sources are excluded from the secondary pool. Query-form rules choose a bounded event budget; direct facts use a small budget, causal/temporal/set questions receive a larger one.

## 5. Marginal constellation coverage / 星座边际覆盖

For broad-set questions, the base relevance is:

`B_i = 0.44unit(F_i) + 0.36unit(S_i) + 0.20unit(L_i)`.

At greedy step `t`, with covered tokens `C_t`:

`query_coverage_i = |T_i ∩ Q| / max(1, |Q|)`

`novel_i(t) = |T_i - C_t| / max(1, |T_i|)`

`redundancy_i(t) = |T_i ∩ C_t| / max(1, |T_i ∪ C_t|)`

`Coverage_i(t) = B_i + 0.10query_coverage + 0.09I[new_entity] + 0.07I[new_group] + 0.04I[new_date] + 0.08novel - 0.13redundancy`

UMD 3.29 preserves this exact equation but maintains overlap mass incrementally:

`o_i(t) = |T_i ∩ C_t|`

`o_i(t+1) = o_i(t) + Σ I[x ∈ T_i], x ∈ (T_selected - C_t)`.

This removes repeated set copying without changing ordering semantics. Two hundred randomized equivalence cases and a permanent reference-oracle test passed.

## 6. Directive gravity / 指令引力

Trusted instruction/preference memories receive a transient bonus only if query mode and topic/family contact match:

`applicability = clip(0.58·lexical_overlap + 0.42·family_match, floor, 1)`

`directive_bonus = 0.24 + 0.14·applicability`

with floor `0.45` for instructions and `0.30` for preferences. History lookup and multi-evidence questions disable this bridge. UMD 3.29 computes query-invariant mode/cues/tokens once and shares the resulting vector between relative gravity and the transient bridge.

## 7. Capsules and conservation / 胶囊与守恒

An output capsule is a provenance tuple:

`C_k = stable_anchor_k ∪ striped_satellites_k`.

If `O_old` is the frozen satellite orbit and `O_new` appends coverage/echo evidence, conservation requires:

`O_old[:b]` to remain an immutable prefix for the protected budget `b`.

Therefore, for every reported prefix `k`:

`Sources_old(@k) ⊆ Sources_new(@k)`.

Strict atomic retrieval separately emits `(source_id,)` for each ranked source.

## 8. Antimatter and current answers / 反物质与当前答案

For non-historical queries, explicit invalidation source IDs are removed from the answer orbit but retained in the audit orbit:

`Answer(C) = C - NegativeSourceIDs`

`Audit(C) = C`.

The immutable negative-ID set is compiled once and shared by all output channels.

## 9. Galactic census / 银河 census

Aggregate mode combines relevance, semantic contact and positive-operation mass:

`Census_i = 0.37L_i + 0.20F_i + 0.13S_i + 0.22contact_i + 0.02min(4, exact_i) + 0.06I[positive]`.

Broad current sets additionally use recency and domain contact. Invalid facts receive a `-2.0` penalty outside historical mode. The census budget is capped at 512 sources for aggregate queries and lower for ordinary broad/historical queries.

## 10. UMD 3.30 slingshot / UMD 3.30 引力弹弓

A numbered fact is:

`e = (subject, relation, object, ordinal, provenance)`.

The current visible edge is:

`e_current(s,r,t) = argmax ordinal(e), source(e) < t`.

The bounded path energy is:

`Energy(path) = Σ[0.20 + min(3, contact(q,r)/5)] + 6·I[last_relation ∈ target(q)]`.

Breadth-first traversal is capped at three edges. The winning terminal fact is promoted to atomic rank one. Its predicted object `a` defines an echo set:

`Echo(a) = {i | casefold(a) is a substring of casefold(text_i)}`

with at most 96 appended satellites. No gold answer or evidence ID is used in this calculation.

## 11. Retrieval metrics / 检索指标

For gold evidence set `Y` and the union of sources in the first `k` retrieval units `R_k`:

- `Any R@k = mean(I[Y ∩ R_k ≠ ∅])`;
- `Full R@k = mean(I[Y ⊆ R_k])`;
- `Micro R@k = Σ|Y ∩ R_k| / Σ|Y|`;
- `MRR = mean(1 / rank(first gold retrieval unit))`.

Final capsule and strict atomic metrics must always be reported separately.

## 12. Complexity and memory / 复杂度与内存

- BM25 scoring is sparse in query postings plus an `O(n)` output vector.
- Neural scoring is bounded by the lexical candidate pool (64 in the published frozen exam).
- Marginal coverage is bounded to 192 candidates; UMD 3.29 replaces large repeated set algebra with incremental postings.
- Slingshot construction is linear in parsed facts and uses a sparse subject adjacency map; traversal is capped at three hops.
- On the largest measured structured context (1,119 chunks, 17,831 facts), the slingshot graph used 5.92 MiB steady Python memory, 15.03 MiB construction peak and 0.652 seconds to build.

