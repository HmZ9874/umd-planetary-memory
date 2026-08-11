# UMD 3.8 Cosmic Memory Architecture

UMD 3.8 preserves the star → planet → memory hierarchy, but turns the missing
production mechanisms into planetary-native structures. It remains compatible
with the encrypted UMD 3.6 store and the UMD 3.7 force model.

## 1. Gravity capsule: one write boundary

The ordinary write path no longer performs a memory transaction followed by a
second planetary transaction. A gravity capsule contains the new memory,
hierarchy changes, entity/time orbits, encrypted planetary state, and audit
operation:

```text
Ω_t = { M_t, H_t, G_t, P_t, A_t }
(revision, Ω)_(t+1) = Commit_SQLite(Ω_t)
Commit_SQLite(Ω_t) = all components persisted, or no component persisted
```

If a commit fails, resident state is reconstructed from the last valid
encrypted revision. Tests verify that neither a memory nor an event survives a
failed capsule.

## 2. Incremental orbital navigation

Deterministic random planes generate orbit signatures. A new memory probes
matching and one-bit neighboring buckets, then creates only bounded navigable
gravitational edges:

```text
σ_l(v) = bits(sign(H_l v))
C(v) = ⋃_l [bucket_l(σ_l(v)) ∪ bucket_l(σ_l(v) xor 2^b)]
N_M(v) = TopM_{u ∈ C(v)} cosine(v, u)
```

BM25 postings, document frequencies, LSH buckets, vectors, and neighbor edges
support `upsert` and `remove`. A sidecar-only revision advances the index clock
without rebuilding the active universe. Restart performs one derived rebuild
from durable active records.

## 3. Neural tidal reranking

UMD 3.8 includes a trainable interaction network. Its signals are semantic
similarity, lexical overlap, query containment, exact identifier agreement,
polarity agreement, length fit, and ordered bigram overlap:

```text
x(q,m) ∈ R^7
h = tanh(xW1 + b1)
T_θ(q,m) = sigmoid(hW2 + b2)
L_pair = log(1 + exp(-(T_θ(q,m+) - T_θ(q,m-))))
```

Useful/rejected choices update the bounded MLP online. Parameters are encrypted
in planetary state and restored after restart. This is an original UMD learner,
not a pretrained cross-encoder checkpoint.

## 4. Entity identity gravity and communities

Entity merging requires both sufficient attraction and a clear winner:

```text
I(e,c) = 0.58 cosine(v_e,v_c)
       + 0.22 Jaccard(trigram(e),trigram(c))
       + 0.14 type_compatible(e,c)
       + 0.06 alias_containment(e,c)

merge(e,c*) iff I(e,c*) ≥ τ_I
             and I(e,c*) - I(e,c_2) ≥ δ_I
```

Close calls enter an ambiguity ledger and remain separate entities, preventing
silent false merges. Evidence-bearing temporal edges form communities through
bounded deterministic label gravity; communities retain supporting memory IDs.

## 5. Context Lagrange assembly

Declarative planets, shared blocks, and procedures compete for one context
budget:

```text
maximize Σ_i z_i F_i
subject to Σ_i z_i chars_i ≤ B, z_i ∈ {0,1}
greedy density ρ_i = F_i / max(32, chars_i)
```

UMD decides what memory crosses the context boundary. The host agent runtime
continues to own model and tool execution.

## 6. Constellation protocol and recovery

Replica placement remains rendezvous-based. UMD 3.8 adds parallel transport,
signed idempotent gravity packets, quorum enforcement, and read repair:

```text
R_k = TopR_r SHA256(tenant || replica_r)
Q = floor(R/2) + 1
sig(packet) = HMAC(K_cluster, canonical_json(packet))
write succeeds iff acknowledgements ≥ Q
repair targets = {r | revision_r < max_j revision_j}
```

`HTTPReplicaTransport` crosses process and machine boundaries. The checkpoint
manager uses SQLite online backup, SHA-256 manifests, integrity validation, and
recovery to a new non-overwriting path.

## 7. Runtime, multimodal, security, and observability

- Prompt/KV/adapter memories are passed to a supplied model runtime only after
  exact fingerprint verification.
- Image, audio, video, and document meteors require a registered semantic
  encoder. UMD stores a digest and caption/metadata, not raw media payload.
- The gateway exposes authenticated REST and MCP. It accepts UMD short-lived
  capabilities or an external OIDC claims verifier; an SSL context enables TLS.
- Bounded spans, counters, latency samples, Prometheus output, and an exporter
  callback provide an OpenTelemetry-compatible integration boundary.
- The end-to-end QA harness measures answer EM/F1, evidence recall, grounded
  citations, abstention, and an optional judge with the same answer model.

## 8. Code status versus deployment evidence

| Area | UMD 3.8 implementation | Still requires external evidence |
|---|---|---|
| Incremental ANN | bounded orbital graph + LSH/BM25 | 1M/10M production hardware run |
| Neural reranking | online pairwise tidal MLP | large labeled-corpus comparison |
| Entity graph | margin disambiguation, temporal paths, communities | broad multilingual resolution corpus |
| Transactions | ordinary write is one SQLite gravity capsule | distributed multi-database transaction deployment |
| Distribution | HTTP, signing, idempotency, quorum, repair | live multi-node/cross-region chaos run |
| HA | online backup and recovery drill | operator-run PITR/RPO/RTO and rolling upgrades |
| Security | RBAC/ABAC, AES-GCM, OIDC boundary, optional TLS | IdP/KMS/HSM deployment and compliance audit |
| Multimodal | native adapter and digest-safe ingest | selected model quality evaluation |
| End-to-end QA | comparable answer/judge harness | official answer-model run and score |
| Long-run quality | synthetic tests labeled synthetic | seven-day real-user drift, cost, failure report |

The last column cannot be truthfully completed by source code alone. It is the
deployment qualification plan, not a hidden algorithmic omission.

