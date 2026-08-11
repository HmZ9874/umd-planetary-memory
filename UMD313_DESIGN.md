# UMD 3.13 relative-gravity design

## Core idea

A memory keeps its durable star and planet. A query creates temporary
attractors for each reasoning subgoal. A bound moon may participate in another
attractor's tidal bridge only when the new net potential exceeds its home
inertia, migration cost and uncertainty. Query-time capture never rewrites the
durable hierarchy.

## Gravity-element language

- `Q`: semantic resonance between a query attractor and a memory.
- `X`: lexical resonance.
- `Xi`: entity lock.
- `Tau`: temporal phase alignment.
- `Lambda`: extracted-relation bridge.
- `Pi`: source provenance confidence.
- `V`: utility and confidence vitality.
- `M`: bounded memory mass.
- `N`: novelty.
- `I`: home-orbit inertia.
- `K`: migration cost.
- `U`: uncertainty.
- `Omega`: the global bridge objective.

Per-memory attraction and capture are:

`A=.30Q+.14X+.15Xi+.08Tau+.12Lambda+.08Pi+.05V+.03M+.05N`

`Delta=A-.10I-.035K-.07U`

`Pcapture=sigmoid(Delta/temperature)`

The system compares every memory's best and second-best attractors. Capture
requires both an absolute probability threshold and a relative-potential
margin, preventing weak fields from causing orbit oscillation.

## Global optimization

Candidate generation scores every authorized active memory with neural and
sparse elements before truncation. Every subgoal reserves its own candidate
quota. The bridge optimizer then maximizes:

`Omega=mean(Delta)+.20new_goal+.12cross_planet+.10relation+.07new_time+.05new_source-.18redundancy`

This is a bounded greedy approximation to weighted set cover with diversity.
It directly targets LoCoMo's cross-session evidence shape: complementary moons
from different planets can share a transient bridge even when they were never
adjacent in storage.

## Invariants

- No query changes a memory's star or planet.
- Every bridge exposes immutable source IDs.
- Scope, tenant, quarantine and redaction checks happen before capture.
- Bridges are not persisted and store no extra plaintext or embeddings.
- Candidate and bridge sizes remain bounded.
