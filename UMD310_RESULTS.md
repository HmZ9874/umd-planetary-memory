# UMD 3.10 Results

## Implemented

- trusted instruction and preference satellite indexes;
- latest-per-family routing with four-version contradiction history;
- scope/ABAC filtering and trusted-source admission;
- prompt-injection quarantine before directive promotion;
- star/planet episode chains and bounded neighbor expansion;
- multi-planet, multi-time, multi-entity coverage selection;
- chronological answer context;
- bounded cold-directive reconstruction after restart;
- UMD 3.9 Observatory server upgraded to UMD 3.10 sessions;
- snapshot-visible cognitive index memory accounting.

## Validation

The UMD 3.10 adversarial suite passes 19/19 checks. The complete UMD 3.5–3.10
regression chain plus benchmark-adapter unit tests passes 168 top-level checks.

The targeted failure replays verify that:

- a formatting instruction semantically distant from a login question still
  reaches a five-item answer context;
- the newest dependency preference reaches recommendation queries while the
  older preference remains available only to contradiction mode;
- directives cannot cross project scopes;
- assistant self-instructions and quarantined injection text cannot enter the
  satellite orbit;
- timeline retrieval spans at least three session planets and can be returned
  in chronological order;
- ordinary exact-identifier retrieval keeps its original top anchor;
- restart reconstructs directives and episode chains from durable storage;
- the Python SDK and platform expose chronological retrieval;
- writes still produce one encrypted transaction/event and the hash chain
  remains valid.

## Memory probe

With 200 active synthetic memories and 10 directive satellites, the cognitive
indexes used approximately 27,848 bytes of owned Python container memory, or
139.24 bytes per active memory. This excludes shared ID strings and timestamps
already owned by memory nodes, and it deliberately contains no copied text or
vectors. The number is an implementation-level estimate, not total process RAM.

## Current boundary

The structural fixes have not yet been assigned a new official LoCoMo,
LongMemEval, or BEAM end-to-end score. A fair score requires freezing this
implementation, rerunning the datasets without label-driven tuning, and using
the same answerer and judge model for every compared system.
