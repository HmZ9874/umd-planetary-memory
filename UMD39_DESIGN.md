# UMD 3.9 Observatory Platform

UMD 3.9 packages the UMD 3.8 cosmic memory engine as a developer platform.
The three platform layers keep the planetary vocabulary while using ordinary,
documented protocols:

```text
Telescope SDK  →  Observatory API  →  UMD Cosmic Core
                         ↓
                  Observatory Console
                         ↓
                   Spaceport Adapters
```

## Telescope SDK

The Python SDK provides synchronous and asynchronous clients with typed memory,
write receipt, search-hit, and API-error objects. It supports local in-process
transport for tests and standard HTTP for deployment. The TypeScript SDK uses
the platform Fetch API and ships runtime JavaScript plus declaration types.

Both SDKs preserve the same operations and field names:

- `add`: scoped, tagged, classified, importance-weighted writes;
- `search`: force, radius, energy, components, and explanation;
- `get` and cursor-based `list`;
- pairwise retrieval feedback;
- entity, graph, community, audit, and system inspection.

## Observatory API

The standard response envelope is:

```json
{
  "data": {},
  "meta": {
    "request_id": "req_...",
    "api_version": "2026-08-09"
  }
}
```

Errors are stable machine-readable objects containing HTTP status, `code`,
message, request ID, and details. `/openapi.json` describes the versioned
surface, while `/health` is the only unauthenticated runtime endpoint.

### Security and reliability

- raw API keys are shown once; only HMAC-peppered hashes are stored;
- keys bind exactly one tenant and principal and support immediate revocation;
- UMD RBAC and ABAC remain authoritative below the platform boundary;
- per-principal read/write minute orbits enforce quotas;
- request and response byte counts, status, operation, and latency enter a
  persistent usage ledger;
- POST requests may carry an `Idempotency-Key`; identical retries replay the
  original response, while a different request receives a 409 conflict;
- page cursors are opaque and stable over `(updated_at, memory_id)` ordering;
- optional external token verification supports an OIDC gateway;
- the HTTP server accepts an SSL context, and the launcher supports certificate
  and key files with TLS 1.2 minimum.

The idempotency ledger is a platform replay mechanism. The underlying memory
write itself still receives UMD 3.8 gravity-capsule atomicity.

## Observatory Console

The dependency-free browser console is served by the same process. It contains:

- universe totals and event-chain health;
- active/cold memory inspection and browser-side filters;
- a memory creation form using idempotent gravity capsules;
- query-comet retrieval with per-force component bars;
- interactive SVG star/planet/memory/entity graphs;
- entity aliases, confidence, and ambiguity counts;
- encrypted audit-chain event inspection;
- session-only API-key storage and connection switching.

All memory-derived HTML is escaped before rendering. SVG labels use
`textContent`; static file resolution rejects traversal outside the console
directory.

## Running locally

For a disposable development universe:

```powershell
& '.\pmd_formula_lab\.venv\Scripts\python.exe' `
  '.\pmd_formula_lab\umd39_server.py' `
  --dev --bootstrap --database '.\umd39-demo.sqlite3'
```

The launcher prints generated development keys and the one-time API key. Open
`http://127.0.0.1:8765/` and paste that API key. For persistent operation, set
URL-safe base64 `UMD39_MASTER_KEY` and `UMD39_API_PEPPER`; do not use ephemeral
development keys with an existing database.

## Current boundary

This release establishes a real local and HTTP platform surface. It does not
yet include a published package registry release, hosted cloud control plane,
PostgreSQL/pgvector backend, third-party marketplace, billing provider, or
independently operated ecosystem. Those are subsequent Spaceport deployment
work, not features silently claimed by the local test suite.

