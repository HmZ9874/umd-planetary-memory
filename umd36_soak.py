"""Restartable synthetic soak runner for UMD 3.6.

Synthetic samples intentionally never qualify as real-user long-running
evidence. Normal UMD36TenantMemory sessions record source='real' by default.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from umd35_core import UMD35Config
from umd36_persistent import UMD36Database, UMD36TenantMemory, master_key_from_base64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--tenant", default="soak")
    parser.add_argument("--principal", default="soak-admin")
    parser.add_argument("--duration-seconds", type=float, default=60.0)
    parser.add_argument("--restart-every", type=int, default=250)
    parser.add_argument("--max-active", type=int, default=128)
    parser.add_argument("--key-env", default="UMD36_MASTER_KEY")
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict[str, object]:
    encoded_key = os.getenv(args.key_env)
    if not encoded_key:
        raise SystemExit(f"set {args.key_env} to a URL-safe base64 32-byte master key")
    key = master_key_from_base64(encoded_key)
    database = UMD36Database(args.database, key)
    database.create_tenant(args.tenant, args.principal)
    config = UMD35Config(
        max_active_total=args.max_active,
        max_active_per_star=max(16, args.max_active // 2),
        max_relation_hops=5,
    )
    memory = UMD36TenantMemory(
        database,
        args.tenant,
        args.principal,
        config=config,
        validation_source="synthetic",
        reflection_interval=100,
    )
    deadline = time.monotonic() + max(0.1, args.duration_seconds)
    operations = 0
    restarts = 0
    failures = 0
    while time.monotonic() < deadline:
        try:
            if operations % 5:
                memory.write(
                    f"Synthetic soak event {operations}: shard {operations % 17} value {operations}.",
                    scope=f"soak-{operations % 8}",
                    fact_key=f"soak:{operations}",
                    explicit_importance=0.8,
                )
            else:
                memory.retrieve(
                    f"shard {operations % 17}",
                    scope=f"soak-{operations % 8}",
                    top_k=3,
                    reinforce=False,
                )
        except Exception:
            failures += 1
        operations += 1
        if args.restart_every > 0 and operations % args.restart_every == 0:
            database.close()
            database = UMD36Database(args.database, key)
            memory = UMD36TenantMemory(
                database,
                args.tenant,
                args.principal,
                validation_source="synthetic",
                reflection_interval=100,
            )
            restarts += 1
    report = {
        "operations": operations,
        "failures": failures,
        "restarts": restarts,
        "snapshot": memory.snapshot(),
        "note": "synthetic soak data is excluded from real-user long-running qualification",
    }
    database.close()
    return report


if __name__ == "__main__":
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2, default=str))
