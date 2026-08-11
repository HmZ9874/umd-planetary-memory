"""Run the UMD 3.9 Observatory API and console."""

from __future__ import annotations

import argparse
import base64
import os
import secrets
import ssl
from pathlib import Path

from umd36_persistent import UMD36Database, UMD36TenantMemory
from umd312_capsules import UMD312ConstellationMemory
from umd39_platform import APIKeyAuthority, UMD39Platform


def _decode_key(name: str, *, dev: bool, key_file: str | None = None) -> bytes:
    value = Path(key_file).read_text(encoding="utf-8").strip() if key_file else os.environ.get(name)
    if value:
        try:
            return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
        except Exception as error:
            raise SystemExit(f"{name} must be URL-safe base64") from error
    if not dev:
        raise SystemExit(f"{name} is required (or use --dev for an ephemeral local universe)")
    key = secrets.token_bytes(32)
    print(f"DEV {name}={base64.urlsafe_b64encode(key).decode().rstrip('=')}")
    return key


def main() -> None:
    parser = argparse.ArgumentParser(description="UMD 3.9 Observatory server")
    parser.add_argument("--database", default="umd39.sqlite3")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--tenant", default="local")
    parser.add_argument("--principal", default="owner")
    parser.add_argument("--bootstrap", action="store_true", help="create tenant and print a new API key")
    parser.add_argument("--dev", action="store_true", help="allow ephemeral generated secrets")
    parser.add_argument("--tls-cert")
    parser.add_argument("--tls-key")
    parser.add_argument("--master-key-file", help="file containing URL-safe base64 master key")
    parser.add_argument("--api-pepper-file", help="file containing URL-safe base64 API-key pepper")
    args = parser.parse_args()

    master_key = _decode_key("UMD39_MASTER_KEY", dev=args.dev, key_file=args.master_key_file)
    api_pepper = _decode_key("UMD39_API_PEPPER", dev=args.dev, key_file=args.api_pepper_file)
    database = UMD36Database(Path(args.database), master_key)
    if args.bootstrap:
        database.create_tenant(args.tenant, args.principal)
    sessions = {}

    def lookup(tenant_id: str, principal_id: str):
        key = tenant_id, principal_id
        if key not in sessions:
            sessions[key] = UMD312ConstellationMemory(
                UMD36TenantMemory(database, tenant_id, principal_id)
            )
        return sessions[key]

    authority = APIKeyAuthority(database, api_pepper)
    if args.bootstrap:
        credential = authority.create(args.tenant, args.principal, "bootstrap")
        print("API key (shown once):", credential["token"])
    platform = UMD39Platform(lookup, authority)
    tls = None
    if args.tls_cert or args.tls_key:
        if not args.tls_cert or not args.tls_key:
            raise SystemExit("--tls-cert and --tls-key must be supplied together")
        tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        tls.minimum_version = ssl.TLSVersion.TLSv1_2
        tls.load_cert_chain(args.tls_cert, args.tls_key)
    server = platform.make_server(args.host, args.port, ssl_context=tls)
    scheme = "https" if tls else "http"
    print(f"UMD Observatory: {scheme}://{args.host}:{server.server_port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        database.close()


if __name__ == "__main__":
    main()
