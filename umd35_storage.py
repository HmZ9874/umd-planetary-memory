"""Archive backends for UMD 3.5.

Only inactive compressed payloads live here. Active vectors and hierarchy
metadata remain owned by the memory engine, so retrieval stays fast.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Protocol


class ArchiveStore(Protocol):
    def put(self, memory_id: str, payload: bytes) -> None: ...

    def get(self, memory_id: str) -> bytes | None: ...

    def contains(self, memory_id: str) -> bool: ...

    def total_bytes(self) -> int: ...

    def count(self) -> int: ...

    def close(self) -> None: ...


class InMemoryArchiveStore:
    """Dependency-free default, suitable for tests and short-lived agents."""

    def __init__(self) -> None:
        self.payloads: dict[str, bytes] = {}

    def put(self, memory_id: str, payload: bytes) -> None:
        self.payloads[memory_id] = payload

    def get(self, memory_id: str) -> bytes | None:
        return self.payloads.get(memory_id)

    def contains(self, memory_id: str) -> bool:
        return memory_id in self.payloads

    def total_bytes(self) -> int:
        return sum(len(payload) for payload in self.payloads.values())

    def count(self) -> int:
        return len(self.payloads)

    def close(self) -> None:
        return None


class SQLiteArchiveStore:
    """Disk-backed compressed archive with atomic upserts and crash recovery."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        with self._connection:
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA synchronous=NORMAL")
            self._connection.execute(
                "CREATE TABLE IF NOT EXISTS archive ("
                "memory_id TEXT PRIMARY KEY, payload BLOB NOT NULL)"
            )

    def put(self, memory_id: str, payload: bytes) -> None:
        with self._lock, self._connection:
            self._connection.execute(
                "INSERT INTO archive(memory_id, payload) VALUES (?, ?) "
                "ON CONFLICT(memory_id) DO UPDATE SET payload=excluded.payload",
                (memory_id, sqlite3.Binary(payload)),
            )

    def get(self, memory_id: str) -> bytes | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT payload FROM archive WHERE memory_id = ?", (memory_id,)
            ).fetchone()
        return bytes(row[0]) if row else None

    def contains(self, memory_id: str) -> bool:
        with self._lock:
            row = self._connection.execute(
                "SELECT 1 FROM archive WHERE memory_id = ?", (memory_id,)
            ).fetchone()
        return row is not None

    def total_bytes(self) -> int:
        with self._lock:
            row = self._connection.execute(
                "SELECT COALESCE(SUM(length(payload)), 0) FROM archive"
            ).fetchone()
        return int(row[0])

    def count(self) -> int:
        with self._lock:
            row = self._connection.execute("SELECT COUNT(*) FROM archive").fetchone()
        return int(row[0])

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> "SQLiteArchiveStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
