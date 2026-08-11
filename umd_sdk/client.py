from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.parse
import urllib.request
import uuid
from typing import Any, Callable, Iterable

from .errors import UMDAPIError
from .models import CapsuleHit, Memory, SearchHit, WriteReceipt


Transport = Callable[[str, str, dict[str, str], bytes], tuple[int, object, str]]


class UMDClient:
    """Typed synchronous client for local or remote UMD 3.9 services."""

    def __init__(
        self, endpoint: str, api_key: str, *, timeout: float = 10.0,
        transport: Transport | None = None,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.transport = transport

    def _request(
        self, method: str, path: str, payload: dict | None = None,
        *, idempotency_key: str | None = None,
    ) -> tuple[object, dict[str, Any]]:
        raw = json.dumps(payload, ensure_ascii=False).encode() if payload is not None else b""
        headers = {
            "Authorization": "Bearer " + self.api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-Request-ID": "sdk_" + uuid.uuid4().hex,
        }
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        if self.transport:
            status, body, _ = self.transport(method, path, headers, raw)
        else:
            request = urllib.request.Request(self.endpoint + path, raw or None, headers, method=method)
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    status, body = response.status, json.loads(response.read().decode())
            except urllib.error.HTTPError as error:
                status = error.code
                body = json.loads(error.read().decode())
        if status >= 400:
            value = body.get("error", {}) if isinstance(body, dict) else {}
            raise UMDAPIError(
                status, value.get("code", "http_error"), value.get("message", str(body)),
                value.get("request_id"), value.get("details"),
            )
        if isinstance(body, dict) and "data" in body:
            return body["data"], body.get("meta", {})
        return body, {}

    def add(
        self, text: str, *, scope: str | None = None, tags: Iterable[str] = (),
        classification: str = "internal", importance: float = 0.0,
        entities: Iterable[str] | None = None, fact_key: str | None = None,
        pinned: bool = False, idempotency_key: str | None = None,
    ) -> WriteReceipt:
        data, _ = self._request("POST", "/v1/memories", {
            "text": text, "scope": scope, "tags": list(tags),
            "classification": classification, "importance": importance,
            "entities": list(entities) if entities is not None else None,
            "fact_key": fact_key, "pinned": pinned,
        }, idempotency_key=idempotency_key)
        return WriteReceipt.from_dict(data)

    def search(
        self, query: str, *, top_k: int = 5, scope: str | None = None,
        budget_chars: int | None = None, chronological: bool = False,
    ) -> list[SearchHit]:
        data, _ = self._request("POST", "/v1/search", {
            "query": query, "top_k": top_k, "scope": scope,
            "budget_chars": budget_chars, "chronological": chronological,
        })
        return [SearchHit.from_dict(item) for item in data]

    def search_capsules(
        self, query: str, *, top_k: int = 10, scope: str | None = None,
        budget_chars: int | None = None,
    ) -> list[CapsuleHit]:
        data, _ = self._request("POST", "/v1/search", {
            "query": query, "top_k": top_k, "scope": scope,
            "budget_chars": budget_chars, "capsules": True,
        })
        return [CapsuleHit.from_dict(item) for item in data]

    def get(self, memory_id: str) -> Memory:
        data, _ = self._request("GET", "/v1/memories/" + urllib.parse.quote(memory_id))
        return Memory.from_dict(data)

    def list_memories(
        self, *, limit: int = 50, cursor: str | None = None,
        state: str | None = None, scope: str | None = None,
    ) -> tuple[list[Memory], str | None]:
        query = {"limit": str(limit)}
        if cursor:
            query["cursor"] = cursor
        if state:
            query["state"] = state
        if scope:
            query["scope"] = scope
        data, meta = self._request("GET", "/v1/memories?" + urllib.parse.urlencode(query))
        return [Memory.from_dict(item) for item in data], meta.get("next_cursor")

    def feedback(
        self, query: str, useful_memory_id: str,
        rejected_memory_ids: Iterable[str] = (),
    ) -> dict[str, float]:
        data, _ = self._request("POST", "/v1/retrieval-feedback", {
            "query": query, "useful_memory_id": useful_memory_id,
            "rejected_memory_ids": list(rejected_memory_ids),
        })
        return data

    def entities(self) -> list[dict[str, Any]]:
        return self._request("GET", "/v1/entities")[0]

    def ambiguities(self) -> list[dict[str, Any]]:
        return self._request("GET", "/v1/ambiguities")[0]

    def graph(self, limit: int = 500) -> dict[str, Any]:
        return self._request("GET", f"/v1/graph?limit={int(limit)}")[0]

    def communities(self) -> list[dict[str, Any]]:
        return self._request("GET", "/v1/communities")[0]

    def audit(self, limit: int = 100) -> list[dict[str, Any]]:
        return self._request("GET", f"/v1/audit?limit={int(limit)}")[0]

    def system(self) -> dict[str, Any]:
        return self._request("GET", "/v1/system")[0]


class AsyncUMDClient:
    """Async facade using threads so the SDK has no third-party dependency."""

    def __init__(self, *args, **kwargs) -> None:
        self.sync = UMDClient(*args, **kwargs)

    async def add(self, *args, **kwargs):
        return await asyncio.to_thread(self.sync.add, *args, **kwargs)

    async def search(self, *args, **kwargs):
        return await asyncio.to_thread(self.sync.search, *args, **kwargs)

    async def search_capsules(self, *args, **kwargs):
        return await asyncio.to_thread(self.sync.search_capsules, *args, **kwargs)

    async def get(self, *args, **kwargs):
        return await asyncio.to_thread(self.sync.get, *args, **kwargs)

    async def list_memories(self, *args, **kwargs):
        return await asyncio.to_thread(self.sync.list_memories, *args, **kwargs)

    async def feedback(self, *args, **kwargs):
        return await asyncio.to_thread(self.sync.feedback, *args, **kwargs)
