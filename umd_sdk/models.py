from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class WriteReceipt:
    memory_id: str
    state: str
    reason: str
    revision: int

    @classmethod
    def from_dict(cls, value: dict[str, Any]):
        return cls(value["memory_id"], value["state"], value["reason"], int(value["revision"]))


@dataclass(frozen=True)
class Memory:
    id: str
    text: str
    state: str
    source: str
    scope: str | None
    kind: str
    entities: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    classification: str = "internal"
    raw: dict[str, Any] = field(default_factory=dict, compare=False)

    @classmethod
    def from_dict(cls, value: dict[str, Any]):
        return cls(
            value["id"], value["text"], value["state"], value["source"],
            value.get("scope"), value["kind"], tuple(value.get("entities", ())),
            tuple(value.get("tags", ())), value.get("classification", "internal"), value,
        )


@dataclass(frozen=True)
class SearchHit:
    memory_id: str
    text: str
    force: float
    orbit_radius: float
    potential_energy: float
    components: dict[str, float]
    explanation: dict[str, Any]

    @classmethod
    def from_dict(cls, value: dict[str, Any]):
        return cls(
            value["memory_id"], value["text"], float(value["force"]),
            float(value["orbit_radius"]), float(value["potential_energy"]),
            value["components"], value["explanation"],
        )


@dataclass(frozen=True)
class CapsuleHit:
    capsule_id: str
    kind: str
    source_ids: tuple[str, ...]
    text: str
    force: float
    components: dict[str, float]
    explanation: dict[str, Any]

    @classmethod
    def from_dict(cls, value: dict[str, Any]):
        return cls(
            value["capsule_id"], value["kind"], tuple(value["source_ids"]),
            value["text"], float(value["force"]), value["components"],
            value["explanation"],
        )
