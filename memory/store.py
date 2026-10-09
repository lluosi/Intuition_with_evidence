from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, TypeVar


EPISODIC_RECORD_SCHEMA_VERSION = "intuition-with-evidence-episodic-record-v1"
T = TypeVar("T")


@dataclass
class CoreMemory:
    profile_text: str
    raw_profile: dict[str, Any] = field(default_factory=dict)


@dataclass
class RoundMemory:
    round_id: str
    session_id: str
    date: str
    user_text: str
    assistant_text: str
    raw_text: str
    image_ids: list[str] = field(default_factory=list)
    local_order: int = 0
    image_paths: list[str] = field(default_factory=list)
    image_captions: list[str] = field(default_factory=list)
    embedding: list[float] | None = None


@dataclass
class ResourceMemory:
    resource_id: str
    modality: str
    raw_path: str
    caption: str
    linked_round_id: str
    linked_session_id: str
    date: str
    tags: list[str] = field(default_factory=list)
    embedding_text: list[float] | None = None


@dataclass
class SessionMemory:
    session_id: str
    date: str
    summary: str
    topics: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    round_ids: list[str] = field(default_factory=list)
    resource_ids: list[str] = field(default_factory=list)
    embedding: list[float] | None = None


@dataclass
class SemanticMemory:
    memory_id: str
    memory_type: str
    content: str
    session_ids: list[str] = field(default_factory=list)
    date: str = ""
    entities: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    source_round_ids: list[str] = field(default_factory=list)
    linked_resource_ids: list[str] = field(default_factory=list)
    confidence: float = 0.8
    status: str = "active"
    embedding: list[float] | None = None


@dataclass
class EpisodicMemory:
    episode_id: str
    summary: str
    session_id: str
    date: str
    start_round_id: str
    end_round_id: str
    source_round_ids: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    linked_resource_ids: list[str] = field(default_factory=list)
    embedding: list[float] | None = None


@dataclass
class CaptionDecision:
    caption_sufficient: bool
    raw_evidence_image_ids: list[str] = field(default_factory=list)
    reason: str = ""


class MemoryStore:
    def __init__(self, core_memory: CoreMemory | None = None):
        self.core_memory = core_memory or CoreMemory(profile_text="")
        self.rounds: dict[str, RoundMemory] = {}
        self.round_order: list[str] = []
        self.resources: dict[str, ResourceMemory] = {}
        self.sessions: dict[str, SessionMemory] = {}
        self.semantic: dict[str, SemanticMemory] = {}
        self.episodic: dict[str, EpisodicMemory] = {}

    def add_round(self, memory: RoundMemory) -> None:
        if memory.round_id not in self.rounds:
            self.round_order.append(memory.round_id)
        self.rounds[memory.round_id] = memory

    def add_resource(self, memory: ResourceMemory) -> None:
        self.resources[memory.resource_id] = memory

    def add_session(self, memory: SessionMemory) -> None:
        self.sessions[memory.session_id] = memory

    def add_semantic(self, memory: SemanticMemory) -> None:
        memory.session_ids = _unique(memory.session_ids)
        memory.source_round_ids = _unique(memory.source_round_ids)
        memory.linked_resource_ids = _unique(memory.linked_resource_ids)
        self.semantic[memory.memory_id] = memory

    def add_episode(self, memory: EpisodicMemory) -> None:
        self.episodic[memory.episode_id] = memory

    def neighbors(self, round_id: str, window: int) -> list[RoundMemory]:
        if round_id not in self.round_order:
            return []
        index = self.round_order.index(round_id)
        return [self.rounds[rid] for rid in self.round_order[max(0, index - window): index + window + 1]]

    def search_sessions(self, query: list[float], top_k: int) -> list[tuple[SessionMemory, float]]:
        return _search(self.sessions.values(), query, top_k, "embedding")

    def search_semantic(self, query: list[float], top_k: int) -> list[tuple[SemanticMemory, float]]:
        return _search(self.semantic.values(), query, top_k, "embedding")

    def search_episodic(self, query: list[float], top_k: int) -> list[tuple[EpisodicMemory, float]]:
        return _search(self.episodic.values(), query, top_k, "embedding")

    def search_rounds(self, query: list[float], top_k: int) -> list[tuple[RoundMemory, float]]:
        return _search(self.rounds.values(), query, top_k, "embedding")

    def search_resources(self, query: list[float], top_k: int) -> list[tuple[ResourceMemory, float]]:
        return _search(self.resources.values(), query, top_k, "embedding_text")

    def to_dict(self) -> dict:
        return {
            "core_memory": asdict(self.core_memory),
            "rounds": [asdict(self.rounds[key]) for key in self.round_order],
            "resources": [asdict(value) for value in self.resources.values()],
            "sessions": [asdict(value) for value in self.sessions.values()],
            "semantic": [asdict(value) for value in self.semantic.values()],
            "episodic": [asdict(value) for value in self.episodic.values()],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MemoryStore":
        store = cls(CoreMemory(**data.get("core_memory", {"profile_text": ""})))
        for item in data.get("rounds", data.get("round_memories", [])):
            store.add_round(RoundMemory(**item))
        for item in data.get("resources", data.get("resource_memories", [])):
            store.add_resource(ResourceMemory(**item))
        for item in data.get("sessions", data.get("session_memories", [])):
            store.add_session(SessionMemory(**item))
        for item in data.get("semantic", data.get("semantic_memories", [])):
            item = dict(item)
            item["memory_type"] = item.pop("memory_type", item.pop("type", "fact"))
            store.add_semantic(SemanticMemory(**item))
        for item in data.get("episodic", data.get("episodic_memories", [])):
            store.add_episode(EpisodicMemory(**item))
        return store


def _search(values: Iterable[T], query: list[float], top_k: int, attribute: str) -> list[tuple[T, float]]:
    scored = [(value, cosine(getattr(value, attribute), query)) for value in values]
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:top_k]


def cosine(left: list[float] | None, right: list[float] | None) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    norm_left = math.sqrt(sum(value * value for value in left))
    norm_right = math.sqrt(sum(value * value for value in right))
    return dot / (norm_left * norm_right) if norm_left and norm_right else 0.0


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))
