from __future__ import annotations

import json
import math
from dataclasses import asdict
from typing import Any

from data.mem_gallery import MemGallery
from models.chat import ChatModel
from .store import CoreMemory, EpisodicMemory, MemoryStore, SemanticMemory, SessionMemory
from .prompts import (
    SEMANTIC_MERGE_PROMPT,
    SEMANTIC_MERGE_SYSTEM,
    SESSION_EXTRACTION_PROMPT,
    SESSION_EXTRACTION_SYSTEM,
)


class MemoryBuilder:
    def __init__(self, reader: MemGallery, client: ChatModel, encoder, merge_threshold: float = 0.72):
        self.reader = reader
        self.client = client
        self.encoder = encoder
        self.merge_threshold = merge_threshold
        self.semantic_counter = 0
        self.episode_counter = 0

    def build(self, dataset: dict[str, Any], *, max_sessions: int = 2, max_rounds: int = 4) -> MemoryStore:
        profile = dataset.get("character_profile", {}) or {}
        store = MemoryStore(CoreMemory(profile_text=self._profile_text(profile), raw_profile=profile))
        speaker = f"user ({profile.get('name')})" if profile.get("name") else "user"
        for session in self.reader.sessions(dataset, max_sessions, max_rounds):
            round_memories = []
            resources = []
            for order, dialogue in enumerate(session["rounds"], start=1):
                round_memory, round_resources = self.reader.to_memories(dialogue, session["session_id"], session["date"], order, speaker)
                round_memory.embedding = self.encoder.encode(round_memory.raw_text)
                for resource in round_resources:
                    resource.embedding_text = self.encoder.encode(resource.caption or resource.resource_id)
                    store.add_resource(resource)
                store.add_round(round_memory)
                round_memories.append(round_memory)
                resources.extend(round_resources)
            self._add_session_level(store, session, round_memories, resources)
        return store

    def _add_session_level(self, store: MemoryStore, session: dict[str, Any], rounds, resources) -> None:
        round_text = "\n\n".join(f"[{item.round_id}]\n{item.raw_text}" for item in rounds)
        extracted = self.client.chat_json([
            {"role": "system", "content": SESSION_EXTRACTION_SYSTEM},
            {"role": "user", "content": SESSION_EXTRACTION_PROMPT.format(
                core_profile=store.core_memory.profile_text,
                session_id=session["session_id"],
                date=session["date"],
                rounds_text=round_text,
            )},
        ])
        if not isinstance(extracted, dict):
            extracted = {}
        summary_data = extracted.get("session_summary", {})
        summary = str(summary_data.get("summary", "")) or (rounds[0].raw_text[:700] if rounds else "")
        store.add_session(SessionMemory(
            session_id=session["session_id"], date=session["date"], summary=summary,
            topics=[str(value) for value in summary_data.get("topics", [])],
            entities=[str(value) for value in summary_data.get("entities", [])],
            round_ids=[item.round_id for item in rounds],
            resource_ids=[item.resource_id for item in resources],
            embedding=self.encoder.encode(summary or session["session_id"]),
        ))
        for candidate in extracted.get("semantic_candidates", []):
            self._add_semantic(store, candidate, session)
        for episode in extracted.get("episodic_memories", []):
            self.episode_counter += 1
            content = str(episode.get("summary", ""))
            store.add_episode(EpisodicMemory(
                episode_id=f"episode_{self.episode_counter}", summary=content,
                session_id=session["session_id"], date=session["date"],
                start_round_id=str(episode.get("start_round_id", rounds[0].round_id if rounds else "")),
                end_round_id=str(episode.get("end_round_id", rounds[-1].round_id if rounds else "")),
                source_round_ids=[str(value) for value in episode.get("source_round_ids", [])],
                topics=[str(value) for value in episode.get("topics", [])],
                entities=[str(value) for value in episode.get("entities", [])],
                linked_resource_ids=[str(value) for value in episode.get("linked_resource_ids", [])],
                embedding=self.encoder.encode(content or session["session_id"]),
            ))

    def _add_semantic(self, store: MemoryStore, candidate: dict[str, Any], session: dict[str, Any]) -> None:
        content = str(candidate.get("content", "")).strip()
        if not content:
            return
        embedding = self.encoder.encode(content)
        similar = [(memory, score) for memory, score in store.search_semantic(embedding, 3) if score >= self.merge_threshold]
        if similar:
            decision = self.client.chat_json([
                {"role": "system", "content": SEMANTIC_MERGE_SYSTEM},
                {"role": "user", "content": SEMANTIC_MERGE_PROMPT.format(
                    candidate=json.dumps(candidate, ensure_ascii=False),
                    existing=json.dumps([asdict(memory) | {"similarity": score} for memory, score in similar], ensure_ascii=False),
                )},
            ])
            action = decision.get("action")
            if action == "ignore":
                return
            target = decision.get("target_memory_id")
            if target and action in {"merge", "conflict"} and target in store.semantic:
                memory = store.semantic[target]
                merged = decision.get("merged_memory") or candidate
                memory.content = str(merged.get("content", memory.content))
                memory.session_ids = list(dict.fromkeys(memory.session_ids + [session["session_id"]]))
                memory.source_round_ids = list(dict.fromkeys(memory.source_round_ids + [str(value) for value in merged.get("source_round_ids", [])]))
                memory.linked_resource_ids = list(dict.fromkeys(memory.linked_resource_ids + [str(value) for value in merged.get("linked_resource_ids", [])]))
                memory.embedding = self.encoder.encode(memory.content)
                if action == "conflict":
                    memory.status = "conflict_candidate"
                return
        self.semantic_counter += 1
        store.add_semantic(SemanticMemory(
            memory_id=f"semantic_{self.semantic_counter}", memory_type=str(candidate.get("type", "fact")), content=content,
            session_ids=[session["session_id"]], date=session["date"],
            entities=[str(value) for value in candidate.get("entities", [])],
            topics=[str(value) for value in candidate.get("topics", [])],
            source_round_ids=[str(value) for value in candidate.get("source_round_ids", [])],
            linked_resource_ids=[str(value) for value in candidate.get("linked_resource_ids", [])],
            confidence=_normalize_confidence(candidate.get("confidence")), embedding=embedding,
        ))

    @staticmethod
    def _profile_text(profile: dict[str, Any]) -> str:
        parts = [
            f"User name: {profile.get('name')}." if profile.get("name") else "",
            f"Persona: {profile.get('persona_summary')}" if profile.get("persona_summary") else "",
            f"Stable traits: {', '.join(profile['traits'])}." if isinstance(profile.get("traits"), list) and profile.get("traits") else "",
            f"Conversation style: {profile.get('conversation_style')}" if profile.get("conversation_style") else "",
        ]
        return "\n".join(value for value in parts if value)

_CONFIDENCE_LABELS = {"low": 0.4, "medium": 0.6, "high": 0.8}


def _normalize_confidence(value: Any, default: float = 0.8) -> float:
    """Convert LLM confidence values to the serialized numeric schema."""
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _CONFIDENCE_LABELS:
            return _CONFIDENCE_LABELS[normalized]
        value = normalized
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(confidence):
        return default
    return min(1.0, max(0.0, confidence))
