from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

from data.mem_gallery import MemGallery
from models.chat import ChatModel
from .store import CaptionDecision, EPISODIC_RECORD_SCHEMA_VERSION, MemoryStore
from .prompts import CAPTION_PROMPT, CAPTION_SYSTEM, SESSION_SELECTION_PROMPT, SESSION_SELECTION_SYSTEM


class MemoryRetriever:
    def __init__(self, client: ChatModel, encoder):
        self.client = client
        self.encoder = encoder

    def retrieve(
        self,
        store: MemoryStore,
        question: str,
        question_image: dict[str, str] | None,
        *,
        session_route_top_m: int = 10,
        selected_session_top_n: int = 3,
        fine_top_k: int = 8,
        evidence_top_k: int = 10,
        neighbor_window: int = 0,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        query = question
        if question_image and question_image.get("caption"):
            query += "\n" + question_image["caption"]
        modality = "text+image" if question_image else "text"
        query_vector = self.encoder.encode(query)
        routed = store.search_sessions(query_vector, session_route_top_m)
        selected = self._select_sessions(store, question, routed, selected_session_top_n)
        routed_ids = {memory.session_id for memory, _ in routed}
        selected_ids = set(selected)
        text_scores = {}
        resource_scores = {}
        provenance = {}
        image_scores = {}

        for memory, score in store.search_semantic(query_vector, fine_top_k):
            for round_id in memory.source_round_ids:
                _record_score(text_scores, provenance, round_id, score, f"semantic:{memory.memory_id}")
        for memory, score in store.search_episodic(query_vector, fine_top_k):
            for round_id in memory.source_round_ids:
                _record_score(text_scores, provenance, round_id, score, f"episodic:{memory.episode_id}")
        for memory, score in store.search_rounds(query_vector, max(evidence_top_k, 2)):
            _record_score(text_scores, provenance, memory.round_id, score, "round")
        for memory, score in store.search_resources(query_vector, fine_top_k):
            _record_score(resource_scores, provenance, memory.linked_round_id, score, f"resource:{memory.resource_id}")
            image_scores.setdefault(memory.linked_round_id, {})[memory.resource_id] = score

        if neighbor_window > 0:
            for round_id, score in list(text_scores.items()):
                for neighbor in store.neighbors(round_id, neighbor_window):
                    _record_score(text_scores, provenance, neighbor.round_id, score * 0.35, f"neighbor:{round_id}")

        ranked = []
        for round_id in set(text_scores) | set(resource_scores):
            text_score = text_scores.get(round_id, 0.0)
            resource_score = resource_scores.get(round_id, 0.0)
            if modality == "text+image":
                final_score = 0.8 * text_score + resource_score + (0.1 if resource_score > 0 else 0.0)
            else:
                final_score = text_score + 0.6 * resource_score + (0.1 if text_score > 0 else 0.0)
            memory = store.rounds.get(round_id)
            session_bonus = 0.0
            if memory and memory.session_id in routed_ids:
                session_bonus += 0.05
            if memory and memory.session_id in selected_ids:
                session_bonus += 0.05
            ranked.append((round_id, final_score + session_bonus, text_score, resource_score, session_bonus))
        ranked.sort(key=lambda item: item[1], reverse=True)

        evidence = []
        for round_id, score, text_score, resource_score, session_bonus in ranked[:evidence_top_k]:
            memory = store.rounds[round_id]
            images = []
            for index in range(max(len(memory.image_ids), len(memory.image_paths), len(memory.image_captions))):
                image_id = memory.image_ids[index] if index < len(memory.image_ids) else f"{round_id}:IMG_{index + 1:03d}"
                images.append({
                    "img_id": image_id,
                    "path": memory.image_paths[index] if index < len(memory.image_paths) else "",
                    "caption": memory.image_captions[index] if index < len(memory.image_captions) else "",
                    "matched": image_id in image_scores.get(round_id, {}),
                    "match_score": image_scores.get(round_id, {}).get(image_id, 0.0),
                })
            evidence.append({
                "round_id": round_id,
                "session_id": memory.session_id,
                "date": memory.date,
                "text": memory.raw_text,
                "score": score,
                "text_score": text_score,
                "resource_score": resource_score,
                "session_prior_bonus": session_bonus,
                "images": images,
                "provenance": list(dict.fromkeys(provenance.get(round_id, []))),
            })
        return evidence, {
            "query_plan": {
                "retrieval_modality": modality,
                "text_query": query,
                "resource_query": query,
            },
            "selected_sessions": selected,
            "routed_sessions": [{"session_id": memory.session_id, "score": score} for memory, score in routed],
        }

    def caption_decision(self, question: str, question_image: dict[str, str] | None, evidence: list[dict[str, Any]]) -> CaptionDecision:
        available = [image.get("img_id", "") for item in evidence for image in item.get("images", []) if image.get("img_id")]
        if not available:
            return CaptionDecision(True, [], "disabled or no evidence images")
        evidence_text = "\n".join(
            f"{item['round_id']}: " + "; ".join(image.get("img_id", "") + "=" + image.get("caption", "") for image in item.get("images", []))
            for item in evidence
        )
        result = self.client.chat_json([
            {"role": "system", "content": CAPTION_SYSTEM},
            {"role": "user", "content": CAPTION_PROMPT.format(
                question=question,
                question_image_caption=(question_image or {}).get("caption", "None"),
                allowed_image_ids="\n".join(available),
                evidence=evidence_text,
            )},
        ])
        requested = [str(value) for value in result.get("raw_evidence_image_ids", []) if str(value) in available]
        return CaptionDecision(bool(result.get("caption_sufficient", not requested)), list(dict.fromkeys(requested)), str(result.get("reason", "")))

    def _select_sessions(self, store: MemoryStore, question: str, routed, top_n: int) -> list[str]:
        if not routed:
            return []
        candidates = "\n".join(f"{memory.session_id} | {memory.summary}" for memory, _ in routed)
        try:
            result = self.client.chat_json([
                {"role": "system", "content": SESSION_SELECTION_SYSTEM},
                {"role": "user", "content": SESSION_SELECTION_PROMPT.format(
                    core_memory=store.core_memory.profile_text,
                    question=question,
                    candidates=candidates,
                )},
            ])
            selected = [str(value) for value in result.get("candidate_sessions", []) if str(value) in store.sessions]
            if selected:
                return selected[:top_n]
        except Exception as error:
            warnings.warn(
                f"Session selection failed; using embedding-ranked sessions: {error}",
                RuntimeWarning,
                stacklevel=2,
            )
        return [memory.session_id for memory, _ in routed[:top_n]]


def _record_score(scores: dict[str, float], provenance: dict[str, list[str]], key: str, score: float, reason: str) -> None:
    scores[key] = max(scores.get(key, 0.0), score)
    provenance.setdefault(key, []).append(reason)


def load_memory_store(path: Path) -> MemoryStore:
    with path.open("r", encoding="utf-8") as handle:
        return MemoryStore.from_dict(json.load(handle))


def make_episodic_record(
    *,
    data_name: str,
    qa_index: int,
    qa: dict[str, Any],
    store: MemoryStore,
    reader: MemGallery,
    retriever: MemoryRetriever,
) -> dict[str, Any]:
    """Retrieve one QA and return the episodic-memory interchange record."""

    question = str(qa.get("raw_question", qa.get("question", ""))).strip()
    if not question:
        raise ValueError(f"QA {qa_index} has an empty question")
    qa_id = str(qa.get("qa_id") or f"{data_name}:{qa_index}")
    question_image = reader.question_image(qa)
    evidence, retrieve_meta = retriever.retrieve(store, question, question_image)
    caption = retriever.caption_decision(question, question_image, evidence)
    character_name = str(store.core_memory.raw_profile.get("name", "")).strip()
    return {
        "schema_version": EPISODIC_RECORD_SCHEMA_VERSION,
        "data_name": data_name,
        "qa_index": qa_index,
        "qa_id": qa_id,
        "session_id": str(qa.get("session_id", "")),
        "question": question,
        "answer": str(qa.get("answer", qa.get("original_answer", ""))).strip(),
        "point": str(qa.get("point", qa.get("category", ""))),
        "clue": qa.get("clue", []),
        "speaker_a": f"user ({character_name})" if character_name else "user",
        "speaker_b": "assistant",
        "core_memory": store.core_memory.profile_text,
        "question_image": question_image,
        "evidence": evidence,
        "retrieved_ids": [item["round_id"] for item in evidence],
        "retrieve_meta": retrieve_meta,
        "caption_sufficiency": {
            "caption_sufficient": caption.caption_sufficient,
            "raw_evidence_image_ids": caption.raw_evidence_image_ids,
            "reason": caption.reason,
        },
        "raw_evidence_image_ids": caption.raw_evidence_image_ids,
    }
