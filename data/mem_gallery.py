from __future__ import annotations

import json
import math
import os
import random
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

from memory.store import ResourceMemory, RoundMemory

SPLIT_NAME = "intra_scene_qa"
DEFAULT_SPLIT_SEED = 42
TASK_SCHEMA_VERSION = "intuition-with-evidence-mem-gallery-text-v1"

class MemGallery:
    def __init__(self, dialog_dir: Path, image_dir: Path):
        self.dialog_dir = dialog_dir
        self.image_dir = image_dir

    def load(self, data_name: str) -> dict[str, Any]:
        with (self.dialog_dir / f"{data_name}.json").open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def sessions(self, dataset: dict[str, Any], max_sessions: int | None = None, max_rounds: int | None = None) -> Iterator[dict[str, Any]]:
        sessions = dataset.get("multi_session_dialogues", [])
        if max_sessions is not None:
            sessions = sessions[:max_sessions]
        for session in sessions:
            rounds = session.get("dialogues", [])
            if max_rounds is not None:
                rounds = rounds[:max_rounds]
            yield {"session_id": session.get("session_id", ""), "date": session.get("date", ""), "rounds": rounds}

    def to_memories(self, dialogue: dict[str, Any], session_id: str, date: str, order: int, speaker: str) -> tuple[RoundMemory, list[ResourceMemory]]:
        round_id = str(dialogue.get("round", f"{session_id}:{order}"))
        user_text = str(dialogue.get("user", "") or "")
        assistant_text = str(dialogue.get("assistant", "") or "")
        raw_text = "\n".join(filter(None, [f"{speaker}: {user_text}" if user_text else "", f"assistant: {assistant_text}" if assistant_text else ""]))
        image_ids = as_list(dialogue.get("image_id"))
        raw_paths = as_list(dialogue.get("input_image"))
        captions = as_list(dialogue.get("image_caption"))
        paths = [self.resolve_image_path(str(path)) for path in raw_paths]
        round_memory = RoundMemory(
            round_id=round_id, session_id=session_id, date=date,
            user_text=user_text, assistant_text=assistant_text, raw_text=raw_text,
            image_ids=[str(value) for value in image_ids], local_order=order,
            image_paths=paths, image_captions=[str(value) for value in captions],
        )
        resources = []
        for index, path in enumerate(paths):
            resource_id = str(image_ids[index]) if index < len(image_ids) else f"{round_id}:IMG_{index + 1:03d}"
            resources.append(ResourceMemory(
                resource_id=resource_id, modality="image", raw_path=path,
                caption=str(captions[index]) if index < len(captions) else "",
                linked_round_id=round_id, linked_session_id=session_id, date=date,
            ))
        return round_memory, resources

    def resolve_image_path(self, image_path: str) -> str:
        if not image_path:
            return ""
        if os.path.isabs(image_path):
            return image_path
        if image_path.startswith("../image/"):
            image_path = image_path.removeprefix("../image/")
        return str((self.image_dir / image_path).resolve())

    @staticmethod
    def questions(dataset: dict[str, Any], max_questions: int | None = None) -> list[dict[str, Any]]:
        values = dataset.get("human-annotated QAs", [])
        return values if max_questions is None else values[:max_questions]

    def question_image(self, question: dict[str, Any]) -> dict[str, str] | None:
        value = question.get("question_image", "")
        if not value:
            return None
        caption = question.get("image_caption") or question.get("question_image_caption", "")
        return {"path": self.resolve_image_path(value), "caption": str(caption)}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return data


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def serialize_profile(profile: dict[str, Any]) -> str:
    lines = ["[PROFILE]"]
    if profile.get("name"):
        lines.append(f"Name: {clean_text(profile['name'])}")
    if profile.get("persona_summary"):
        lines.append(f"Persona summary: {clean_text(profile['persona_summary'])}")
    if profile.get("traits"):
        traits = ", ".join(clean_text(item) for item in as_list(profile["traits"]))
        lines.append(f"Traits: {traits}")
    if profile.get("conversation_style"):
        lines.append(
            f"Conversation style: {clean_text(profile['conversation_style'])}"
        )
    return "\n".join(lines)


def pair_values(ids: Sequence[Any], captions: Sequence[Any]) -> Iterable[tuple[str, str]]:
    max_length = max(len(ids), len(captions))
    for index in range(max_length):
        image_id = clean_text(ids[index]) if index < len(ids) else ""
        caption = clean_text(captions[index]) if index < len(captions) else ""
        yield image_id, caption


def serialize_context(
    data: dict[str, Any],
    source_path: Path,
) -> tuple[list[str], dict[str, Any]]:
    context = []
    profile = data.get("character_profile") or {}
    if profile:
        context.append(serialize_profile(profile))

    image_count = 0
    missing_caption_count = 0
    missing_image_paths: list[str] = []

    for session in data.get("multi_session_dialogues") or []:
        session_id = clean_text(session.get("session_id"))
        date = clean_text(session.get("date"))

        for dialogue in session.get("dialogues") or []:
            round_id = clean_text(dialogue.get("round"))
            header = f"[SESSION {session_id}]"
            if date:
                header += f"[DATE {date}]"
            if round_id:
                header += f"[ROUND {round_id}]"

            parts = [
                header,
                f"USER: {clean_text(dialogue.get('user'))}",
                f"ASSISTANT: {clean_text(dialogue.get('assistant'))}",
            ]

            image_ids = as_list(dialogue.get("image_id"))
            captions = as_list(dialogue.get("image_caption"))
            image_paths = as_list(dialogue.get("input_image"))
            for index, (image_id, caption) in enumerate(pair_values(image_ids, captions)):
                image_count += 1
                if not caption:
                    missing_caption_count += 1
                image_path = image_paths[index] if index < len(image_paths) else ""
                if image_path:
                    resolved = (source_path.parent / image_path).resolve()
                    if not resolved.exists():
                        missing_image_paths.append(image_path)

                image_label = image_id or f"{round_id}:IMAGE_{index + 1}"
                parts.append(f"[IMAGE {image_label}]")
                parts.append(f"CAPTION: {caption or '[IMAGE_CAPTION_MISSING]'}")

            context.append("\n".join(parts))

    metadata = {
        "dialogue_image_count": image_count,
        "missing_dialogue_caption_count": missing_caption_count,
        "missing_dialogue_image_paths": sorted(set(missing_image_paths)),
    }
    return context, metadata


def question_caption(qa: dict[str, Any]) -> str:
    caption = qa.get("image_caption")
    if isinstance(caption, list):
        caption = " ".join(clean_text(item) for item in caption)
    return clean_text(caption)


def serialize_qa(qa: dict[str, Any], task_id: str, index: int) -> dict[str, Any]:
    raw_question = clean_text(qa.get("question"))
    caption = question_caption(qa)
    if caption:
        question = f"Question image caption: {caption}\nQuestion: {raw_question}"
    else:
        question = raw_question

    return {
        "qa_id": f"{task_id}_{index:04d}",
        "point": clean_text(qa.get("point")) or "UNKNOWN",
        "question": question,
        "raw_question": raw_question,
        "answer": clean_text(qa.get("answer")),
        "question_image": qa.get("question_image"),
        "question_image_caption": caption or None,
        "session_id": as_list(qa.get("session_id")),
        "clue": as_list(qa.get("clue")),
    }


def build_task(source_path: Path) -> dict[str, Any]:
    data = load_json(source_path)
    task_id = source_path.stem
    context, context_metadata = serialize_context(data, source_path)
    qas = [
        serialize_qa(qa, task_id, index)
        for index, qa in enumerate(data.get("human-annotated QAs") or [], start=1)
    ]

    return {
        "schema_version": TASK_SCHEMA_VERSION,
        "task_id": task_id,
        "source_file": source_path.name,
        "context": context,
        "qa_pairs": qas,
        "metadata": {
            "character_profile": data.get("character_profile") or {},
            "num_context_segments": len(context),
            "num_qas": len(qas),
            "qa_point_counts": dict(Counter(qa["point"] for qa in qas)),
            **context_metadata,
        },
    }


def select_source_files(dialog_dir: Path, scene: str | None) -> list[Path]:
    files = sorted(dialog_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No JSON dialogue files found in {dialog_dir}")
    if scene is None:
        return files

    scene_path = Path(scene)
    candidates = {scene, scene_path.name, scene_path.stem}
    selected = [path for path in files if path.name in candidates or path.stem in candidates]
    if not selected:
        available = ", ".join(path.name for path in files)
        raise FileNotFoundError(f"Scene {scene!r} not found. Available: {available}")
    return selected


def split_qas_by_point(
    task: dict[str, Any],
    train_ratio: float,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1")

    rng = random.Random(seed)
    groups: dict[str, list[dict[str, Any]]] = {}
    for qa in task["qa_pairs"]:
        groups.setdefault(qa["point"], []).append(qa)

    train_qas: list[dict[str, Any]] = []
    test_qas: list[dict[str, Any]] = []
    split_counts: dict[str, dict[str, int]] = {}

    for point in sorted(groups):
        group = list(groups[point])
        rng.shuffle(group)
        # Singleton categories stay in train; other categories keep a test example.
        if len(group) == 1:
            train_count = 1
        else:
            train_count = min(
                len(group) - 1,
                max(1, math.floor(len(group) * train_ratio)),
            )

        point_train = group[:train_count]
        point_test = group[train_count:]
        train_qas.extend(point_train)
        test_qas.extend(point_test)
        split_counts[point] = {
            "total": len(group),
            "train": len(point_train),
            "test": len(point_test),
        }

    def with_qas(qas: list[dict[str, Any]], split: str) -> dict[str, Any]:
        return {
            **task,
            "split": split,
            "qa_pairs": qas,
            "metadata": {
                **task["metadata"],
                "split": split,
                "num_qas": len(qas),
                "qa_point_counts": dict(Counter(qa["point"] for qa in qas)),
                "split_strategy": "intra_scene_qa_stratified_by_point",
                "train_ratio": train_ratio,
                "split_seed": seed,
            },
        }

    summary = {
        "task_id": task["task_id"],
        "total_qas": len(task["qa_pairs"]),
        "train_qas": len(train_qas),
        "test_qas": len(test_qas),
        "by_point": split_counts,
    }
    return with_qas(train_qas, "train"), with_qas(test_qas, "test"), summary


def write_jsonl(records: Sequence[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_scene_outputs(
    task: dict[str, Any],
    train_task: dict[str, Any],
    test_task: dict[str, Any],
    summary: dict[str, Any],
    output_dir: Path,
    source_dialog_dir: Path,
    train_ratio: float,
    seed: int,
) -> dict[str, str]:
    scene_dir = output_dir / task["task_id"]
    canonical_path = scene_dir / "canonical" / "tasks.jsonl"
    train_path = scene_dir / "splits" / SPLIT_NAME / "train.jsonl"
    test_path = scene_dir / "splits" / SPLIT_NAME / "test.jsonl"
    summary_path = scene_dir / "splits" / SPLIT_NAME / "summary.json"

    write_jsonl([task], canonical_path)
    write_jsonl([train_task], train_path)
    write_jsonl([test_task], test_path)

    scene_summary = {
        "schema_version": TASK_SCHEMA_VERSION,
        "source_dialog_dir": str(source_dialog_dir),
        "scene_dir": str(scene_dir),
        "num_tasks": 1,
        "train_ratio": train_ratio,
        "seed": seed,
        "tasks": [summary],
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(scene_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "task_id": task["task_id"],
        "scene_dir": str(scene_dir),
        "canonical": str(canonical_path),
        "train": str(train_path),
        "test": str(test_path),
        "summary": str(summary_path),
        "task_summary": summary,
    }
