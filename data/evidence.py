from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from memory.prompts import build_answer_content, format_constraint
from memory.store import EPISODIC_RECORD_SCHEMA_VERSION

EPISODIC_SCHEMA_VERSION = EPISODIC_RECORD_SCHEMA_VERSION

def load_task(path: Path, task_id: str | None) -> dict[str, Any]:
    """Load the selected task from a JSONL QA split."""
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            task = json.loads(line)
            if task_id is None or task.get("task_id") == task_id:
                return task
    wanted = task_id or "the first task"
    raise ValueError(f"Could not find {wanted} in {path}")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            records.append(item)
    if not records:
        raise ValueError(f"No records found in episodic record artifact: {path}")
    return records


def validate_episodic_records(records: Iterable[dict[str, Any]]) -> None:
    seen: set[str] = set()
    required = {
        "schema_version",
        "data_name",
        "qa_id",
        "question",
        "answer",
        "point",
        "core_memory",
        "evidence",
        "question_image",
        "caption_sufficiency",
    }
    for index, record in enumerate(records, start=1):
        missing = sorted(required - set(record))
        if missing:
            raise ValueError(f"Cache record {index} is missing fields: {missing}")
        if record["schema_version"] != EPISODIC_SCHEMA_VERSION:
            raise ValueError(
                f"Cache record {index} has schema {record['schema_version']!r}; "
                f"expected {EPISODIC_SCHEMA_VERSION!r}"
            )
        qa_id = str(record.get("qa_id", ""))
        if not qa_id:
            raise ValueError(f"Cache record {index} has no qa_id")
        if qa_id in seen:
            raise ValueError(f"Duplicate qa_id in episodic record artifact: {qa_id}")
        seen.add(qa_id)
        if not str(record.get("answer", "")).strip():
            raise ValueError(f"Cache record {qa_id} has an empty answer")
        if not isinstance(record.get("evidence"), list):
            raise ValueError(f"Cache record {qa_id} has non-list evidence")
        question_image = record.get("question_image")
        if question_image is not None and not isinstance(question_image, dict):
            raise ValueError(f"Cache record {qa_id} has invalid question_image")


def validate_episodic_records_against_task(
    records: list[dict[str, Any]],
    task: dict[str, Any],
) -> None:
    """Ensure cached supervision is exactly the requested outer-train split."""

    task_qas = {str(item.get("qa_id", "")): item for item in task.get("qa_pairs", [])}
    for record in records:
        qa_id = str(record["qa_id"])
        if qa_id not in task_qas:
            raise ValueError(f"Episodic record QA {qa_id} is not present in the training split")
        qa = task_qas[qa_id]
        expected_answer = str(qa.get("answer", "")).strip()
        if str(record.get("answer", "")).strip() != expected_answer:
            raise ValueError(f"Answer mismatch between cache and split for {qa_id}")
        expected_question = str(qa.get("raw_question", qa.get("question", ""))).strip()
        if str(record.get("question", "")).strip() != expected_question:
            raise ValueError(f"Question mismatch between cache and split for {qa_id}")


def _raw_evidence_image_ids(record: dict[str, Any]) -> list[str]:
    explicit = record.get("raw_evidence_image_ids")
    if isinstance(explicit, list):
        return [str(item) for item in explicit]
    caption_state = record.get("caption_sufficiency", {})
    return [str(item) for item in caption_state.get("raw_evidence_image_ids", [])]


def build_episodic_answer_content(
    record: dict[str, Any],
    prompt_dir: Path,
) -> list[dict[str, Any]]:
    constraint = format_constraint(record.get("point", ""), prompt_dir)
    return build_answer_content(
        str(record.get("core_memory", "")),
        list(record.get("evidence", [])),
        str(record.get("question", "")),
        record.get("question_image"),
        _raw_evidence_image_ids(record),
        str(record.get("speaker_a", "user")),
        str(record.get("speaker_b", "assistant")),
        constraint,
    )


def build_direct_answer_content(
    record: dict[str, Any],
    prompt_dir: Path,
) -> list[dict[str, Any]]:
    """Build the standard prompt with empty episodic-evidence fields for latent-only QA."""
    constraint = format_constraint(record.get("point", ""), prompt_dir)
    return build_answer_content(
        "",
        [],
        str(record.get("question", "")),
        record.get("question_image"),
        [],
        str(record.get("speaker_a", "user")),
        str(record.get("speaker_b", "assistant")),
        constraint,
    )


def count_attached_images(content: list[dict[str, Any]]) -> int:
    return sum(item.get("type") in {"image", "image_url"} for item in content)
