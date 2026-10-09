from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any

SESSION_EXTRACTION_SYSTEM = """Build evidence-grounded hierarchical memories. Return one JSON object only."""
SESSION_EXTRACTION_PROMPT = """Core profile:
{core_profile}

Session {session_id} on {date}:
{rounds_text}

Return JSON with session_summary, semantic_candidates, and episodic_memories. Each memory must cite source round ids. Every semantic candidate must provide confidence as a numeric value from 0.0 to 1.0, not a qualitative label."""
SEMANTIC_MERGE_SYSTEM = """Merge a candidate semantic memory with existing memories. Return one JSON object only."""
SEMANTIC_MERGE_PROMPT = """Candidate:
{candidate}

Similar memories:
{existing}

Return action add|merge|ignore|conflict, target_memory_id, merged_memory, and reason."""
SESSION_SELECTION_SYSTEM = """Choose relevant sessions from the candidates. Return one JSON object only."""
SESSION_SELECTION_PROMPT = """Core memory:
{core_memory}

Question:
{question}

Candidates:
{candidates}

Return candidate_sessions as a list of ids."""
CAPTION_SYSTEM = """Decide whether retrieved image captions are sufficient. Return one JSON object only."""
CAPTION_PROMPT = """Question:
{question}

Question image caption:
{question_image_caption}

Allowed image ids:
{allowed_image_ids}

Evidence:
{evidence}

Return caption_sufficient, raw_evidence_image_ids, and reason."""
ANSWER_SYSTEM = """Answer the question concisely using the supplied memory evidence. Return only the answer."""
IMAGE_PROMPT = "Question or evidence image:"


def load_prompt(prompt_dir: Path, filename: str) -> str:
    path = prompt_dir / filename
    if not path.is_file():
        raise FileNotFoundError(f"Official Mem-Gallery prompt not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def format_constraint(category: str, prompt_dir: Path) -> str | None:
    filename = {"AR": "ar_prompt.txt", "CD": "cd_prompt.txt", "VS": "vs_prompt.txt"}.get(str(category).upper())
    return load_prompt(prompt_dir, filename) if filename else None


def build_answer_content(
    core_memory: str,
    evidence: list[dict[str, Any]],
    question: str,
    question_image: dict[str, Any] | None,
    raw_evidence_image_ids: list[str],
    speaker_a: str = "user",
    speaker_b: str = "assistant",
    constraint: str | None = None,
    *,
    question_image_label: str = "Question image:",
) -> list[dict[str, Any]]:
    lines = ["Core memory:\n" + core_memory, "Retrieved evidence:"]
    for item in evidence:
        lines.append(f"[{item['round_id']}] {item['text']}")
        for image in item.get("images", []):
            lines.append(f"image_id={image.get('img_id', '')}; caption={image.get('caption', '')}")
    prompt = f"The conversation is between {speaker_a} and {speaker_b}.\nQuestion: {question}"
    if constraint:
        prompt += "\n" + constraint
    content = [{"type": "text", "text": "\n\n".join(lines) + "\n\n" + prompt}]
    attached = set()
    if question_image and question_image.get("path"):
        content.extend((
            {"type": "text", "text": question_image_label},
            {"type": "image_url", "image_url": {"url": _data_url(question_image["path"])}},
        ))
        attached.add(str(question_image["path"]))
    requested = set(raw_evidence_image_ids)
    for item in evidence:
        for image in item.get("images", []):
            path = str(image.get("path", ""))
            if image.get("img_id") not in requested or not path or path in attached:
                continue
            content.extend((
                {"type": "text", "text": f"Evidence image {image.get('img_id')}:"},
                {"type": "image_url", "image_url": {"url": _data_url(path)}},
            ))
            attached.add(path)
    return content


def drop_lowest_match_image(
    image_ids: list[str], evidence: list[dict[str, Any]],
) -> tuple[list[str], str | None]:
    """Drop one selected evidence image, preserving the question image."""

    if not image_ids:
        return [], None
    scores = {
        str(image.get("img_id", "")): float(image.get("match_score", 0.0))
        for item in evidence
        for image in item.get("images", [])
    }
    dropped = min(image_ids, key=lambda image_id: scores.get(image_id, 0.0))
    return [image_id for image_id in image_ids if image_id != dropped], dropped


def _data_url(path: str) -> str:
    file_path = Path(path)
    mime = mimetypes.guess_type(file_path.name)[0] or "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(file_path.read_bytes()).decode("ascii")
