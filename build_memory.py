"""Build a scene's episodic memory and answer its annotated questions."""

import argparse
from functools import partial
import json
from pathlib import Path

from config import load_settings, safe_model_name
from data.mem_gallery import MemGallery
from memory.episodic import MemoryBuilder
from memory.prompts import ANSWER_SYSTEM, IMAGE_PROMPT, build_answer_content
from memory.retrieval import MemoryRetriever
from memory.store import MemoryStore
from models.chat import answer_client, memory_client
from models.embeddings import create_embedding_encoder
from models.qwen import QwenAnswer


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-name", required=True)
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--answer-backend", choices=("local", "api"), default="local")
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--image-min-pixels", type=int, default=200704)
    parser.add_argument("--image-max-pixels", type=int, default=802816)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--answer-model", default=None)
    parser.add_argument("--answer-api-base", default=None)
    parser.add_argument("--answer-api-key", default=None)
    parser.add_argument("--max-sessions", type=int, default=2)
    parser.add_argument("--max-rounds", type=int, default=4)
    parser.add_argument("--max-questions", type=int, default=2)
    parser.add_argument("--rebuild-memory", action="store_true")
    return parser.parse_args()


def answer_api(
    client, model, core_memory, evidence, question, question_image, image_ids,
    speaker_a="user", speaker_b="assistant", format_constraint=None,
    system_prompt=None, *, max_new_tokens=256,
):
    active_ids = list(dict.fromkeys(image_ids))
    dropped = []
    while True:
        content = build_answer_content(
            core_memory, evidence, question, question_image, active_ids,
            speaker_a, speaker_b, format_constraint, question_image_label=IMAGE_PROMPT,
        )
        try:
            answer = client.chat(
                [
                    {"role": "system", "content": system_prompt or ANSWER_SYSTEM},
                    {"role": "user", "content": content},
                ],
                model=model,
                max_tokens=max_new_tokens,
            )
            return answer, active_ids, dropped
        except Exception as error:
            if "context length" not in str(error).lower() or not active_ids:
                raise
            scores = {
                image.get("img_id"): image.get("match_score", 0.0)
                for item in evidence for image in item.get("images", [])
            }
            removed = min(active_ids, key=lambda image_id: scores.get(image_id, 0.0))
            active_ids.remove(removed)
            dropped.append(removed)


def qa_key(qa):
    return json.dumps([
        qa.get("question", ""),
        qa.get("category", qa.get("point", "")),
        qa.get("original_answer", qa.get("answer", "")),
        qa.get("question_image", ""),
    ], ensure_ascii=False, sort_keys=True)


def read_optional_prompt(path):
    return path.read_text(encoding="utf-8").strip() if path.exists() else None


def run_dataset(args, settings):
    gallery = MemGallery(settings.dialog_dir, settings.image_dir)
    client = memory_client(settings)
    encoder = create_embedding_encoder(settings)
    system_prompt = read_optional_prompt(settings.prompt_dir / "sys_prompt.txt")
    if args.answer_backend == "local":
        model = QwenAnswer(
            args.model_path,
            dtype=args.dtype,
            image_min_pixels=args.image_min_pixels,
            image_max_pixels=args.image_max_pixels,
            max_new_tokens=args.max_new_tokens,
        )
        answer = model.answer
        model_name = model.model_name
    else:
        answer = partial(
            answer_api, answer_client(settings), settings.answer_model,
            max_new_tokens=args.max_new_tokens,
        )
        model_name = settings.answer_model
    retriever = MemoryRetriever(client, encoder)
    dataset = gallery.load(args.data_name)
    output_dir = settings.result_dir / safe_model_name(model_name) / "episodic_memory"
    output_dir.mkdir(parents=True, exist_ok=True)
    memory_file = output_dir / f"{args.data_name}_memory.json"
    result_file = output_dir / f"{args.data_name}_results.json"
    if memory_file.exists() and not args.rebuild_memory:
        store = MemoryStore.from_dict(json.loads(memory_file.read_text(encoding="utf-8")))
    else:
        store = MemoryBuilder(gallery, client, encoder, settings.semantic_merge_threshold).build(
            dataset, max_sessions=args.max_sessions, max_rounds=args.max_rounds,
        )
        memory_file.write_text(json.dumps(store.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    results = json.loads(result_file.read_text(encoding="utf-8")) if result_file.exists() else []
    completed = {qa_key(qa) for qa in results}
    formats = {"AR": "ar_prompt.txt", "CD": "cd_prompt.txt", "VS": "vs_prompt.txt"}
    for index, qa in enumerate(gallery.questions(dataset, args.max_questions), start=1):
        if not qa.get("question") or qa_key(qa) in completed:
            continue
        question_image = gallery.question_image(qa)
        evidence, retrieval_meta = retriever.retrieve(
            store, qa["question"], question_image, neighbor_window=0, evidence_top_k=10,
        )
        decision = retriever.caption_decision(qa["question"], question_image, evidence)
        format_file = formats.get(str(qa.get("point", "") or "").upper())
        constraint = read_optional_prompt(settings.prompt_dir / format_file) if format_file else None
        text, used_images, dropped_images = answer(
            store.core_memory.profile_text, evidence, qa["question"], question_image,
            decision.raw_evidence_image_ids, speaker_a="user", speaker_b="assistant",
            format_constraint=constraint, system_prompt=system_prompt,
        )
        result = {
            "qa_index": index,
            "qa_id": str(qa.get("qa_id") or f"{args.data_name}:{index}"),
            "session_id": str(qa.get("session_id", "")),
            "question": qa["question"],
            "question_image": qa.get("question_image", ""),
            "question_image_caption": question_image.get("caption", "") if question_image else "",
            "original_answer": qa.get("answer", ""),
            "category": qa.get("point", ""),
            "system_answer": text,
            "retrieved_ids": [item["round_id"] for item in evidence],
            "evidence": evidence,
            "retrieve_meta": retrieval_meta,
            "caption_sufficiency": {
                "caption_sufficient": decision.caption_sufficient,
                "raw_evidence_image_ids": used_images,
                "dropped_raw_evidence_image_ids": dropped_images,
                "reason": decision.reason,
            },
        }
        results.append(result)
        completed.add(qa_key(result))
        result_file.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return result_file


def main():
    args = parse_args()
    settings = load_settings(
        answer_model=args.answer_model,
        answer_api_base=args.answer_api_base,
        answer_api_key=args.answer_api_key,
        embedding_model=args.embedding_model,
    )
    required = {"MEMORY_API_BASE": settings.memory_api_base}
    if args.answer_backend == "api":
        required.update(ANSWER_MODEL=settings.answer_model, ANSWER_API_BASE=settings.answer_api_base)
    missing = [name for name, value in required.items()
               if not value or (value.startswith("<") and value.endswith(">"))]
    if missing:
        raise ValueError("Set real values for: " + ", ".join(missing))
    if settings.embedding_model.startswith("<"):
        raise ValueError("Set EMBEDDING_MODEL in .env or pass --embedding-model")
    if args.answer_backend == "local":
        if args.model_path is None:
            raise ValueError("--model-path is required unless --answer-backend api is selected")
        args.model_path = args.model_path.expanduser().resolve()
        if not args.model_path.is_dir():
            raise FileNotFoundError(f"Local model directory not found: {args.model_path}")
    if args.image_min_pixels <= 0 or args.image_max_pixels <= 0:
        raise ValueError("--image-min-pixels and --image-max-pixels must be positive")
    if args.image_min_pixels > args.image_max_pixels:
        raise ValueError("--image-min-pixels cannot exceed --image-max-pixels")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    print(f"Results saved to: {run_dataset(args, settings)}")


if __name__ == "__main__":
    main()
