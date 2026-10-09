"""Evaluate episodic, latent, hybrid, or adaptive memory."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path

from config import VENDORED_PROMPT_DIR, load_settings
from data.mem_gallery import MemGallery
from data.evidence import build_direct_answer_content, load_jsonl, validate_episodic_records
from memory.prompts import build_answer_content, drop_lowest_match_image, format_constraint, load_prompt
from memory.retrieval import MemoryRetriever, load_memory_store, make_episodic_record
from memory.router import RouterThresholds, load_thresholds, route, probe_router
from models.chat import memory_client
from models.embeddings import create_embedding_encoder
from models.qwen import (
    encode_content, generate_encoded_ids, is_cuda_oom_error, load_qwen25vl_config, resolve_dtype,
)

DEFAULT_PROMPT_DIR = VENDORED_PROMPT_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--memory-mode", choices=("episodic", "latent", "hybrid", "adaptive"), default="hybrid"
    )
    parser.add_argument("--retrieval-mode", choices=("online", "fixed"), default="online")
    parser.add_argument(
        "--memory-store", type=Path, default=None,
        help="episodic-memory JSON for online retrieval.",
    )
    parser.add_argument(
        "--episodic-records", type=Path, default=None,
        help="Frozen episodic-evidence JSONL for controlled evaluation.",
    )
    parser.add_argument(
        "--qa-split", type=Path, default=None,
        help="Converted QA split for online or latent evaluation.",
    )
    parser.add_argument("--data-name", default=None)
    parser.add_argument("--benchmark-root", type=Path, default=None)
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--benchmark-prompt-dir", type=Path, default=DEFAULT_PROMPT_DIR)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path, default=None)
    parser.add_argument(
        "--router-config", type=Path, default=None,
        help="Optional calibrated JSD/confidence thresholds for adaptive mode.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-qa", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--image-min-pixels", type=int, default=200704)
    parser.add_argument("--image-max-pixels", type=int, default=802816)
    parser.add_argument("--rerun-existing-results", action="store_true")
    args = parser.parse_args()
    for name in (
        "memory_store", "episodic_records", "qa_split", "benchmark_root",
        "benchmark_prompt_dir", "model_path", "adapter_path", "output_dir", "router_config",
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.expanduser().resolve())
    if args.memory_mode in {"latent", "hybrid", "adaptive"} and args.adapter_path is None:
        raise ValueError("--adapter-path is required for latent, hybrid, and adaptive evaluation")
    if args.router_config is not None and args.memory_mode != "adaptive":
        raise ValueError("--router-config requires --memory-mode adaptive")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if args.max_qa is not None and args.max_qa <= 0:
        raise ValueError("--max-qa must be positive")
    if args.memory_mode == "episodic" and args.adapter_path is not None:
        raise ValueError("episodic evaluation uses the base model; omit --adapter-path")
    if args.memory_mode == "latent":
        if args.qa_split is None or not args.data_name:
            raise ValueError("latent evaluation requires --qa-split and --data-name")
        if args.retrieval_mode != "online":
            raise ValueError("latent evaluation has no retrieval; omit --retrieval-mode")
        if args.memory_store is not None or args.episodic_records is not None:
            raise ValueError("latent evaluation does not accept --memory-store or --episodic-records")
    elif args.retrieval_mode == "online":
        if args.memory_store is None or args.qa_split is None or not args.data_name:
            raise ValueError("online retrieval requires --memory-store, --qa-split, and --data-name")
        if args.episodic_records is not None:
            raise ValueError("online retrieval does not accept --episodic-records")
    else:
        if args.episodic_records is None:
            raise ValueError("fixed retrieval requires --episodic-records")
        if args.memory_store is not None:
            raise ValueError("fixed retrieval does not accept --memory-store")
    return args


def load_split_task(path, data_name):
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            task = json.loads(line)
            if str(task.get("task_id")) == data_name:
                return task
    raise ValueError(f"No task {data_name!r} in split: {path}")


def _settings(args):
    settings = load_settings(embedding_model=args.embedding_model)
    return replace(settings, benchmark_root=args.benchmark_root) if args.benchmark_root is not None else settings


def load_latent_records(args):
    settings = _settings(args)
    adapter = MemGallery(settings.dialog_dir, settings.image_dir)
    dataset = adapter.load(args.data_name)
    profile = dataset.get("character_profile", {}) or {}
    name = str(profile.get("name", "")).strip()
    speaker_a = f"user ({name})" if name else "user"
    task = load_split_task(args.qa_split, args.data_name)
    records = []
    for index, qa in enumerate(task.get("qa_pairs", []), start=1):
        question = str(qa.get("raw_question", qa.get("question", ""))).strip()
        if not question:
            raise ValueError(f"QA {index} has an empty question")
        record = {
            "data_name": args.data_name,
            "qa_index": index,
            "qa_id": str(qa.get("qa_id") or f"{args.data_name}:{index}"),
            "session_id": str(qa.get("session_id", "")),
            "question": question,
            "answer": str(qa.get("answer", qa.get("original_answer", ""))).strip(),
            "point": str(qa.get("point", qa.get("category", ""))),
            "clue": qa.get("clue", []),
            "speaker_a": speaker_a,
            "speaker_b": "assistant",
            "question_image": adapter.question_image(qa),
            "core_memory": "",
            "evidence": [],
            "retrieved_ids": [],
            "retrieve_meta": {"retrieval_mode": "none"},
            "caption_sufficiency": {
                "caption_sufficient": True,
                "raw_evidence_image_ids": [],
                "reason": "latent mode",
            },
            "raw_evidence_image_ids": [],
        }
        if args.memory_mode == "adaptive":
            record["_source_qa"] = qa
        records.append(record)
    if not records:
        raise ValueError("The selected QA split is empty")
    return records


def load_records(args):
    if args.memory_mode == "latent" or (
        args.memory_mode == "adaptive" and args.retrieval_mode == "online"
    ):
        return load_latent_records(args)
    if args.retrieval_mode == "fixed":
        records = load_jsonl(args.episodic_records)
        validate_episodic_records(records)
        return records
    settings = _settings(args)
    task = load_split_task(args.qa_split, args.data_name)
    store = load_memory_store(args.memory_store)
    reader = MemGallery(settings.dialog_dir, settings.image_dir)
    retriever = MemoryRetriever(memory_client(settings), create_embedding_encoder(settings))
    records = [
        make_episodic_record(
            data_name=args.data_name, qa_index=index, qa=qa, store=store,
            reader=reader, retriever=retriever,
        )
        for index, qa in enumerate(task.get("qa_pairs", []), start=1)
    ]
    if not records:
        raise ValueError("The selected QA split is empty")
    return records


def online_evidence_loader(args):
    """Initialize retrieval only when an adaptive query needs episodic evidence."""
    store = reader = retriever = None

    def retrieve(record):
        nonlocal store, reader, retriever
        if retriever is None:
            settings = _settings(args)
            store = load_memory_store(args.memory_store)
            reader = MemGallery(settings.dialog_dir, settings.image_dir)
            retriever = MemoryRetriever(memory_client(settings), create_embedding_encoder(settings))
        return make_episodic_record(
            data_name=record["data_name"],
            qa_index=record["qa_index"],
            qa=record["_source_qa"],
            store=store,
            reader=reader,
            retriever=retriever,
        )

    return retrieve


def load_existing_results(path):
    if not path.exists():
        return []
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, list) else []


def save_results(path, results):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


def generate_answer(model, processor, system_prompt, content, input_device, args, torch):
    encoded = encode_content(processor, system_prompt, content, input_device, args)
    token_ids = generate_encoded_ids(model, processor, encoded, args.max_new_tokens, torch)
    return processor.tokenizer.decode(token_ids, skip_special_tokens=True).strip()


def answer_record(model, processor, system_prompt, record, prompt_dir, input_device, args, torch):
    if args.memory_mode == "latent":
        content = build_direct_answer_content(record, prompt_dir)
        answer = generate_answer(model, processor, system_prompt, content, input_device, args, torch)
        return answer, dict(record["caption_sufficiency"])
    state = dict(record["caption_sufficiency"])
    image_ids = list(dict.fromkeys(
        record.get("raw_evidence_image_ids", state.get("raw_evidence_image_ids", []))
    ))
    dropped = []
    while True:
        content = build_answer_content(
            record["core_memory"],
            record["evidence"],
            record["question"],
            record.get("question_image"),
            image_ids,
            record.get("speaker_a", "user"),
            record.get("speaker_b", "assistant"),
            format_constraint(record["point"], prompt_dir),
        )
        try:
            answer = generate_answer(model, processor, system_prompt, content, input_device, args, torch)
            updated = dict(state)
            updated["raw_evidence_image_ids"] = image_ids
            updated["dropped_raw_evidence_image_ids"] = dropped
            return answer, updated
        except Exception as error:
            if not is_cuda_oom_error(error) or not image_ids:
                raise
            image_ids, removed = drop_lowest_match_image(image_ids, record["evidence"])
            if removed is None:
                raise
            dropped.append(removed)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def without_evidence(record):
    return {
        **record,
        "core_memory": "",
        "evidence": [],
        "retrieved_ids": [],
        "raw_evidence_image_ids": [],
        "retrieve_meta": {"retrieval_mode": "none"},
        "caption_sufficiency": {
            "caption_sufficient": True,
            "raw_evidence_image_ids": [],
            "dropped_raw_evidence_image_ids": [],
            "reason": "adaptive router selected latent memory",
        },
    }


def answer_adaptive(
    model, processor, system_prompt, record, prompt_dir, input_device, args, torch,
    thresholds, retrieve=None,
):
    content = build_direct_answer_content(record, prompt_dir)
    encoded = encode_content(processor, system_prompt, content, input_device, args)
    token_ids, signals = probe_router(
        model, processor.tokenizer,
        lambda limit: generate_encoded_ids(model, processor, encoded, limit, torch),
        args.max_new_tokens, torch,
    )
    selected = route(signals["jsd_normalized"], signals["confidence"], thresholds)
    router_meta = {
        **signals,
        "policy": "first_token_jsd_native_confidence8_v1",
        "selected_memory_mode": selected,
        "thresholds": thresholds.to_dict(),
        "threshold_source": str(args.router_config) if args.router_config is not None else "default",
    }
    if selected == "latent":
        used_record = without_evidence(record)
        answer = processor.tokenizer.decode(token_ids, skip_special_tokens=True).strip()
        return answer, used_record["caption_sufficiency"], used_record, router_meta
    used_record = retrieve(record) if retrieve is not None else record
    branch_args = argparse.Namespace(**(vars(args) | {"memory_mode": selected}))
    if selected == "episodic":
        try:
            with model.disable_adapter():
                answer, state = answer_record(
                    model, processor, system_prompt, used_record,
                    prompt_dir, input_device, branch_args, torch,
                )
        finally:
            model.eval().requires_grad_(False)
    else:
        answer, state = answer_record(
            model, processor, system_prompt, used_record,
            prompt_dir, input_device, branch_args, torch,
        )
    return answer, state, used_record, router_meta


def result_item(record, answer, caption_state, args, router_meta=None):
    result = {
        "sample_id": record["data_name"],
        "qa_index": record["qa_index"],
        "qa_id": record["qa_id"],
        "session_id": record.get("session_id", ""),
        "speaker_a": record.get("speaker_a", "user"),
        "speaker_b": record.get("speaker_b", "assistant"),
        "question": record["question"],
        "question_image": (record.get("question_image") or {}).get("path", ""),
        "question_image_caption": (record.get("question_image") or {}).get("caption", ""),
        "original_answer": record["answer"],
        "category": record["point"],
        "system_answer": answer,
        "memory_mode": args.memory_mode,
        "retrieval_mode": "none" if args.memory_mode == "latent" else args.retrieval_mode,
        "retrieved_ids": record["retrieved_ids"],
        "clue": record.get("clue", []),
        "evidence": record["evidence"],
        "retrieve_meta": record["retrieve_meta"],
        "caption_sufficiency": caption_state,
    }
    if router_meta is not None:
        result["selected_memory_mode"] = router_meta["selected_memory_mode"]
        result["router"] = router_meta
        if router_meta["selected_memory_mode"] == "latent":
            result["retrieval_mode"] = "none"
    return result


def validate_adaptive_resume(existing, thresholds):
    """Prevent changing thresholds while silently reusing previous routed answers."""
    for item in existing:
        meta = item.get("router", {})
        if (
            item.get("memory_mode") != "adaptive"
            or meta.get("policy") != "first_token_jsd_native_confidence8_v1"
            or meta.get("thresholds") != thresholds.to_dict()
        ):
            raise ValueError(
                "Existing adaptive results use a different router configuration; "
                "use a new --output-dir or --rerun-existing-results"
            )


def adaptive_run_config(args, thresholds):
    """Bind resumed adaptive answers to their inputs, checkpoints, and generation."""
    def file_identity(path):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return {"path": str(path), "sha256": digest.hexdigest()}

    def checkpoint_identity(path):
        if not path.is_dir():
            raise FileNotFoundError(f"Local checkpoint directory not found: {path}")
        files = {}
        for item in sorted(path.iterdir()):
            if item.is_file() and item.suffix in {
                ".json", ".safetensors", ".bin", ".model", ".txt", ".jinja"
            }:
                stat = item.stat()
                files[item.name] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        return {"path": str(path), "files": files}

    inputs = {}
    for name in ("qa_split", "memory_store", "episodic_records"):
        path = getattr(args, name)
        if path is not None:
            inputs[name] = file_identity(path)
    prompts = {
        path.name: file_identity(path) for path in sorted(args.benchmark_prompt_dir.glob("*.txt"))
    }
    config = {
        "schema": "adaptive_inference_run_v1",
        "thresholds": thresholds.to_dict(),
        "model": checkpoint_identity(args.model_path),
        "adapter": checkpoint_identity(args.adapter_path),
        "inputs": inputs,
        "prompts": prompts,
        "data_name": args.data_name,
        "benchmark_root": str(args.benchmark_root) if args.benchmark_root is not None else None,
        "retrieval_mode": args.retrieval_mode,
        "max_new_tokens": args.max_new_tokens,
        "image_min_pixels": args.image_min_pixels,
        "image_max_pixels": args.image_max_pixels,
        "dtype": args.dtype,
    }
    if args.retrieval_mode == "online":
        settings = _settings(args)
        config["retrieval"] = {
            "benchmark_root": str(settings.benchmark_root),
            "memory_model": settings.memory_model,
            "embedding_backend": settings.embedding_backend,
            "embedding_model": settings.embedding_model,
            # Endpoint URLs can contain credentials; retain only their hashes.
            "memory_api_base_sha256": hashlib.sha256(settings.memory_api_base.encode()).hexdigest(),
            "embedding_api_base_sha256": hashlib.sha256(settings.embedding_api_base.encode()).hexdigest(),
        }
        config["inputs"]["dialog"] = file_identity(settings.dialog_dir / f"{args.data_name}.json")
    return config


def prepare_adaptive_output(
    output_path,
    existing,
    config,
    *,
    rerun,
):
    config_path = output_path.with_name("config.json")
    if existing:
        previous = (
            json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else None
        )
        if previous != config:
            raise ValueError(
                "Existing adaptive results use different inputs, checkpoints, or generation settings; "
                "use a new --output-dir or --rerun-existing-results"
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if rerun:
        # Clear old answers before replacing their identity, even if model load
        # or the first new generation subsequently fails.
        save_results(output_path, [])
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    thresholds = load_thresholds(args.router_config) if args.memory_mode == "adaptive" else None
    records = load_records(args)
    if args.max_qa is not None:
        records = records[:args.max_qa]
    run_name = args.memory_mode if args.memory_mode == "latent" else f"{args.memory_mode}_{args.retrieval_mode}"
    output_path = args.output_dir / run_name / "results.json"
    existing = [] if args.rerun_existing_results else load_existing_results(output_path)
    if thresholds is not None:
        validate_adaptive_resume(existing, thresholds)
        prepare_adaptive_output(
            output_path, existing, adaptive_run_config(args, thresholds), rerun=args.rerun_existing_results,
        )
    retrieve = (
        online_evidence_loader(args)
        if args.memory_mode == "adaptive" and args.retrieval_mode == "online"
        else None
    )
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(
        args.model_path, local_files_only=True, trust_remote_code=True, use_fast=True,
    )
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    load_kwargs = {
        "config": load_qwen25vl_config(args.model_path),
        "torch_dtype": resolve_dtype(torch, args.dtype),
        "local_files_only": True,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if torch.cuda.is_available():
        load_kwargs["device_map"] = "auto"
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model_path, **load_kwargs)
    if args.adapter_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(
            base_model, str(args.adapter_path), is_trainable=False, local_files_only=True,
        )
    else:
        model = base_model
    model.eval().requires_grad_(False)
    input_device = model.get_input_embeddings().weight.device
    system_prompt = load_prompt(args.benchmark_prompt_dir, "sys_prompt.txt")
    done = {str(item.get("qa_id", "")) for item in existing}
    for record in records:
        if record["qa_id"] in done:
            continue
        started = time.perf_counter()
        if thresholds is not None:
            answer, state, used_record, router_meta = answer_adaptive(
                model, processor, system_prompt, record, args.benchmark_prompt_dir,
                input_device, args, torch, thresholds, retrieve,
            )
            item = result_item(used_record, answer, state, args, router_meta)
        else:
            answer, state = answer_record(
                model, processor, system_prompt, record,
                args.benchmark_prompt_dir, input_device, args, torch,
            )
            item = result_item(record, answer, state, args)
        item["inference_seconds"] = round(time.perf_counter() - started, 6)
        existing.append(item)
        done.add(record["qa_id"])
        save_results(output_path, existing)
    print(f"Results written to: {output_path}")


if __name__ == "__main__":
    main()
