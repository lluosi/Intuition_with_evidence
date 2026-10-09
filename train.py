"""Train a LoRA initialization with context adaptation and answer-only QA loss."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from config import VENDORED_PROMPT_DIR
from data.evidence import (
    build_direct_answer_content,
    build_episodic_answer_content,
    count_attached_images,
    load_task,
    load_jsonl,
    validate_episodic_records,
    validate_episodic_records_against_task,
)
from memory.latent import (
    chunk_ids,
    gradient_norm,
    parameter_norm,
    print_gpu_memory,
    restore,
    inner_update,
    seed_everything,
    snapshot,
    lora_parameters,
    write_jsonl,
)
from memory.prompts import drop_lowest_match_image, load_prompt
from models.qwen import (
    encode_answer_only_example,
    is_cuda_oom_error,
    load_qwen25vl_config,
    resolve_dtype,
)


DEFAULT_PROMPT_DIR = VENDORED_PROMPT_DIR
DEFAULT_TRAINING_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--data-name", required=True)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--episodic-records", type=Path, required=True)
    parser.add_argument("--task-id", default=None)
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--micro-batch-chunks", type=int, default=4)
    parser.add_argument("--inner-steps", type=int, default=4)
    parser.add_argument("--inner-lr", type=float, default=1e-4)
    parser.add_argument("--outer-steps", type=int, default=5)
    parser.add_argument("--outer-lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--outer-supervision", choices=("episodic", "qa-only"), default="episodic")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-train-qa", type=int, default=None)
    parser.add_argument("--lora-rank", type=int, default=256)
    parser.add_argument("--lora-alpha", type=int, default=256)
    parser.add_argument("--last-n-layers", type=int, default=4)
    parser.add_argument("--target-modules", nargs="+", default=["q_proj", "k_proj", "v_proj", "o_proj"])
    parser.add_argument("--benchmark-prompt-dir", type=Path, default=DEFAULT_PROMPT_DIR)
    parser.add_argument("--image-min-pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--image-max-pixels", type=int, default=1024 * 28 * 28)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument(
        "--save-every",
        type=int,
        default=0,
        help="Save an intermediate meta-init every N outer steps; 0 disables it.",
    )
    args = parser.parse_args()
    args.model_path = args.model_path.expanduser().resolve()
    args.train_data = args.train_data.expanduser().resolve()
    args.episodic_records = args.episodic_records.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.benchmark_prompt_dir = args.benchmark_prompt_dir.expanduser().resolve()
    return args


def validate_args(args: argparse.Namespace) -> None:
    if args.inner_steps <= 0 or args.outer_steps <= 0:
        raise ValueError("inner-steps and outer-steps must be > 0")
    if args.micro_batch_chunks <= 0:
        raise ValueError("micro-batch-chunks must be > 0")
    if args.image_min_pixels <= 0 or args.image_max_pixels <= 0:
        raise ValueError("image pixel limits must be > 0")
    if args.image_min_pixels > args.image_max_pixels:
        raise ValueError("image-min-pixels cannot exceed image-max-pixels")
    if not args.model_path.is_dir():
        raise FileNotFoundError(f"Model path not found: {args.model_path}")
    if not args.train_data.is_file():
        raise FileNotFoundError(f"Training split not found: {args.train_data}")
    if not args.episodic_records.is_file():
        raise FileNotFoundError(f"episodic record artifact not found: {args.episodic_records}")
    if not args.benchmark_prompt_dir.is_dir():
        raise FileNotFoundError(f"Prompt directory not found: {args.benchmark_prompt_dir}")


def outer_backward(model, processor, records, system_prompt, prompt_dir, input_device, args, torch):
    model.train()
    num_examples = len(records)
    total_loss = 0.0
    total_tokens = 0
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise RuntimeError("No trainable parameters found for outer QA backward")
    for index, record in enumerate(records, start=1):
        if args.outer_supervision == "qa-only":
            current_ids = []
        else:
            raw_ids = record.get("raw_evidence_image_ids")
            if not isinstance(raw_ids, list):
                caption_state = record.get("caption_sufficiency", {})
                raw_ids = caption_state.get("raw_evidence_image_ids", [])
            current_ids = list(dict.fromkeys(str(item) for item in raw_ids))
        dropped_ids = []
        expected_minimum = 1 if record.get("question_image") else 0
        while True:
            if args.outer_supervision == "qa-only":
                user_content = build_direct_answer_content(record, prompt_dir)
            else:
                attempt_record = dict(record)
                attempt_record["raw_evidence_image_ids"] = current_ids
                user_content = build_episodic_answer_content(attempt_record, prompt_dir)
            if count_attached_images(user_content) < expected_minimum:
                raise RuntimeError(
                    f"Question image was not attached for cached QA {record.get('qa_id')}"
                )
            batch = labels = outputs = raw_loss = example_grads = None
            try:
                batch = encode_answer_only_example(
                    processor,
                    system_prompt,
                    user_content,
                    str(record["answer"]),
                    input_device,
                    args.image_min_pixels,
                    args.image_max_pixels,
                )
                labels = batch["labels"]
                answer_tokens = int(labels.ne(-100).sum().item())
                outputs = model(**batch, use_cache=False)
                raw_loss = outputs.loss
                # Accumulate only after the whole example succeeds, so an OOM retry
                # cannot leave partial gradients from a failed attempt.
                example_grads = torch.autograd.grad(
                    raw_loss / num_examples,
                    trainable_parameters,
                    allow_unused=True,
                )
            except Exception as error:
                if (
                    not is_cuda_oom_error(error)
                    or args.outer_supervision != "episodic"
                    or not current_ids
                ):
                    raise
                current_ids, dropped = drop_lowest_match_image(
                    current_ids,
                    list(record.get("evidence", [])),
                )
                if dropped is None:
                    raise
                dropped_ids.append(dropped)
                del outputs, raw_loss, labels, batch, user_content
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(
                    "outer QA CUDA OOM with raw evidence image; "
                    f"qa_id={record.get('qa_id')} drop={dropped} and retry; "
                    f"remaining_raw_evidence_images={len(current_ids)}; "
                    "question image remains attached."
                )
                continue
            for parameter, gradient in zip(trainable_parameters, example_grads):
                if gradient is None:
                    continue
                detached_gradient = gradient.detach()
                if parameter.grad is None:
                    parameter.grad = detached_gradient
                else:
                    parameter.grad.add_(detached_gradient)
            loss_value = float(raw_loss.detach().float().item())
            total_loss += loss_value * answer_tokens
            total_tokens += answer_tokens
            print(
                f"outer QA backward {index}/{num_examples}: "
                f"qa_id={record.get('qa_id')} loss={loss_value:.6f} "
                f"answer_tokens={answer_tokens} images={count_attached_images(user_content)} "
                f"raw_evidence_images={len(current_ids)} dropped={dropped_ids} "
                f"supervision={args.outer_supervision}"
            )
            if dropped_ids:
                record["raw_evidence_image_ids"] = list(current_ids)
                record["dropped_raw_evidence_image_ids"] = list(dropped_ids)
            del example_grads, outputs, raw_loss, labels, batch, user_content
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            break
    return total_loss / max(total_tokens, 1)


def main() -> None:
    args = parse_args()
    validate_args(args)

    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    seed_everything(torch, DEFAULT_TRAINING_SEED)
    task_id = args.task_id or args.data_name
    train_task = load_task(args.train_data, task_id)
    records = load_jsonl(args.episodic_records)
    validate_episodic_records(records)
    validate_episodic_records_against_task(records, train_task)
    wrong_dataset = sorted(
        {str(record["data_name"]) for record in records if record["data_name"] != args.data_name}
    )
    if wrong_dataset:
        raise ValueError(
            f"episodic record artifact contains data_name values different from {args.data_name}: "
            f"{wrong_dataset}"
        )
    if args.max_train_qa is not None:
        records = records[: args.max_train_qa]
    if not records:
        raise ValueError("No episodic training records selected")
    question_image_qas = sum(bool(record.get("question_image")) for record in records)

    system_prompt = load_prompt(args.benchmark_prompt_dir, "sys_prompt.txt")
    context_text = "\n\n".join(train_task["context"])
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    run_name = args.run_name or (
        f"latent_{args.outer_supervision}_{args.data_name}_rank{args.lora_rank}"
        f"_layers{args.last_n_layers}_inner{args.inner_steps}"
        f"_meta{args.outer_steps}_lr{args.outer_lr:g}_{stamp}"
    )
    run_root = args.output_root / run_name
    init_dir = run_root / "adapters" / "init"
    checkpoints_dir = run_root / "adapters" / "checkpoints"
    metrics_path = run_root / "metrics.jsonl"
    run_root.mkdir(parents=True, exist_ok=True)

    print(f"run_root: {run_root}")
    print(f"task_id: {train_task['task_id']}")
    print(f"train_data: {args.train_data}")
    print(f"episodic_records: {args.episodic_records}")
    print(f"outer_supervision: {args.outer_supervision}")
    print(f"benchmark_prompt_dir: {args.benchmark_prompt_dir}")
    print(f"train_qas: {len(records)}")
    print(f"question_image_qas: {question_image_qas}")
    print(f"CUDA devices visible: {torch.cuda.device_count()}")

    model_path = args.model_path.resolve()
    processor = AutoProcessor.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
        use_fast=True,
    )
    tokenizer = processor.tokenizer
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has neither pad_token_id nor eos_token_id")
        tokenizer.pad_token = tokenizer.eos_token

    encoded_context = tokenizer(
        context_text,
        add_special_tokens=True,
        truncation=False,
        return_attention_mask=False,
    )
    chunks = chunk_ids(encoded_context["input_ids"], args.chunk_size)
    print(f"context tokens: {len(encoded_context['input_ids'])}")
    print(f"context chunks: {len(chunks)}")

    model_config = load_qwen25vl_config(model_path, normalize_rope=True)
    if args.dtype == "bf16" and torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
        raise RuntimeError("bf16 is not supported by the visible CUDA device")
    load_kwargs = {
        "config": model_config,
        "torch_dtype": resolve_dtype(torch, args.dtype),
        "trust_remote_code": True,
        "local_files_only": True,
        "low_cpu_mem_usage": True,
    }
    if torch.cuda.is_available():
        load_kwargs["device_map"] = "auto"
    print("Loading model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        **load_kwargs,
    )
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    num_layers = getattr(model.config, "num_hidden_layers", None)
    if num_layers is None:
        raise ValueError("Could not read num_hidden_layers from model config")
    layers_to_transform = list(
        range(max(0, num_layers - args.last_n_layers), num_layers)
    )
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        target_modules=args.target_modules,
        layers_to_transform=layers_to_transform,
        layers_pattern="layers",
        use_rslora=True,
    )
    model = get_peft_model(model, peft_config, autocast_adapter_dtype=True)
    for name, parameter in model.named_parameters():
        if "lora" not in name.lower():
            parameter.requires_grad = False

    named_params = lora_parameters(model)
    if not named_params:
        raise RuntimeError("No trainable LoRA parameters were found")
    trainable_count = sum(parameter.numel() for _, parameter in named_params)
    total_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"LoRA layers: {layers_to_transform}")
    print(f"LoRA targets: {args.target_modules}")
    print(
        f"trainable LoRA params: {trainable_count:,d} / {total_count:,d} "
        f"({100 * trainable_count / total_count:.4f}%)"
    )
    print(f"model device map: {getattr(model, 'hf_device_map', 'single-device')}")
    print_gpu_memory(torch)

    input_device = model.get_input_embeddings().weight.device
    outer_optimizer = torch.optim.AdamW(
        [parameter for _, parameter in named_params],
        lr=args.outer_lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    episodic_manifest_path = args.episodic_records.with_suffix(".manifest.json")
    episodic_manifest = None
    if episodic_manifest_path.is_file():
        episodic_manifest = json.loads(episodic_manifest_path.read_text(encoding="utf-8"))
    config_snapshot = {
        "created_at": stamp,
        "run_name": run_name,
        "model_path": str(model_path),
        "train_data": str(args.train_data),
        "episodic_records": str(args.episodic_records),
        "episodic_records_manifest": episodic_manifest,
        "benchmark_prompt_dir": str(args.benchmark_prompt_dir),
        "task_id": train_task["task_id"],
        "train_qas": len(records),
        "question_image_qas": question_image_qas,
        "context_tokens": len(encoded_context["input_ids"]),
        "context_chunks": len(chunks),
        "seed": DEFAULT_TRAINING_SEED,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "last_n_layers": args.last_n_layers,
        "target_modules": args.target_modules,
        "inner_steps": args.inner_steps,
        "inner_lr": args.inner_lr,
        "outer_steps": args.outer_steps,
        "outer_lr": args.outer_lr,
        "image_min_pixels": args.image_min_pixels,
        "image_max_pixels": args.image_max_pixels,
        "outer_supervision": args.outer_supervision,
        "latent_memory_objective": f"{args.outer_supervision}_outer_official_prompt_multimodal_answer_only_ce",
        "question_image_policy": "always_attach_when_present",
        "raw_evidence_image_oom_policy": (
            "start_with_all_selected_images_then_drop_lowest_match_score_until_success;captions_preserved"
            if args.outer_supervision == "episodic"
            else "not_applicable_no_episodic_evidence"
        ),
        "first_order": True,
        "tgu": False,
    }
    (run_root / "config.json").write_text(
        json.dumps(config_snapshot, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    metrics = []
    started = time.perf_counter()
    for outer_step in range(1, args.outer_steps + 1):
        step_started = time.perf_counter()
        theta_init = snapshot(named_params)
        outer_optimizer.zero_grad(set_to_none=True)

        inner_loss = inner_update(
            model,
            chunks,
            tokenizer,
            input_device,
            named_params,
            args,
            torch,
        )
        for _, parameter in named_params:
            parameter.grad = None

        qa_loss = outer_backward(
            model,
            processor,
            records,
            system_prompt,
            args.benchmark_prompt_dir,
            input_device,
            args,
            torch,
        )
        fast_grad_norm = gradient_norm(named_params)
        fast_grads = {
            name: (
                parameter.grad.detach().clone()
                if parameter.grad is not None
                else torch.zeros_like(parameter.detach())
            )
            for name, parameter in named_params
        }

        # First-order meta-update: apply the adapted model's QA gradients to
        # the initialization, without differentiating through the inner steps.
        restore(named_params, theta_init)
        outer_optimizer.zero_grad(set_to_none=True)
        for name, parameter in named_params:
            parameter.grad = fast_grads[name].to(
                device=parameter.device,
                dtype=parameter.dtype,
            )
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                [parameter for _, parameter in named_params],
                args.max_grad_norm,
            )
        outer_optimizer.step()
        for _, parameter in named_params:
            parameter.grad = None

        record = {
            "outer_step": outer_step,
            "inner_context_loss": inner_loss,
            "train_outer_qa_loss": qa_loss,
            "fast_grad_l2": fast_grad_norm,
            "lora_l2": parameter_norm(named_params),
            "step_seconds": round(time.perf_counter() - step_started, 3),
        }
        metrics.append(record)
        write_jsonl(metrics_path, metrics)
        print(
            f"outer step {outer_step}/{args.outer_steps}: "
            f"inner_loss={inner_loss:.6f}, outer_qa_loss={qa_loss:.6f}, "
            f"fast_grad_l2={fast_grad_norm:.6f}, "
            f"lora_l2={record['lora_l2']:.6f}, seconds={record['step_seconds']:.1f}"
        )
        if args.save_every > 0 and outer_step % args.save_every == 0:
            checkpoint_dir = checkpoints_dir / f"step_{outer_step:04d}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(checkpoint_dir)
            processor.save_pretrained(checkpoint_dir)
            print(f"saved intermediate meta-init: {checkpoint_dir}")

    init_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(init_dir)
    processor.save_pretrained(init_dir)
    print(f"saved final meta-init: {init_dir}")
    print(f"metrics: {metrics_path}")
    print(f"elapsed seconds: {time.perf_counter() - started:.2f}")
    print("LATENT_MEMORY_TRAIN_OK")


if __name__ == "__main__":
    main()
