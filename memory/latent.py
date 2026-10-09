"""LoRA context adaptation and parameter snapshots for first-order training."""

from __future__ import annotations

import json
import math
import os
import random


def seed_everything(torch, seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def chunk_ids(token_ids, chunk_size):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    return [
        list(token_ids[start : start + chunk_size])
        for start in range(0, len(token_ids), chunk_size)
    ]


def context_batch(chunks, pad_token_id, device):
    import torch

    if not chunks:
        raise ValueError("Cannot create a context batch from no chunks")
    width = max(len(chunk) for chunk in chunks)
    input_ids = torch.full((len(chunks), width), pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((len(chunks), width), dtype=torch.long, device=device)
    for row, chunk in enumerate(chunks):
        length = len(chunk)
        input_ids[row, :length] = torch.tensor(chunk, dtype=torch.long, device=device)
        attention_mask[row, :length] = 1
    labels = input_ids.masked_fill(attention_mask == 0, -100)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def lora_parameters(model):
    return [
        (name, param)
        for name, param in model.named_parameters()
        if param.requires_grad and "lora" in name.lower()
    ]


def snapshot(named_params):
    return {name: param.detach().clone() for name, param in named_params}


def restore(named_params, state):
    for name, param in named_params:
        param.data.copy_(state[name].to(device=param.device, dtype=param.dtype))


def parameter_norm(named_params):
    total = sum(float(param.detach().float().pow(2).sum().cpu()) for _, param in named_params)
    return math.sqrt(total)


def gradient_norm(named_params):
    total = sum(
        float(param.grad.detach().float().pow(2).sum().cpu())
        for _, param in named_params
        if param.grad is not None
    )
    return math.sqrt(total)


def inner_update(model, chunks, tokenizer, input_device, named_params, args, torch):
    if not chunks:
        raise ValueError("The training context produced no token chunks")
    model.train()
    lora_params = [param for _, param in named_params]
    optimizer = torch.optim.AdamW(
        lora_params,
        lr=args.inner_lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    num_micro_batches = math.ceil(len(chunks) / args.micro_batch_chunks)
    last_mean_loss = 0.0
    for _ in range(args.inner_steps):
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        total_tokens = 0
        for start in range(0, len(chunks), args.micro_batch_chunks):
            batch = context_batch(
                chunks[start : start + args.micro_batch_chunks], tokenizer.pad_token_id, input_device
            )
            outputs = model(**batch, use_cache=False)
            labels = batch["labels"]
            prediction_tokens = labels[:, 1:].ne(-100).sum().item()
            # Keep equal micro-batch weighting for the update; token weighting
            # below is only for reporting the context loss.
            loss = outputs.loss / num_micro_batches
            loss.backward()
            total_loss += outputs.loss.detach().float().item() * prediction_tokens
            total_tokens += prediction_tokens
            del outputs, loss, labels, batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(lora_params, args.max_grad_norm)
        optimizer.step()
        last_mean_loss = total_loss / max(total_tokens, 1)
    return last_mean_loss


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def print_gpu_memory(torch):
    if not torch.cuda.is_available():
        return
    for index in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(index) / 1024**3
        reserved = torch.cuda.memory_reserved(index) / 1024**3
        print(f"GPU {index}: allocated={allocated:.2f} GiB, reserved={reserved:.2f} GiB")
