from __future__ import annotations

import argparse
import base64
import math
from io import BytesIO
from pathlib import Path
from typing import Any

from memory.prompts import ANSWER_SYSTEM, build_answer_content, drop_lowest_match_image


def load_qwen25vl_config(model_path: Path, normalize_rope: bool = False):
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=True,
    )
    if normalize_rope:
        rope_scaling = getattr(config, "rope_scaling", None)
        if isinstance(rope_scaling, dict) and rope_scaling.get("rope_type") == "mrope":
            config.rope_scaling = dict(rope_scaling)
            config.rope_scaling["rope_type"] = "default"
            print("Normalized runtime rope_scaling rope_type from 'mrope' to 'default'; kept mrope_section.")
    return config


def resolve_dtype(torch, dtype_name: str):
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype_name]


def is_cuda_oom_error(error: Exception) -> bool:
    return error.__class__.__name__ == "OutOfMemoryError" or "CUDA out of memory" in str(error)


def normalize_multimodal_content(
    content: list[dict[str, Any]],
    image_min_pixels: int | None,
    image_max_pixels: int | None,
) -> list[dict[str, Any]]:
    """Convert API image parts to processor input, preserving explicit pixel limits."""

    normalized = []
    for original in content:
        item = dict(original)
        if item.get("type") == "image_url":
            image_url = item.get("image_url", {})
            item = {"type": "image", "image": image_url.get("url", "")}
        if item.get("type") == "image":
            if image_min_pixels is not None:
                item.setdefault("min_pixels", image_min_pixels)
            if image_max_pixels is not None:
                item.setdefault("max_pixels", image_max_pixels)
        normalized.append(item)
    return normalized


def _resize_qwen_image(image: Any, item: dict[str, Any]) -> Any:
    """Apply the Qwen2.5-VL 28-pixel patch/merge resizing rule."""

    from PIL import Image

    if image.mode == "RGBA":
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])
        image = background
    else:
        image = image.convert("RGB")
    width, height = image.size
    factor = 28
    min_pixels = int(item.get("min_pixels", 4 * factor**2))
    max_pixels = int(item.get("max_pixels", 16384 * factor**2))
    if max_pixels < min_pixels:
        raise ValueError("max_pixels must be >= min_pixels")
    if max(height, width) / min(height, width) > 200:
        raise ValueError("Image aspect ratio must be <= 200")

    resized_height = max(factor, round(height / factor) * factor)
    resized_width = max(factor, round(width / factor) * factor)
    if resized_height * resized_width > max_pixels:
        scale = math.sqrt((height * width) / max_pixels)
        resized_height = max(factor, math.floor(height / scale / factor) * factor)
        resized_width = max(factor, math.floor(width / scale / factor) * factor)
    elif resized_height * resized_width < min_pixels:
        scale = math.sqrt(min_pixels / (height * width))
        resized_height = max(factor, math.ceil(height * scale / factor) * factor)
        resized_width = max(factor, math.ceil(width * scale / factor) * factor)
    return image.resize((resized_width, resized_height))


def process_vision_info_compat(
    messages: list[dict[str, Any]],
) -> tuple[list[Any] | None, Any]:
    """Use qwen_vl_utils when installed, with an image-only local fallback."""

    try:
        from qwen_vl_utils import process_vision_info

        return process_vision_info(messages)
    except ModuleNotFoundError as error:
        if error.name != "qwen_vl_utils":
            raise

    from PIL import Image

    images = []
    for message in messages:
        content = message.get("content", [])
        if not isinstance(content, list):
            continue
        for item in content:
            if item.get("type") not in {"image", "image_url"}:
                continue
            source = item.get("image")
            if source is None:
                source = item.get("image_url", {}).get("url", "")
            if isinstance(source, Image.Image):
                image = source.copy()
            elif isinstance(source, str) and source.startswith("data:image"):
                if "base64," not in source:
                    raise ValueError("Only base64 data-image URLs are supported")
                payload = base64.b64decode(source.split("base64,", 1)[1])
                image = Image.open(BytesIO(payload))
            elif isinstance(source, str) and source.startswith("file://"):
                image = Image.open(source[7:])
            elif isinstance(source, str) and not source.startswith(("http://", "https://")):
                image = Image.open(source)
            else:
                raise ValueError(
                    "The local qwen_vl_utils fallback supports PIL, local paths, "
                    "file:// paths and base64 data-image URLs only"
                )
            images.append(_resize_qwen_image(image, item))
    return (images or None), None


def encode_answer_only_example(
    processor: Any,
    system_prompt: str,
    user_content: list[dict[str, Any]],
    answer: str,
    device: Any,
    image_min_pixels: int | None,
    image_max_pixels: int | None,
) -> dict[str, Any]:
    """Encode one multimodal QA and mask every token before the answer."""

    import torch

    normalized_content = normalize_multimodal_content(
        user_content,
        image_min_pixels,
        image_max_pixels,
    )
    prompt_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": normalized_content},
    ]
    full_messages = [
        *prompt_messages,
        {"role": "assistant", "content": answer},
    ]

    prompt_text = processor.apply_chat_template(
        prompt_messages,
        add_generation_prompt=True,
        tokenize=False,
    )
    full_text = processor.apply_chat_template(
        full_messages,
        add_generation_prompt=False,
        tokenize=False,
    )
    image_inputs, video_inputs = process_vision_info_compat(prompt_messages)
    prompt_batch = processor(
        text=[prompt_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    full_batch = processor(
        text=[full_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    prompt_ids = prompt_batch["input_ids"][0]
    full_ids = full_batch["input_ids"][0]
    prompt_length = int(prompt_ids.numel())
    if prompt_length >= int(full_ids.numel()):
        raise ValueError("The encoded answer contains no supervised tokens")
    if not torch.equal(full_ids[:prompt_length], prompt_ids):
        raise ValueError(
            "Prompt tokens are not an exact prefix of the full conversation. "
            "Refusing to construct a potentially shifted answer mask."
        )

    labels = full_batch["input_ids"].clone()
    labels[:, :prompt_length] = -100
    attention_mask = full_batch.get("attention_mask")
    if attention_mask is not None:
        labels = labels.masked_fill(attention_mask.eq(0), -100)
    if int(labels.ne(-100).sum().item()) <= 0:
        raise ValueError("No answer tokens remain after label masking")

    encoded = {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in full_batch.items()
    }
    encoded["labels"] = labels.to(device)
    return encoded


def encode_content(
    processor: Any,
    system_prompt: str,
    content: list[dict[str, Any]],
    input_device: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    normalized = normalize_multimodal_content(
        content, args.image_min_pixels, args.image_max_pixels,
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": normalized},
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    images, videos = process_vision_info_compat(messages)
    encoded = processor(text=[prompt], images=images, videos=videos, padding=True, return_tensors="pt")
    return {
        key: value.to(input_device) if hasattr(value, "to") else value
        for key, value in encoded.items()
    }


def generate_encoded_ids(
    model: Any,
    processor: Any,
    encoded: dict[str, Any],
    max_new_tokens: int,
    torch: Any,
) -> list[int]:
    with torch.no_grad():
        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )
    return generated[0, encoded["input_ids"].shape[-1]:].tolist()


class QwenAnswer:
    def __init__(
        self,
        model_path: Path,
        *,
        dtype: str = "bf16",
        image_min_pixels: int = 200704,
        image_max_pixels: int = 802816,
        max_new_tokens: int = 256,
    ) -> None:
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.torch = torch
        self.image_min_pixels = image_min_pixels
        self.image_max_pixels = image_max_pixels
        self.max_new_tokens = max_new_tokens
        self.model_name = model_path.name
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
            use_fast=True,
        )
        if self.processor.tokenizer.pad_token_id is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
        load_kwargs = {
            "config": load_qwen25vl_config(model_path),
            "torch_dtype": resolve_dtype(torch, dtype),
            "local_files_only": True,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        if torch.cuda.is_available():
            load_kwargs["device_map"] = "auto"
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            **load_kwargs,
        )
        self.model.eval()
        self.input_device = self.model.get_input_embeddings().weight.device

    def answer(
        self,
        core_memory: str,
        evidence: list[dict[str, Any]],
        question: str,
        question_image: dict[str, str] | None,
        image_ids: list[str],
        speaker_a: str = "user",
        speaker_b: str = "assistant",
        format_constraint: str | None = None,
        system_prompt: str | None = None,
    ) -> tuple[str, list[str], list[str]]:
        active_ids = list(dict.fromkeys(image_ids))
        dropped = []
        while True:
            content = build_answer_content(
                core_memory,
                evidence,
                question,
                question_image,
                active_ids,
                speaker_a,
                speaker_b,
                format_constraint,
            )
            try:
                return self._generate(system_prompt or ANSWER_SYSTEM, content), active_ids, dropped
            except Exception as error:
                if not is_cuda_oom_error(error) or not active_ids:
                    raise
                active_ids, removed = drop_lowest_match_image(active_ids, evidence)
                if removed is None:
                    raise
                dropped.append(removed)
                if self.torch.cuda.is_available():
                    self.torch.cuda.empty_cache()

    def _generate(self, system_prompt: str, content: list[dict[str, Any]]) -> str:
        encoded = encode_content(
            self.processor, system_prompt, content, self.input_device, self,
        )
        token_ids = generate_encoded_ids(
            self.model, self.processor, encoded, self.max_new_tokens, self.torch,
        )
        return self.processor.tokenizer.decode(token_ids, skip_special_tokens=True).strip()
