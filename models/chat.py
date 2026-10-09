from __future__ import annotations

import json
import time
from dataclasses import dataclass


@dataclass
class ClientConfig:
    api_key: str
    api_base: str
    model: str
    timeout: float = 300.0
    max_retries: int = 3
    retry_delay: float = 20.0
    json_retries: int = 2


class ChatModel:
    def __init__(self, config: ClientConfig):
        from openai import OpenAI

        if not config.api_base:
            raise ValueError("An API base is required for the model client.")
        self.config = config
        self.chat_client = OpenAI(api_key=config.api_key, base_url=config.api_base)

    def chat(self, messages, *, model=None, json_mode=False, max_tokens=None) -> str:
        params = {
            "model": model or self.config.model,
            "messages": messages,
            "temperature": 0.0,
            "timeout": self.config.timeout,
        }
        if json_mode:
            params["response_format"] = {"type": "json_object"}
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        response = self._with_retry(self.chat_client.chat.completions.create, params)
        return response.choices[0].message.content or ""

    def chat_json(self, messages, *, model=None):
        last_error = None
        current = list(messages)
        for _ in range(max(1, self.config.json_retries)):
            raw = self.chat(current, model=model, json_mode=True)
            try:
                return _parse_json(raw)
            except json.JSONDecodeError as error:
                last_error = error
                current = [
                    *messages,
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": "Return one valid JSON object only."},
                ]
        raise last_error or ValueError("The model did not return JSON.")

    def _with_retry(self, operation, params):
        last_error = None
        for attempt in range(max(1, self.config.max_retries)):
            try:
                return operation(**params)
            except Exception as error:
                last_error = error
                if "context length" in str(error).lower():
                    raise
                if attempt + 1 < self.config.max_retries:
                    time.sleep(self.config.retry_delay * (attempt + 1))
        raise last_error or RuntimeError("Request failed.")


def _parse_json(raw: str):
    text = raw.strip()
    if text.startswith(chr(96) * 3):
        text = text.strip(chr(96)).strip()
        if text.startswith("json"):
            text = text[4:].lstrip()
    return json.loads(text)


def memory_client(settings) -> ChatModel:
    return ChatModel(ClientConfig(
        api_key=settings.memory_api_key,
        api_base=settings.memory_api_base,
        model=settings.memory_model,
        timeout=settings.timeout,
        max_retries=settings.max_retries,
        retry_delay=settings.retry_delay,
        json_retries=settings.json_retries,
    ))


def answer_client(settings) -> ChatModel:
    return ChatModel(ClientConfig(
        api_key=settings.answer_api_key,
        api_base=settings.answer_api_base,
        model=settings.answer_model,
        timeout=settings.timeout,
        max_retries=settings.max_retries,
        retry_delay=settings.retry_delay,
        json_retries=settings.json_retries,
    ))
