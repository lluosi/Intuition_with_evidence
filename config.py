from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
VENDORED_PROMPT_DIR = PACKAGE_ROOT / "mem_gallery" / "prompt"


def _env_value(name: str) -> str | None:
    value = os.environ.get(name, "").strip()
    return None if not value or (value.startswith("<") and value.endswith(">")) else value


def _load_env_file(path: Path) -> None:
    """Load simple KEY=VALUE lines without overriding existing environment variables."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _project_root() -> Path:
    configured = _env_value("MEMORY_PROJECT_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    for candidate in (Path.cwd(), PACKAGE_ROOT, PACKAGE_ROOT.parent):
        candidate = candidate.resolve()
        if (candidate / "Mem-Gallery").exists() or (candidate / "benchmark").exists():
            return candidate
    return PACKAGE_ROOT.resolve()


def _benchmark_root(project_root: Path) -> Path:
    configured = _env_value("MEM_GALLERY_ROOT")
    if configured:
        return Path(configured).expanduser() / "benchmark"
    for candidate in (project_root / "Mem-Gallery", project_root):
        benchmark = candidate / "benchmark"
        if (benchmark / "data" / "dialog").exists():
            return benchmark
    return project_root / "benchmark"


def safe_model_name(name: str) -> str:
    return name.replace("/", "_").replace("\\", "_").replace(":", "_")


@dataclass
class Settings:
    project_root: Path
    benchmark_root: Path
    result_dir: Path
    memory_api_key: str
    memory_api_base: str
    memory_model: str
    embedding_backend: str
    embedding_api_key: str
    embedding_api_base: str
    embedding_model: str
    answer_model: str
    answer_api_key: str
    answer_api_base: str
    timeout: float = 300.0
    max_retries: int = 3
    retry_delay: float = 20.0
    json_retries: int = 2
    semantic_merge_threshold: float = 0.72

    @property
    def dialog_dir(self) -> Path:
        return self.benchmark_root / "data" / "dialog"

    @property
    def image_dir(self) -> Path:
        return self.benchmark_root / "data" / "image"

    @property
    def prompt_dir(self) -> Path:
        configured = _env_value("MEM_GALLERY_PROMPT_DIR")
        return Path(configured).expanduser() if configured else VENDORED_PROMPT_DIR



def load_settings(
    *,
    answer_model: str | None = None,
    answer_api_base: str | None = None,
    answer_api_key: str | None = None,
    embedding_model: str | None = None,
) -> Settings:
    env_file = Path(os.environ.get("MEMORY_ENV_FILE", PACKAGE_ROOT / ".env")).expanduser()
    _load_env_file(env_file)
    project_root = _project_root()
    benchmark_root = _benchmark_root(project_root)
    memory_api_key = os.environ.get("MEMORY_API_KEY", "")
    return Settings(
        project_root=project_root,
        benchmark_root=benchmark_root,
        result_dir=Path(_env_value("MEMORY_RESULT_DIR") or (PACKAGE_ROOT / "results")).expanduser(),
        memory_api_key=memory_api_key,
        memory_api_base=os.environ.get("MEMORY_API_BASE", ""),
        memory_model=os.environ.get("MEMORY_MODEL", "<MEMORY_MODEL>"),
        embedding_backend=os.environ.get("EMBEDDING_BACKEND", "local").lower(),
        embedding_api_key=os.environ.get("EMBEDDING_API_KEY", memory_api_key),
        embedding_api_base=os.environ.get("EMBEDDING_API_BASE", ""),
        embedding_model=embedding_model or os.environ.get("EMBEDDING_MODEL", "<EMBEDDING_MODEL>"),
        answer_model=answer_model or os.environ.get("ANSWER_MODEL", ""),
        answer_api_key=answer_api_key if answer_api_key is not None else os.environ.get("ANSWER_API_KEY", ""),
        answer_api_base=answer_api_base if answer_api_base is not None else os.environ.get("ANSWER_API_BASE", ""),
        timeout=float(os.environ.get("MEMORY_TIMEOUT", "300")),
        max_retries=int(os.environ.get("MEMORY_MAX_RETRIES", "3")),
        retry_delay=float(os.environ.get("MEMORY_RETRY_DELAY", "20")),
        json_retries=int(os.environ.get("MEMORY_JSON_RETRIES", "2")),
    )
