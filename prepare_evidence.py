"""Retrieve and freeze evidence for a converted QA split."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from memory.retrieval import (
    EPISODIC_RECORD_SCHEMA_VERSION,
    MemoryRetriever,
    load_memory_store,
    make_episodic_record,
)
from config import load_settings
from data.mem_gallery import MemGallery
from models.chat import memory_client
from models.embeddings import create_embedding_encoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-name", required=True)
    parser.add_argument("--memory-store", type=Path, required=True)
    parser.add_argument("--qa-split", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, default=None)
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--max-qa", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    for name in ("memory_store", "qa_split", "output"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.benchmark_root is not None:
        args.benchmark_root = args.benchmark_root.expanduser().resolve()
    return args


def load_task(path: Path, data_name: str) -> dict[str, Any]:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            task = json.loads(line)
            if str(task.get("task_id")) == data_name:
                return task
    raise ValueError(f"No task {data_name!r} in split: {path}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists; pass --overwrite: {args.output}")
    if not args.memory_store.is_file() or not args.qa_split.is_file():
        raise FileNotFoundError("--memory-store and --qa-split must both exist")

    settings = load_settings(embedding_model=args.embedding_model)
    if args.benchmark_root is not None:
        settings = replace(settings, benchmark_root=args.benchmark_root)
    task = load_task(args.qa_split, args.data_name)
    qas = list(task.get("qa_pairs", []))
    if args.max_qa is not None:
        qas = qas[:args.max_qa]
    if not qas:
        raise ValueError("The requested QA split is empty")

    store = load_memory_store(args.memory_store)
    reader = MemGallery(settings.dialog_dir, settings.image_dir)
    retriever = MemoryRetriever(memory_client(settings), create_embedding_encoder(settings))
    records = [
        make_episodic_record(
            data_name=args.data_name,
            qa_index=index,
            qa=qa,
            store=store,
            reader=reader,
            retriever=retriever,
        )
        for index, qa in enumerate(qas, start=1)
    ]
    write_jsonl(args.output, records)
    manifest = {
        "schema_version": EPISODIC_RECORD_SCHEMA_VERSION,
        "data_name": args.data_name,
        "memory_store": str(args.memory_store),
        "memory_store_sha256": sha256_file(args.memory_store),
        "qa_split": str(args.qa_split),
        "qa_split_sha256": sha256_file(args.qa_split),
        "records": len(records),
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Episodic evidence records: {args.output} ({len(records)})")


if __name__ == "__main__":
    main()
