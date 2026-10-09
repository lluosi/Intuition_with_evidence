"""Convert Mem-Gallery scenes and split their annotated QA pairs."""

import argparse
import json
from pathlib import Path

from data.mem_gallery import (
    DEFAULT_SPLIT_SEED,
    TASK_SCHEMA_VERSION,
    build_task,
    select_source_files,
    split_qas_by_point,
    write_scene_outputs,
)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Mem-Gallery dialogue files into converted JSONL tasks."
    )
    parser.add_argument(
        "--dialog-dir",
        type=Path,
        required=True,
        help="Official Mem-Gallery dialogue directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output root. Each scene is written under <output-dir>/<scene_name>/...",
    )
    parser.add_argument("--scene", type=str, default=None)
    parser.add_argument("--train-ratio", type=float, default=0.4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dialog_dir = args.dialog_dir.resolve()
    output_dir = args.output_dir.resolve()
    source_files = select_source_files(dialog_dir, args.scene)

    tasks = [build_task(path) for path in source_files]
    manifest_records = []
    for task in tasks:
        train_task, test_task, summary = split_qas_by_point(
            task,
            train_ratio=args.train_ratio,
            seed=DEFAULT_SPLIT_SEED,
        )
        manifest_records.append(
            write_scene_outputs(
                task,
                train_task,
                test_task,
                summary,
                output_dir,
                dialog_dir,
                args.train_ratio,
                DEFAULT_SPLIT_SEED,
            )
        )

    manifest = {
        "schema_version": TASK_SCHEMA_VERSION,
        "source_dialog_dir": str(dialog_dir),
        "num_tasks": len(tasks),
        "train_ratio": args.train_ratio,
        "seed": DEFAULT_SPLIT_SEED,
        "layout": "per_scene",
        "scenes": manifest_records,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Converted {len(tasks)} task(s) from {dialog_dir}")
    print(f"Manifest: {manifest_path}")
    for record in manifest_records:
        print(f"Scene:    {record['scene_dir']}")
        print(f"Canonical:{record['canonical']}")
        print(f"Train:    {record['train']}")
        print(f"Test:     {record['test']}")
    for item in (record["task_summary"] for record in manifest_records):
        print(
            f"{item['task_id']}: total={item['total_qas']}, "
            f"train={item['train_qas']}, test={item['test_qas']}"
        )
        print(f"  by point: {item['by_point']}")


if __name__ == "__main__":
    main()
