"""Score saved answers with the bundled Mem-Gallery evaluator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evaluate-retrieval", action="store_true")
    parser.add_argument("--retrieval-top-k", type=int, default=10)
    parser.add_argument("--enable-llm-judge", action="store_true")
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--judge-api-key", default=None)
    parser.add_argument("--judge-api-base", default=None)
    parser.add_argument("--judge-max-retries", type=int, default=5)
    parser.add_argument("--judge-timeout", type=int, default=60)
    args = parser.parse_args()
    args.results = args.results.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if args.enable_llm_judge:
        missing = [name for name, value in {
            "--judge-model": args.judge_model,
            "--judge-api-key": args.judge_api_key,
            "--judge-api-base": args.judge_api_base,
        }.items() if not value]
        if missing:
            raise ValueError("LLM judge requires " + ", ".join(missing))
    return args


def main() -> None:
    args = parse_args()
    if not args.results.is_file():
        raise FileNotFoundError(args.results)
    from mem_gallery.evaluation import evaluation as official

    judge_config = None
    if args.enable_llm_judge:
        judge_config = {
            "name": args.judge_model,
            "api_key": args.judge_api_key,
            "base_url": args.judge_api_base,
            "prompt_path": str(Path(official.__file__).with_name("llm_judge.txt")),
            "max_retries": args.judge_max_retries,
            "timeout": args.judge_timeout,
        }
    summary = official.main(
        str(args.results),
        enable_llm_judge=args.enable_llm_judge,
        judge_config=judge_config,
        eval_retrieval_metrics=args.evaluate_retrieval,
        eval_topk=args.retrieval_top_k,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Mem-Gallery metrics: {args.output}")


if __name__ == "__main__":
    main()
