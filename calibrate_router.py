import argparse
import hashlib
import json
from pathlib import Path

from memory.router import fit, read_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Calibration JSONL only")
    parser.add_argument("--output", required=True, type=Path, help="New router configuration JSON")
    args = parser.parse_args()
    try:
        config = fit(read_rows(args.input))
        # Preserve provenance without embedding private paths, QA IDs or data.
        config["calibration"]["input_sha256"] = hashlib.sha256(args.input.read_bytes()).hexdigest()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(config, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Calibration failed: {exc}\n")
    print(json.dumps({"output": str(args.output), "thresholds": config["thresholds"], "calibration": config["calibration"]}))


if __name__ == "__main__":
    main()
