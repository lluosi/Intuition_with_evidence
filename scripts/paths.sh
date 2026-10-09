#!/usr/bin/env bash
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

PYTHON=${PYTHON:-python}
SCENE=${SCENE:-scene}
MODEL_PATH=${MODEL_PATH:-checkpoints/Qwen2.5-VL}
RUN_NAME=${RUN_NAME:-$SCENE}
DATA_ROOT=${DATA_ROOT:-datasets}
RUN_ROOT=${RUN_ROOT:-runs}
OUTPUT_DIR=${OUTPUT_DIR:-results/$RUN_NAME}
ADAPTER_PATH=${ADAPTER_PATH:-$RUN_ROOT/$RUN_NAME/adapters/init}
PROMPT_DIR=${PROMPT_DIR:-mem_gallery/prompt}

# 复用 Python 的 .env 配置和文件命名，避免两处维护记忆路径。
if [[ -z ${MEMORY_STORE:-} ]]; then
    MEMORY_STORE=$("$PYTHON" -c '
import sys
from pathlib import Path
from config import load_settings, safe_model_name
print(load_settings().result_dir / safe_model_name(Path(sys.argv[1]).name)
      / "episodic_memory" / (sys.argv[2] + "_memory.json"))
' "$MODEL_PATH" "$SCENE")
fi
