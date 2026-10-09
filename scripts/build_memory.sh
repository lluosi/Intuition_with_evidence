#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/paths.sh"

# 默认范围用于检查环境，正式实验按场景设置这三个限额。
exec "$PYTHON" build_memory.py \
    --data-name "$SCENE" \
    --model-path "$MODEL_PATH" \
    --dtype bf16 \
    --max-sessions "${MAX_SESSIONS:-2}" \
    --max-rounds "${MAX_ROUNDS:-4}" \
    --max-questions "${MAX_QUESTIONS:-2}" \
    "$@"
