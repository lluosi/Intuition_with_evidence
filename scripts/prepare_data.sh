#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/paths.sh"

DIALOG_DIR=${DIALOG_DIR:-$("$PYTHON" -c 'from config import load_settings; print(load_settings().dialog_dir)')}

exec "$PYTHON" prepare_data.py \
    --dialog-dir "$DIALOG_DIR" \
    --output-dir "$DATA_ROOT" \
    --scene "$SCENE" \
    --train-ratio 0.4 \
    "$@"
