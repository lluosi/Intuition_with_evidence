#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/paths.sh"

exec "$PYTHON" calibrate_router.py \
    --input "${CALIBRATION_DATA:-$DATA_ROOT/$SCENE/calibration.jsonl}" \
    --output "${ROUTER_CONFIG:-$RUN_ROOT/$RUN_NAME/router.json}" \
    "$@"
