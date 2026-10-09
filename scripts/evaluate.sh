#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/paths.sh"

MODE=${MODE:-adaptive}
RETRIEVAL=${RETRIEVAL:-online}
if [[ $MODE == latent ]]; then
    if [[ $RETRIEVAL != online ]]; then
        echo "latent does not use retrieval; omit RETRIEVAL=fixed" >&2
        exit 2
    fi
    result_name=latent
    args=()
else
    result_name=${MODE}_${RETRIEVAL}
    args=(--evaluate-retrieval)
fi

exec "$PYTHON" evaluate.py \
    --results "$OUTPUT_DIR/$result_name/results.json" \
    --output "$OUTPUT_DIR/$result_name/metrics.json" \
    "${args[@]}" "$@"
