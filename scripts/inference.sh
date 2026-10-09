#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/paths.sh"

MODE=${MODE:-adaptive}
RETRIEVAL=${RETRIEVAL:-online}
args=(
    --memory-mode "$MODE"
    --model-path "$MODEL_PATH"
    --output-dir "$OUTPUT_DIR"
    --benchmark-prompt-dir "$PROMPT_DIR"
    --dtype bf16
    --max-new-tokens 256
)

if [[ $MODE != episodic ]]; then
    args+=(--adapter-path "$ADAPTER_PATH")
fi
if [[ $MODE == latent ]]; then
    if [[ $RETRIEVAL != online ]]; then
        echo "latent does not use retrieval; omit RETRIEVAL=fixed" >&2
        exit 2
    fi
    args+=(--data-name "$SCENE" --qa-split "$DATA_ROOT/$SCENE/splits/intra_scene_qa/test.jsonl")
elif [[ $RETRIEVAL == fixed ]]; then
    args+=(--retrieval-mode fixed --episodic-records "$DATA_ROOT/$SCENE/evidence_test.jsonl")
else
    args+=(
        --retrieval-mode "$RETRIEVAL"
        --data-name "$SCENE"
        --qa-split "$DATA_ROOT/$SCENE/splits/intra_scene_qa/test.jsonl"
        --memory-store "$MEMORY_STORE"
    )
fi
if [[ -n ${ROUTER_CONFIG:-} ]]; then
    args+=(--router-config "$ROUTER_CONFIG")
fi

exec "$PYTHON" inference.py "${args[@]}" "$@"
