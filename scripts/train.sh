#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/paths.sh"

exec "$PYTHON" train.py \
    --model-path "$MODEL_PATH" \
    --data-name "$SCENE" \
    --train-data "$DATA_ROOT/$SCENE/splits/intra_scene_qa/train.jsonl" \
    --episodic-records "$DATA_ROOT/$SCENE/evidence_train.jsonl" \
    --output-root "$RUN_ROOT" \
    --run-name "$RUN_NAME" \
    --benchmark-prompt-dir "$PROMPT_DIR" \
    --dtype bf16 \
    --lora-rank 256 \
    --lora-alpha 256 \
    --last-n-layers 4 \
    --target-modules q_proj k_proj v_proj o_proj \
    --chunk-size 256 \
    --micro-batch-chunks 4 \
    --inner-steps 4 \
    --inner-lr 1e-4 \
    --outer-steps 5 \
    --outer-lr 5e-5 \
    --weight-decay 0.01 \
    --outer-supervision episodic \
    "$@"
