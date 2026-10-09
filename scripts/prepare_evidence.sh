#!/usr/bin/env bash
set -euo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/paths.sh"

SPLIT=${SPLIT:-train}
exec "$PYTHON" prepare_evidence.py \
    --data-name "$SCENE" \
    --memory-store "$MEMORY_STORE" \
    --qa-split "$DATA_ROOT/$SCENE/splits/intra_scene_qa/$SPLIT.jsonl" \
    --output "$DATA_ROOT/$SCENE/evidence_$SPLIT.jsonl" \
    "$@"
