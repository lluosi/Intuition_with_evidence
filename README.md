# Intuition with Evidence

Implementation code for paper Intuition with Evidence: Adaptive Dual Memory for Multimodal Agents

## Setup


```bash
pip install -r requirements.txt
cp .env.example .env
```

Configure `.env`:

- `MEM_GALLERY_ROOT`: Mem-Gallery directory containing `benchmark/data/dialog` and `benchmark/data/image`.
- `MEMORY_API_KEY`, `MEMORY_API_BASE`, `MEMORY_MODEL`: OpenAI-compatible API credentials, endpoint, and model for memory extraction.
- `EMBEDDING_MODEL`: Local model path, or model name for API embeddings. For API embeddings, also set the backend, endpoint, and key in `.env` as shown in `.env.example`.

## Run

Set the scene and model path. `SCENE` is the dialogue JSON filename without its extension.

```bash
export SCENE=your_scene
export MODEL_PATH=/path/to/Qwen2.5-VL
export RUN_NAME="$SCENE"

bash scripts/prepare_data.sh
bash scripts/build_memory.sh
bash scripts/prepare_evidence.sh
bash scripts/train.sh
bash scripts/inference.sh
bash scripts/evaluate.sh
```

For full experiments, set `MAX_SESSIONS`, `MAX_ROUNDS`, and `MAX_QUESTIONS` to positive integers covering the desired data range.

Inference defaults to `adaptive` mode with online retrieval. To use another mode, prefix both the inference and evaluation commands
with the same setting: `MODE=episodic`, `MODE=latent`, or `MODE=hybrid`.

Adapters are saved to `runs/$RUN_NAME/adapters/init/`; predictions and evaluation metrics are saved under `results/$RUN_NAME/`.
See `scripts/train.sh` for training parameters and `scripts/paths.sh` for shared paths.
