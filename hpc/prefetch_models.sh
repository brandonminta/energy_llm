#!/bin/bash
# Run on a LOGIN node (internet access). Downloads the exact revision pinned
# in each config into $HF_HOME (on scratch), then checks that it loads with
# HF_HUB_OFFLINE=1 as the compute nodes will. Llama is gated: export
# HF_TOKEN beforehand (never write it to a file).
#
#   hpc/prefetch_models.sh configs/qwen25_3b.yaml configs/llama32_3b.yaml
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source hpc/env.sh
activate_env
for CONFIG in "$@"; do
    python - "$CONFIG" <<'PY'
import sys
from huggingface_hub import snapshot_download
from energy_llm.config import load_config
cfg = load_config(sys.argv[1])
path = snapshot_download(cfg.model.id, revision=cfg.model.revision,
                         allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken"])
print(f"{cfg.model.id}@{cfg.model.revision} -> {path}")
PY
    HF_HUB_OFFLINE=1 python - "$CONFIG" <<'PY'
import sys
from transformers import AutoConfig, AutoTokenizer
from energy_llm.config import load_config
cfg = load_config(sys.argv[1])
AutoConfig.from_pretrained(cfg.model.id, revision=cfg.model.revision)
AutoTokenizer.from_pretrained(cfg.model.id, revision=cfg.model.revision)
print(f"offline load OK: {cfg.model.id}")
PY
done
