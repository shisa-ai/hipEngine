#!/usr/bin/env bash
# Create the pinned torch-oracle venv for VibeVoice-TTS.
#
# The oracle must run the *reference* implementation (the community fork) under
# the transformers version it was written against. The host's transformers
# (5.15.0) cannot import the fork, so the oracle needs its own interpreter
# environment. This script builds it reproducibly.
#
# The venv is created with --system-site-packages so the host's ROCm torch
# (2.13.0+rocm10.0.0) is reused instead of reinstalling a mismatched wheel.
# Everything else is pinned; see scripts/vibevoice_tts_oracle_requirements.txt.
#
# Usage:
#     scripts/setup_vibevoice_tts_oracle_env.sh [venv-path]
#
# Default venv path: /home/lhl/venvs/vibevoice-tts-oracle
#
# After it finishes, verify the environment with:
#     <venv>/bin/python -c "import torch, transformers; \
#         print(transformers.__version__, torch.__version__, torch.cuda.is_available())"

set -euo pipefail

VENV="${1:-/home/lhl/venvs/vibevoice-tts-oracle}"
REQS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/vibevoice_tts_oracle_requirements.txt"
BASE_PYTHON="${BASE_PYTHON:-python3}"

if [[ ! -f "$REQS" ]]; then
    echo "error: requirements file not found: $REQS" >&2
    exit 1
fi

echo "==> creating venv at $VENV (system site packages: reuses host ROCm torch)"
rm -rf "$VENV"
"$BASE_PYTHON" -m venv --system-site-packages "$VENV"

echo "==> installing pinned dependencies"
"$VENV/bin/pip" install --quiet --upgrade pip
"$VENV/bin/pip" install --quiet -r "$REQS"

echo "==> verifying"
"$VENV/bin/python" - <<'PY'
import importlib.metadata as md
import torch, transformers, tokenizers, huggingface_hub

def pin(name):
    try:
        return md.version(name)
    except Exception:
        return "<not installed>"

print(f"  transformers    {transformers.__version__}")
print(f"  tokenizers      {tokenizers.__version__}")
print(f"  huggingface_hub {huggingface_hub.__version__}")
print(f"  torch           {torch.__version__}")
print(f"  HIP available   {torch.cuda.is_available()}")

# The two constraints that silently break the oracle if they drift.
assert transformers.__version__ == "4.51.3", (
    f"transformers must be 4.51.3, got {transformers.__version__}; "
    "later versions register vibevoice_acoustic_tokenizer and block the fork import"
)
assert md.version("huggingface-hub").split(".")[0] == "0", (
    "huggingface-hub must stay <1.0 for transformers 4.51.3"
)
print("  environment OK")
PY

echo
echo "==> done. Use the oracle with:"
echo "    PYTHONPATH=/home/lhl/VibeVoice-community $VENV/bin/python scripts/vibevoice_tts_oracle_torch.py"
