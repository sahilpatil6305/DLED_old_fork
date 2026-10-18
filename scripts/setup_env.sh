#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3.10}"
VENV="${DLED_VENV:-$ROOT/.venv}"
if [[ "$(uname -s)" == "Darwin" ]]; then
  DEFAULT_TORCH_INDEX="https://download.pytorch.org/whl/cpu"
else
  DEFAULT_TORCH_INDEX="https://download.pytorch.org/whl/cu121"
fi
TORCH_INDEX="${PYTORCH_INDEX_URL:-$DEFAULT_TORCH_INDEX}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python 3.10 is required. Install it and rerun, or set PYTHON_BIN to a Python 3.10 executable." >&2
  exit 2
fi
PY_VERSION="$($PYTHON_BIN -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [[ "$PY_VERSION" != "3.10" ]]; then
  echo "Expected Python 3.10, found $PY_VERSION ($PYTHON_BIN)." >&2
  exit 2
fi
"$PYTHON_BIN" -m venv "$VENV"
"$VENV/bin/python" -m pip install --upgrade 'pip==24.3.1'
"$VENV/bin/python" -m pip install 'torch==2.2.2' 'torchvision==0.17.2' --index-url "$TORCH_INDEX"
"$VENV/bin/python" -m pip install -r "$ROOT/requirements.txt"
"$VENV/bin/python" - <<'PY'
import torch, torchvision
print(f"Ready: torch={torch.__version__}, torchvision={torchvision.__version__}, CUDA available={torch.cuda.is_available()}")
PY
