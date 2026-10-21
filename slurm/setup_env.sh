#!/usr/bin/env bash
set -euo pipefail
VENV="${KVC_VENV:-$HOME/venvs/kvc}"
PY="${PYTHON:-python3.11}"
cd "$(dirname "$0")/.."
[ -x "$VENV/bin/python" ] || "$PY" -m venv "$VENV"
"$VENV/bin/python" -m pip install --upgrade pip
"$VENV/bin/python" -m pip install torch --index-url https://download.pytorch.org/whl/cu128
"$VENV/bin/python" -m pip install -r requirements-dev.txt
"$VENV/bin/python" -c "import torch; print(torch.__version__, torch.version.cuda)"
