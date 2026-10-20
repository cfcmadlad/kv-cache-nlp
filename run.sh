#!/usr/bin/env bash
cd "$(dirname "$0")" || exit 1
NEEDED="import torch, numpy, tokenizers, tqdm"

find_python() {
    for candidate in python3 python; do
        if command -v "$candidate" >/dev/null 2>&1 &&
           "$candidate" -c "import sys; assert sys.version_info >= (3, 10)" 2>/dev/null; then
            PY="$candidate"
            return 0
        fi
    done
    return 1
}

if ! find_python; then
    echo "Python 3.10 or newer was not found. Installing it..."
    if [ "$(uname)" = "Darwin" ] && command -v brew >/dev/null 2>&1; then
        brew install python@3.12
    elif command -v apt-get >/dev/null 2>&1; then
        sudo apt-get update && sudo apt-get install -y python3 python3-venv python3-pip
    else
        echo "Could not install Python automatically. Install Python 3.10 or newer, then run this again."
        exit 1
    fi
    find_python || { echo "Python 3.10 or newer is still not available (your system python3 may be older)."
                     echo "Install Python 3.10 or newer, open a new terminal and run this again."; exit 1; }
fi
echo "Using Python: $(command -v "$PY")"

if "$PY" -c "$NEEDED" 2>/dev/null; then
    VPY="$PY"
else
    VENV="${XDG_CACHE_HOME:-$HOME/.cache}/kv-cache-nlp/venv"
    venv_python() { if [ -x "$VENV/bin/python" ]; then echo "$VENV/bin/python"; else echo "$VENV/Scripts/python.exe"; fi; }
    if [ ! -x "$VENV/bin/python" ] && [ ! -f "$VENV/Scripts/python.exe" ]; then
        echo "Creating a Python environment in $VENV (one time only)..."
        mkdir -p "$(dirname "$VENV")"
        if ! "$PY" -m venv "$VENV"; then
            rm -rf "$VENV"
            echo "Could not create the Python environment. On Debian or Ubuntu run:"
            echo "  sudo apt-get install python3-venv"
            exit 1
        fi
    fi
    VPY=$(venv_python)
    if ! "$VPY" -c "$NEEDED" 2>/dev/null; then
        if ! "$VPY" -c "import torch" 2>/dev/null; then
            if [ "$(uname)" = "Darwin" ]; then
                echo "Installing PyTorch (one time only)..."
                "$VPY" -m pip install torch || exit 1
            else
                index=https://download.pytorch.org/whl/cpu
                if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
                    index=https://download.pytorch.org/whl/cu126
                fi
                echo "Installing PyTorch from $index (one time only; the GPU build is about 2.5 GB)..."
                "$VPY" -m pip install torch --index-url "$index" || exit 1
            fi
        fi
        echo "Installing the other requirements..."
        "$VPY" -m pip install -r requirements.txt || exit 1
    fi
fi

exec "$VPY" scripts/demo.py
