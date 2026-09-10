#!/usr/bin/env bash
# sa3-ui.command — double-click launcher for the Stable Audio 3 MLX UI (macOS).
#
# It cd's into your stable-audio-3 MLX folder, checks the local .venv the
# ./sa3 wrapper uses, opens the browser, and runs sa3_mlx_ui.py.
#
# If your stable-audio-3 lives somewhere other than ~/stable-audio-3, edit
# MLX_DIR below.

set -euo pipefail

MLX_DIR="$HOME/stable-audio-3/optimized/mlx"

cd "$MLX_DIR" 2>/dev/null || {
  echo "Could not find: $MLX_DIR"
  echo "Edit MLX_DIR in this launcher to point at your stable-audio-3/optimized/mlx folder."
  read -r -p "Press return to close."
  exit 1
}

PY="$MLX_DIR/.venv/bin/python"
if [ ! -x "$PY" ]; then
  echo "No virtual environment at $PY"
  echo "Run ./install.sh inside $MLX_DIR first, then: uv pip install gradio"
  read -r -p "Press return to close."
  exit 1
fi

if ! "$PY" -c "import gradio" 2>/dev/null; then
  echo "Gradio isn't installed in the .venv. Installing it now..."
  uv pip install gradio || "$PY" -m pip install gradio
fi

# Give the server a moment to start, then open the browser.
( sleep 3; open "http://127.0.0.1:7860" ) &

echo "Starting the Stable Audio 3 MLX UI...  (close this window or Ctrl-C to stop)"
exec "$PY" sa3_mlx_ui.py
