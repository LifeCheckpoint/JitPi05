#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "This script must run inside Ubuntu/WSL2." >&2
  exit 1
fi

sudo apt-get update
sudo apt-get install -y \
  build-essential \
  cmake \
  curl \
  ffmpeg \
  git \
  libegl1 \
  libgl1 \
  libglew-dev \
  libglfw3 \
  libglfw3-dev \
  libosmesa6-dev \
  patchelf \
  python3-dev

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "The Windows NVIDIA driver is not exposed to WSL2." >&2
  exit 1
fi
nvidia-smi

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

UV_CONCURRENT_DOWNLOADS="${UV_CONCURRENT_DOWNLOADS:-1}" \
UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}" \
uv sync --frozen
uv run jitpi05-doctor

cat <<'EOF'
JitPi05 is ready.
For headless simulation use:
  MUJOCO_GL=egl uv run jitpi05-eval-sim
EOF
