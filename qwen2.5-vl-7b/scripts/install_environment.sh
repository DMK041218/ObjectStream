#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python3.11}"
VENV_DIR="${VENV_DIR:-.venv}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-1}"

"${PYTHON_BIN}" -m venv "${VENV_DIR}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

python -m pip install pip==26.2.1 setuptools==75.8.2 wheel==0.45.1
python -m pip install \
  --index-url "${TORCH_INDEX_URL}" \
  torch==2.6.0 torchvision==0.21.0
python -m pip install -r requirements/runtime.txt
python scripts/repair_decord_wheel.py
python -m pip install -e models/qwen-vl-utils
python -m pip install -e models/qwen2-5-vl
python -m pip install -e .

if [[ "${INSTALL_FLASH_ATTN}" == "1" ]]; then
  python -m pip install flash-attn==2.7.4.post1 --no-build-isolation
fi

python scripts/check_environment.py
