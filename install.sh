#!/usr/bin/env bash
# Run from a clone on Linux with Python 3.12 and the GPU driver installed.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
.venv/bin/python check_installation.py
printf '\nNext: source .venv/bin/activate\nThen: python check_installation.py --run-example\n'
