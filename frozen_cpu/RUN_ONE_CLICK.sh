#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON=${PYTHON:-python3}
[ -x .venv/bin/python ] || "$PYTHON" -m venv .venv
PY=.venv/bin/python
if [ ! -f .deps_v27_ok ]; then "$PY" -m pip install -U pip; "$PY" -m pip install -r requirements.txt; touch .deps_v27_ok; fi
"$PY" VERIFY_FREEZE.py
"$PY" DOWNLOAD_FRESH_DATA.py
"$PY" PREPARE_FRESH_DATA.py
"$PY" v27_fresh_cpu.py
"$PY" ANALYZE_FRESH_RESULTS.py
"$PY" PACKAGE_RESULTS.py
