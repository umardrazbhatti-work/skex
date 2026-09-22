#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m skex.experiments.runner --plan "${1:-experiments/plans/00_smoke.yaml}" "${@:2}"
