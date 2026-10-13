#!/usr/bin/env bash
# Server-specific launcher for the verified CPU 16-31 configuration.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_dir"
if [ "$#" -eq 0 ]; then
  printf '%s\n' 'Usage: ./run_mindllm_server.sh python main.py [arguments]' >&2
  exit 2
fi
export VIRTUAL_ENV="$project_dir/.venv-repaired"
export PYTHONFAULTHANDLER=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
exec taskset -c 16-31 uv run --active --no-sync "$@"
