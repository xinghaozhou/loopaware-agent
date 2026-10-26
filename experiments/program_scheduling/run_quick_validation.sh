#!/usr/bin/env bash
set -euo pipefail

cd /root/loopaware-agent
export UV_CACHE_DIR=/tmp/loopaware-uv-cache
export PYTHONPATH=src:.

run_dir=results/program_scheduling/bfcl16_seed20261026
mkdir -p "$run_dir"

uv run python -m experiments.program_scheduling.generate_traces \
  --output "$run_dir/trace.json" \
  --programs 16 \
  --seed 20261026 \
  --max-new-tokens 128 \
  --resume

uv run python -m experiments.program_scheduling.run_experiment \
  --trace "$run_dir/trace.json" \
  --output "$run_dir/results.json"
