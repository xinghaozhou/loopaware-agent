#!/usr/bin/env bash
set -euo pipefail

cd /root/loopaware-agent
export UV_CACHE_DIR=/tmp/loopaware-uv-cache
export PYTHONPATH=src:.

run_dir=results/program_scheduling/bfcl64_seed20261026
shared_turn_cache=results/program_scheduling/bfcl16_seed20261026/turns
mkdir -p "$run_dir"

uv run python -m experiments.program_scheduling.generate_traces \
  --output "$run_dir/trace.json" \
  --turn-cache "$shared_turn_cache" \
  --programs 64 \
  --seed 20261026 \
  --max-new-tokens 128 \
  --resume

for program_count in 32 64; do
  output_dir="$run_dir/programs${program_count}"
  mkdir -p "$output_dir"
  uv run python -m experiments.program_scheduling.run_experiment \
    --trace "$run_dir/trace.json" \
    --program-limit "$program_count" \
    --output "$output_dir/results.json"
  uv run python -m experiments.program_scheduling.plot_results \
    --trace "$run_dir/trace.json" \
    --results "$output_dir/results.json" \
    --output-dir "$output_dir/plots"
done
