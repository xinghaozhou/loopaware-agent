#!/usr/bin/env bash
set -euo pipefail

cd /root/loopaware-agent
export HF_HOME=/tmp/loopaware-hf
export UV_CACHE_DIR=/tmp/loopaware-uv-cache
export PYTORCH_ALLOC_CONF=expandable_segments:True

base=results/program_scheduling/bfcl64_seed20261026
trace="$base/trace.json"

for policy in GPU_PLAS UT_LAS ORACLE_TOKEN_SUF ORACLE_UT_SUF ORACLE_RUNTIME_SJF; do
  output="$base/${policy,,}_b4_full"
  mkdir -p "$output"
  uv run python -m experiments.program_scheduling.run_full_compute_batch \
    --source-trace "$trace" \
    --output-dir "$output" \
    --batch-size 4 \
    --policy "$policy" \
    --programs 64 \
    --seed 20261026 \
    --max-new-tokens 128 \
    --reuse-measurements-from \
      "$base/fcfs_b4_full" \
      "$base/gpu_plas_b4_full" \
      "$base/ut_las_b4_full" \
      "$base/oracle_token_suf_b4_full" \
      "$base/oracle_ut_suf_b4_full" \
    --resume 2>&1 | tee -a "$output/run.log"
done
