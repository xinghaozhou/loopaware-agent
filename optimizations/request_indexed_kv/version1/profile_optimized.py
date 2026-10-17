"""Run the existing coarse profile with isolated optimized cache classes."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from optimizations.request_indexed_kv.version1.optimized_cache import make_optimized_ut_cache
from vendor.ouro_hf import modeling_ouro


# Cache creation is the only integration seam changed for this isolated copy.
modeling_ouro.make_ut_cache = make_optimized_ut_cache

from experiments import profile_adaptive_kv as baseline_profile  # noqa: E402


def main() -> None:
    baseline_profile.main()
    args = baseline_profile.parse_args()
    result_path = args.output_dir / "results.json"
    report_path = args.output_dir / "report.md"
    if not result_path.exists() or not report_path.exists():
        return
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["experiment"] = "request_indexed_in_place_kv_profile"
    result["optimization"] = {
        "original": "geometric in-place full-per-depth KV",
        "pred90_suf": "stable request-indexed in-place Last-Available KV",
    }
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    report = report_path.read_text(encoding="utf-8").replace(
        "# Original Ouro vs Adaptive Pred90: KV Runtime Profile",
        "# Optimized Original Ouro vs Optimized Adaptive Pred90: KV Runtime Profile",
        1,
    )
    report_path.write_text(report, encoding="utf-8")
    control_path = Path(
        os.environ.get(
            "KV_CONTROL_RESULTS",
            PROJECT_ROOT
            / "results"
            / "kv_profile_resumable_20261017"
            / "results.json",
        )
    )
    if control_path.exists():
        control = json.loads(control_path.read_text(encoding="utf-8"))

        def sync_calls(system: dict) -> int:
            for row in system["profile"]["top_cuda_api_calls"]:
                if row["name"] == "cudaStreamSynchronize":
                    return int(row["count"])
            return 0

        lines = [
            "# Request-Indexed KV Profile Comparison",
            "",
            f"Control: `{control_path}`",
            "",
            "| System | Metric | Control | Optimized | Change |",
            "|---|---|---:|---:|---:|",
        ]
        for system_name, label in (
            ("original", "Original Ouro"),
            ("pred90_suf", "Pred90/SUF"),
        ):
            before = control["systems"][system_name]
            after = result["systems"][system_name]
            metrics = (
                (
                    "Decode wall time (s)",
                    before["timing"]["mean"],
                    after["timing"]["mean"],
                ),
                (
                    "KV wall-equivalent CUDA time (s)",
                    before["profile"]["coarse_regions"]["kv_prepare"]["cuda_seconds"],
                    after["profile"]["coarse_regions"]["kv_prepare"]["cuda_seconds"],
                ),
                (
                    "KV share of clean wall (%)",
                    before["profile"]["coarse_regions"]["kv_prepare"]["cuda_percent_of_wall"],
                    after["profile"]["coarse_regions"]["kv_prepare"]["cuda_percent_of_wall"],
                ),
                (
                    "cudaStreamSynchronize calls",
                    float(sync_calls(before)),
                    float(sync_calls(after)),
                ),
            )
            for metric, old, new in metrics:
                change = 100.0 * (new / old - 1.0) if old else 0.0
                lines.append(
                    f"| {label} | {metric} | {old:.4f} | {new:.4f} | {change:+.2f}% |"
                )
        lines.extend(
            [
                "",
                "Negative change means the optimized implementation reduced the metric.",
                "Both systems use the same workload, model revision, runtime layer count, and profiling protocol.",
                "",
            ]
        )
        (args.output_dir / "comparison.md").write_text(
            "\n".join(lines), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
