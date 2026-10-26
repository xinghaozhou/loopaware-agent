"""Create dependency-free SVG plots for the BFCL scheduling validation."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


COLORS = ["#4c78a8", "#f58518", "#e45756", "#72b7b2", "#54a24b", "#b279a2"]


def svg_start(title: str, width: int = 900, height: int = 520) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="450" y="32" text-anchor="middle" font-family="sans-serif" font-size="20">{html.escape(title)}</text>',
    ]


def bar_plot(path: Path, title: str, labels: list[str], values: list[float], ylabel: str) -> None:
    lines = svg_start(title)
    maximum = max(values) * 1.1
    left, top, bottom, plot_width = 90, 55, 430, 760
    bar_width = plot_width / len(values) * 0.65
    for index, (label, value) in enumerate(zip(labels, values, strict=True)):
        x = left + (index + 0.5) * plot_width / len(values) - bar_width / 2
        height = (bottom - top) * value / maximum
        y = bottom - height
        lines.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{height:.1f}" fill="{COLORS[index]}"/>')
        lines.append(f'<text x="{x + bar_width / 2:.1f}" y="{y - 7:.1f}" text-anchor="middle" font-family="sans-serif" font-size="12">{value:.3f}</text>')
        lines.append(f'<text x="{x + bar_width / 2:.1f}" y="450" text-anchor="middle" font-family="sans-serif" font-size="11" transform="rotate(25 {x + bar_width / 2:.1f} 450)">{html.escape(label)}</text>')
    lines.extend([
        f'<line x1="{left}" y1="{bottom}" x2="{left + plot_width}" y2="{bottom}" stroke="black"/>',
        f'<text x="20" y="250" text-anchor="middle" font-family="sans-serif" font-size="13" transform="rotate(-90 20 250)">{html.escape(ylabel)}</text>',
        "</svg>",
    ])
    path.write_text("\n".join(lines) + "\n")


def scatter(path: Path, title: str, xs: list[float], ys: list[float], xlabel: str) -> None:
    lines = svg_start(title)
    left, top, right, bottom = 85, 55, 860, 445
    xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
    for x, y in zip(xs, ys, strict=True):
        px = left + (x - xmin) / (xmax - xmin or 1) * (right - left)
        py = bottom - (y - ymin) / (ymax - ymin or 1) * (bottom - top)
        lines.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4" fill="#4c78a8" fill-opacity="0.7"/>')
    lines.extend([
        f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="black"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="black"/>',
        f'<text x="470" y="495" text-anchor="middle" font-family="sans-serif" font-size="14">{html.escape(xlabel)}</text>',
        '<text x="20" y="250" text-anchor="middle" font-family="sans-serif" font-size="14" transform="rotate(-90 20 250)">Full-compute GPU runtime (ms)</text>',
        "</svg>",
    ])
    path.write_text("\n".join(lines) + "\n")


def cdf(path: Path, simulations: list[dict]) -> None:
    lines = svg_start("Program completion-time CDF")
    left, top, right, bottom = 85, 55, 860, 445
    all_values = [row["completion_time_ms"] / 1000 for sim in simulations for row in sim["programs"]]
    maximum = max(all_values)
    for index, simulation in enumerate(simulations):
        values = sorted(row["completion_time_ms"] / 1000 for row in simulation["programs"])
        points = []
        for rank, value in enumerate(values, start=1):
            x = left + value / maximum * (right - left)
            y = bottom - rank / len(values) * (bottom - top)
            points.append(f"{x:.1f},{y:.1f}")
        lines.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{COLORS[index]}" stroke-width="2"/>')
        lines.append(f'<text x="650" y="{75 + index * 19}" font-family="sans-serif" font-size="12" fill="{COLORS[index]}">{html.escape(simulation["policy"])}</text>')
    lines.extend([
        f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="black"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="black"/>',
        '<text x="470" y="490" text-anchor="middle" font-family="sans-serif" font-size="14">Completion time (s)</text>',
        '<text x="20" y="250" text-anchor="middle" font-family="sans-serif" font-size="14" transform="rotate(-90 20 250)">CDF</text>',
        "</svg>",
    ])
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    trace, results = json.loads(args.trace.read_text()), json.loads(args.results.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = results["summaries"]
    labels = [row["policy"] for row in summaries]
    plas = next(row["mean_ttlt_ms"] for row in summaries if row["policy"] == "GPU_PLAS")
    bar_plot(args.output_dir / "normalized_mean_ttlt.svg", "Normalized mean TTLT", labels, [row["mean_ttlt_ms"] / plas for row in summaries], "Mean TTLT / GPU-PLAS")
    bar_plot(args.output_dir / "p95_ttlt.svg", "P95 program completion time", labels, [row["p95_ttlt_ms"] / 1000 for row in summaries], "P95 TTLT (s)")
    cdf(args.output_dir / "completion_cdf.svg", results["simulations"])
    included = set(results.get("program_ids", []))
    turns = [
        turn
        for program in trace["programs"]
        if not included or program["program_id"] in included
        for turn in program["turns"]
    ]
    runtime = [turn["full_compute_gpu_time_ms"] for turn in turns]
    scatter(args.output_dir / "selected_ut_vs_runtime.svg", "Selected UT workload vs runtime", [turn["sum_selected_ut"] for turn in turns], runtime, "Sum selected UT")
    scatter(args.output_dir / "tokens_vs_runtime.svg", "Output tokens vs runtime", [turn["output_token_count"] for turn in turns], runtime, "Output token count")
    characterization = results["characterization"]
    bar_plot(
        args.output_dir / "pairwise_accuracy.svg",
        "READY-pair runtime ordering under GPU-PLAS",
        ["Selected UT", "Output tokens"],
        [characterization["ready_pairwise_selected_ut_runtime_accuracy_under_gpu_plas"], characterization["ready_pairwise_output_tokens_runtime_accuracy_under_gpu_plas"]],
        "Tie-aware accuracy",
    )


if __name__ == "__main__":
    main()
