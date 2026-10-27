"""Compare FCFS, the old LAS proxy, and Autellix-style PLAS on BFCL traces."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from .autellix_plas import PLASConfig, simulate_batched


POLICIES = ("FCFS", "GPU_PLAS_PROXY", "AUTELLIX_PLAS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--program-counts", type=int, nargs="+", default=[32, 64])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--boundaries-ms", type=float, nargs="+", default=None)
    parser.add_argument("--quanta-ms", type=float, nargs="+", default=None)
    parser.add_argument("--anti-starvation-beta", type=float, default=8.0)
    args = parser.parse_args()
    bundle = json.loads(args.trace.read_text(encoding="utf-8"))
    defaults = PLASConfig()
    config = PLASConfig(
        boundaries_ms=tuple(args.boundaries_ms or defaults.boundaries_ms),
        quanta_ms=tuple(args.quanta_ms or defaults.quanta_ms),
        anti_starvation_beta=args.anti_starvation_beta,
    )
    experiments = []
    for count in args.program_counts:
        programs = bundle["programs"][:count]
        if len(programs) != count:
            raise ValueError(f"trace has fewer than {count} programs")
        for batch_size in args.batch_sizes:
            for policy in POLICIES:
                result = simulate_batched(programs, policy, batch_size=batch_size, config=config)
                experiments.append(result)
                summary = result["summary"]
                print(
                    f"DONE programs={count} batch={batch_size} policy={policy} "
                    f"mean_ttlt_ms={summary['mean_program_ttlt_ms']:.1f} "
                    f"preemptions={summary['preemptions']}",
                    flush=True,
                )
    payload = {
        "configuration": {
            "source_trace": str(args.trace),
            "source_configuration": bundle.get("configuration"),
            "program_counts": args.program_counts,
            "batch_sizes": args.batch_sizes,
            "policies": list(POLICIES),
            "queue_boundaries_ms": list(config.boundaries_ms),
            "queue_quanta_ms": list(config.quanta_ms),
            "anti_starvation_beta": config.anti_starvation_beta,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "experiments": experiments,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
