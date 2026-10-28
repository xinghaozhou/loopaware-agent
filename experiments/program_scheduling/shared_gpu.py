"""Invariants for one non-overlapping physical GPU batch timeline."""

from __future__ import annotations

import math
from typing import Any


def verify_single_shared_gpu(
    schedule: list[dict[str, Any]], *, maximum_batch_size: int
) -> dict[str, Any]:
    previous_finish = -math.inf
    for batch in schedule:
        if not 1 <= int(batch["batch_size"]) <= maximum_batch_size:
            raise RuntimeError("physical batch width is outside the configured limit")
        if float(batch["start_time_ms"]) < previous_finish - 1e-6:
            raise RuntimeError("overlapping physical GPU batches detected")
        if float(batch["finish_time_ms"]) <= float(batch["start_time_ms"]):
            raise RuntimeError("non-positive physical batch duration")
        previous_finish = float(batch["finish_time_ms"])
    return {
        "verified": True,
        "execution_model": "one_shared_static_gpu_batch",
        "maximum_overlapping_gpu_batches": 1,
        "configured_batch_size": maximum_batch_size,
        "observed_maximum_batch_size": max(
            (int(batch["batch_size"]) for batch in schedule), default=0
        ),
        "batches_checked": len(schedule),
    }
