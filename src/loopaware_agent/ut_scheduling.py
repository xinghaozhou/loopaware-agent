"""Analytical UT-aware batching primitives.

Predicted UT is used only for batch formation and scheduling.  All execution
accounting is based on true, threshold-derived UT labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Iterable, Sequence


@dataclass(frozen=True)
class UTSample:
    """One active next-token sample available to the scheduler."""

    sample_id: str
    true_ut: int
    predicted_ut: int


@dataclass(frozen=True)
class UTBatch:
    """A fixed-capacity batch in formation order."""

    samples: tuple[UTSample, ...]
    formation_index: int

    @property
    def predicted_depth(self) -> int:
        return max(sample.predicted_ut for sample in self.samples)


@dataclass(frozen=True)
class BatchExecution:
    """True-UT lockstep execution accounting for one adaptively exiting batch."""

    batch: UTBatch
    true_max_ut: int
    active_rows_by_ut: tuple[int, ...]
    execution_cost: int
    useful_work: int
    wasted_slots: int
    true_ut_range: int
    completion_time: int


@dataclass(frozen=True)
class ScheduleEvaluation:
    batches: tuple[BatchExecution, ...]
    total_execution_cost: int
    total_useful_work: int
    total_wasted_slots: int
    mean_true_ut_range: float
    mean_batch_completion_time: float


def _validated_samples(
    samples: Iterable[UTSample], *, batch_size: int, u_max: int
) -> list[UTSample]:
    values = list(samples)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if u_max <= 0:
        raise ValueError("u_max must be positive")
    if not values:
        raise ValueError("samples must not be empty")
    if len(values) % batch_size:
        raise ValueError("sample count must be divisible by batch_size")
    for sample in values:
        if not 1 <= sample.true_ut <= u_max:
            raise ValueError(f"true_ut outside [1, {u_max}] for {sample.sample_id}")
        if not 1 <= sample.predicted_ut <= u_max:
            raise ValueError(
                f"predicted_ut outside [1, {u_max}] for {sample.sample_id}"
            )
    return values


def _batches(values: Sequence[UTSample], batch_size: int) -> list[UTBatch]:
    return [
        UTBatch(tuple(values[start : start + batch_size]), start // batch_size)
        for start in range(0, len(values), batch_size)
    ]


def form_fcfs_batches(
    samples: Iterable[UTSample], *, batch_size: int, u_max: int
) -> list[UTBatch]:
    """Form fixed-size batches without changing original request order."""

    values = _validated_samples(samples, batch_size=batch_size, u_max=u_max)
    return _batches(values, batch_size)


def form_homogeneous_batches(
    samples: Iterable[UTSample], *, batch_size: int, u_max: int
) -> list[UTBatch]:
    """Stably group adjacent samples after sorting by predicted UT."""

    values = _validated_samples(samples, batch_size=batch_size, u_max=u_max)
    ordered = sorted(values, key=lambda sample: sample.predicted_ut)
    return _batches(ordered, batch_size)


def schedule_shortest_ut_first(batches: Iterable[UTBatch]) -> list[UTBatch]:
    """Stably schedule batches from smallest to largest predicted depth."""

    values = list(batches)
    if not values:
        raise ValueError("batches must not be empty")
    return sorted(
        values,
        key=lambda batch: (batch.predicted_depth, batch.formation_index),
    )


def evaluate_lockstep_adaptive_exit(
    batches: Iterable[UTBatch], *, batch_size: int, u_max: int
) -> ScheduleEvaluation:
    """Evaluate adaptive exits under the fixed-width lockstep cost model.

    Rows become inactive after their true UT, but the batch occupies all
    ``batch_size`` recurrence slots until its slowest row exits.  Predictions
    are deliberately absent from every cost calculation.
    """

    values = list(batches)
    if not values:
        raise ValueError("batches must not be empty")
    completion_time = 0
    executions = []
    for batch in values:
        if len(batch.samples) != batch_size:
            raise ValueError("every batch must contain exactly batch_size samples")
        true_uts = [sample.true_ut for sample in batch.samples]
        if any(not 1 <= value <= u_max for value in true_uts):
            raise ValueError(f"true UT must lie in [1, {u_max}]")
        true_max = max(true_uts)
        active = tuple(
            sum(value >= recurrence for value in true_uts)
            for recurrence in range(1, true_max + 1)
        )
        useful = sum(active)
        cost = batch_size * true_max
        completion_time += cost
        executions.append(
            BatchExecution(
                batch=batch,
                true_max_ut=true_max,
                active_rows_by_ut=active,
                execution_cost=cost,
                useful_work=useful,
                wasted_slots=cost - useful,
                true_ut_range=true_max - min(true_uts),
                completion_time=completion_time,
            )
        )
    return ScheduleEvaluation(
        batches=tuple(executions),
        total_execution_cost=sum(item.execution_cost for item in executions),
        total_useful_work=sum(item.useful_work for item in executions),
        total_wasted_slots=sum(item.wasted_slots for item in executions),
        mean_true_ut_range=mean(item.true_ut_range for item in executions),
        mean_batch_completion_time=mean(
            item.completion_time for item in executions
        ),
    )
