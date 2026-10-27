"""Leakage-conscious loading and target preparation for the Atomity challenge.

A full aggregated target series is constructed as the assignment requires,
but the final 24 observed target labels are discarded after the chronological
split. They are excluded from all model inputs, reports and EDA outputs.
The known-at-forecast-origin, static metadata fields are separated explicitly
from whole-trace CPU summary fields and VM deletion timestamps.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

READING_COLUMNS = ("timestamp", "vm_id", "min_cpu", "max_cpu", "avg_cpu")
STATIC_METADATA_COLUMNS = (
    "vm_id", "vm_category", "virtual_core_bucket", "memory_gb_bucket"
)
TARGET_COLUMNS = ("active_vm_count", "aggregate_cpu_utilization")


@dataclass
class TrainingBundle:
    readings: pd.DataFrame  # Only training timestamps, with VM-level telemetry
    metadata: pd.DataFrame  # Only static categorical/bucketed attributes
    targets: pd.DataFrame  # Only training timestamps
    all_timestamps: np.ndarray  # Timestamp VALUES only, never future CPU values
    forecast_timestamps: np.ndarray
    audit: dict


def load_training_bundle(
    data_dir: str | Path,
    *,
    train_steps: int | None = 110,
    horizon: int = 24,
) -> TrainingBundle:
    """Read supplied files, verify structural integrity, return train-only data.

    Notes:
      - Full-series timestamp values are public and necessary to prepare the
        24-row forecast template. The future CPU measurements remain unused.
      - ``train_steps``/``horizon`` are parameters, not hardcoded dataset facts:
        the loader can be checked on a smaller smoke dataset.
      - Strict metadata matching prevents silent duplication during joins.
    """
    if (train_steps is not None and train_steps < 1) or horizon < 1:
        raise ValueError("train_steps and horizon must both be positive")

    data_dir = Path(data_dir)
    readings_path = data_dir / "vm_cpu_readings.csv.gz"
    metadata_path = data_dir / "vm_metadata.csv.gz"

    cpu = pd.read_csv(readings_path, dtype={"vm_id": "string"})
    meta = pd.read_csv(metadata_path, dtype={"vm_id": "string"})
    if missing := set(READING_COLUMNS) - set(cpu.columns):
        raise ValueError(f"Missing CPU columns: {sorted(missing)}")
    if missing := set(STATIC_METADATA_COLUMNS) - set(meta.columns):
        raise ValueError(f"Missing metadata columns: {sorted(missing)}")
    if cpu.empty:
        raise ValueError("CPU data is empty")
    if cpu[["timestamp", "vm_id"]].isna().any().any():
        raise ValueError("Null timestamp or VM identifier")
    if meta["vm_id"].isna().any() or meta["vm_id"].duplicated().any():
        raise ValueError("Metadata must have one non-null row per VM")
    if cpu.duplicated(["timestamp", "vm_id"]).any():
        raise ValueError("Duplicate (timestamp, vm_id) observations")
    if not pd.api.types.is_numeric_dtype(cpu["timestamp"]):
        raise ValueError("Expected numeric relative-second timestamps")

    times = np.sort(cpu["timestamp"].unique())
    if not np.isfinite(times).all():
        raise ValueError("Timestamps must be finite")
    if train_steps is None:
        train_steps = len(times) - horizon
    if train_steps < 1:
        raise ValueError("Need training timestamps before the forecast horizon")
    if len(times) < train_steps + horizon:
        raise ValueError(
            f"Need at least {train_steps + horizon} timestamps; received {len(times)}"
        )
    # In the supplied assignment, exactly 110 + 24 timestamps exist. For a
    # replacement smoke dataset, use parameters matching its intended split.
    if len(times) != train_steps + horizon:
        raise ValueError(
            "Train/horizon parameters must cover all timestamps: "
            f"got {len(times)} timestamps, train_steps={train_steps}, horizon={horizon}"
        )
    if len(times) > 1:
        diffs = np.diff(times)
        if not np.all(diffs == 300):
            raise ValueError("Missing/irregular timestamp intervals detected")

    cpu_vm_ids = set(cpu["vm_id"].unique())
    metadata_vm_ids = set(meta["vm_id"].unique())
    if cpu_vm_ids != metadata_vm_ids:
        raise ValueError("CPU and metadata VM populations do not match")

    training_mask = cpu.timestamp.isin(times[:train_steps])
    for column in ('min_cpu', 'avg_cpu', 'max_cpu'):
        try:
            pd.to_numeric(cpu.loc[training_mask, column], errors='raise')
        except (ValueError, TypeError) as error:
            raise ValueError(f'Training {column} must be numeric') from error
        # Future label contents must not affect training dtype or aggregation.
        cpu[column] = pd.to_numeric(cpu[column], errors='coerce')

    # Required target construction: aggregate the complete timestamp series,
    # then split chronologically. This step computes held-out *labels*, but they
    # are discarded immediately and never returned, reported, fit, or inspected.
    # The forecast timestamps come only from the public time coordinates.
    # An absent VM row or null avg_cpu is unavailable telemetry, not zero CPU.
    # Count only valid averages; refuse an entirely unobserved training timestamp.
    full_targets = (
        cpu.loc[cpu.avg_cpu.notna()].groupby("timestamp", sort=True)
        .agg(active_vm_count=("vm_id", "nunique"),
             aggregate_cpu_utilization=("avg_cpu", "mean"))
        .reindex(times).rename_axis("timestamp").reset_index()
    )
    targets = full_targets.iloc[:train_steps].copy().reset_index(drop=True)
    del full_targets
    if targets[list(TARGET_COLUMNS)].isna().any().any():
        raise ValueError("Training timestamp has no valid CPU observations")
    # Empty future groups can upcast full-series counts; restore the training
    # count dtype explicitly so even label dtypes cannot cross the boundary.
    targets['active_vm_count'] = targets['active_vm_count'].astype(np.int64)

    # Leakage boundary: retain only training-window VM readings for all EDA.
    # Never use metadata whole-trace CPU summaries or deletion timestamps.
    train = cpu.loc[cpu["timestamp"].isin(times[:train_steps]), list(READING_COLUMNS)].copy()
    missing_averages = int(train.avg_cpu.isna().sum())
    train = train.loc[train.avg_cpu.notna()].copy()
    if train[["min_cpu", "avg_cpu", "max_cpu"]].isna().any().any():
        raise ValueError("Training CPU measurements contain nulls")
    vals = train[["min_cpu", "avg_cpu", "max_cpu"]].to_numpy(dtype=float)
    if not np.isfinite(vals).all() or np.any((vals < 0) | (vals > 100)):
        raise ValueError("Training CPU measurements must be finite in [0,100]")
    if not ((train.min_cpu <= train.avg_cpu) & (train.avg_cpu <= train.max_cpu)).all():
        raise ValueError("Training CPU minimum/average/maximum ordering invalid")

    audit = {
        "total_rows": int(len(cpu)),
        "unique_vms": int(len(cpu_vm_ids)),
        "metadata_rows": int(len(meta)),
        "all_timestamp_count": int(len(times)),
        "training_timestamp_count": int(len(targets)),
        "training_row_count": int(len(train)),
        "forecast_timestamp_count": int(horizon),
        "cadence_seconds": int(times[1] - times[0]) if len(times) > 1 else None,
        "training_missing_measurements": int(train[list(READING_COLUMNS)].isna().sum().sum()),
        "training_excluded_null_avg_cpu": missing_averages,
        "training_duplicate_vm_timestamp_pairs": 0,
    }
    return TrainingBundle(
        readings=train,
        metadata=meta.loc[:, list(STATIC_METADATA_COLUMNS)].copy(),
        targets=targets,
        all_timestamps=times,
        forecast_timestamps=times[train_steps:],
        audit=audit,
    )
