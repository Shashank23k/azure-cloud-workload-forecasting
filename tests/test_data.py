"""Data-contract, smoke-data, and holdout-isolation tests."""
from __future__ import annotations

import pandas as pd
import pytest

from cwfc.data import load_training_bundle


def _toy_files(tmp_path, holdout_values=(92., 99.)):
    cpu = pd.DataFrame(
        [
            {"timestamp": t * 300, "vm_id": vm,
             "min_cpu": val, "max_cpu": val, "avg_cpu": val}
            for t in range(6)
            for vm, val in (("vm_a", [10, 20, 30, 40, *holdout_values][t]),
                            ("vm_b", [11, 21, 31, 41, *holdout_values][t]))
        ]
    )
    meta = pd.DataFrame([
        {"vm_id": "vm_a", "vm_category": "Interactive", "virtual_core_bucket": 2,
         "memory_gb_bucket": 4, "avg_cpu": 99.},
        {"vm_id": "vm_b", "vm_category": "Unknown", "virtual_core_bucket": 4,
         "memory_gb_bucket": 8, "avg_cpu": 99.},
    ])
    cpu.to_csv(tmp_path / "vm_cpu_readings.csv.gz", index=False, compression="gzip")
    meta.to_csv(tmp_path / "vm_metadata.csv.gz", index=False, compression="gzip")


def test_holdout_cpu_values_cannot_change_training_targets(tmp_path):
    _toy_files(tmp_path, (95., 99.))
    first = load_training_bundle(tmp_path, train_steps=4, horizon=2)
    _toy_files(tmp_path, (0., 1.))
    second = load_training_bundle(tmp_path, train_steps=4, horizon=2)
    pd.testing.assert_frame_equal(first.targets, second.targets)
    pd.testing.assert_frame_equal(first.readings, second.readings)
    assert len(first.forecast_timestamps) == 2
    assert "avg_cpu" not in first.metadata.columns  # whole-trace stat forbidden
    assert first.targets.active_vm_count.tolist() == [2] * 4
    _toy_files(tmp_path, (float('nan'), float('nan')))
    redacted = load_training_bundle(tmp_path, train_steps=4, horizon=2)
    pd.testing.assert_frame_equal(first.targets, redacted.targets)


def test_fails_with_duplicate_vm_timestamp(tmp_path):
    _toy_files(tmp_path)
    path = tmp_path / "vm_cpu_readings.csv.gz"
    cpu = pd.read_csv(path)
    pd.concat([cpu, cpu.iloc[:1]], ignore_index=True).to_csv(
        path, index=False, compression="gzip"
    )
    with pytest.raises(ValueError, match="Duplicate"):
        load_training_bundle(tmp_path, train_steps=4, horizon=2)


def test_fails_with_irregular_timestamp(tmp_path):
    _toy_files(tmp_path)
    path = tmp_path / "vm_cpu_readings.csv.gz"
    cpu = pd.read_csv(path)
    cpu.loc[cpu.timestamp == 900, "timestamp"] = 901
    cpu.to_csv(path, index=False, compression="gzip")
    with pytest.raises(ValueError, match="irregular"):
        load_training_bundle(tmp_path, train_steps=4, horizon=2)
