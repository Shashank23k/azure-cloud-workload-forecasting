"""Reproduce submission without exposing holdout labels to forecasting code."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import evaluate
from .data import TARGET_COLUMNS, load_training_bundle
from .models import forecast

COLUMNS = ['timestamp', *TARGET_COLUMNS]


def validate_predictions(frame: pd.DataFrame, timestamps) -> None:
    expected = np.asarray(timestamps)
    if list(frame.columns) != COLUMNS or len(frame) != len(expected) or not len(frame):
        raise ValueError('Prediction schema or horizon mismatch')
    if not np.array_equal(frame.timestamp.to_numpy(), expected):
        raise ValueError('Prediction timestamps do not match template')
    if frame.timestamp.duplicated().any() or not frame.timestamp.is_monotonic_increasing:
        raise ValueError('Prediction timestamps must be unique and sorted')
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError('Predictions must be finite')
    if (frame.active_vm_count < 0).any() or not frame.aggregate_cpu_utilization.between(0,100).all():
        raise ValueError('Predictions outside target bounds')


def generate_predictions(train: pd.DataFrame, timestamps):
    """The only future information accepted by this API is timestamp coordinates."""
    timestamps = np.asarray(timestamps)
    if len(timestamps) < 1 or timestamps[0] <= train.timestamp.iloc[-1]:
        raise ValueError('Forecast timestamps must follow training')
    _, _, selected = evaluate(train, len(timestamps))
    frame = pd.DataFrame({'timestamp': timestamps})
    for target in TARGET_COLUMNS:
        frame[target] = forecast(train[target], len(timestamps), selected[target], target)
    validate_predictions(frame, timestamps)
    return frame, selected


def run(data, output, horizon=24):
    bundle = load_training_bundle(data, train_steps=None, horizon=horizon)
    frame, selected = generate_predictions(bundle.targets, bundle.forecast_timestamps)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    validate_predictions(pd.read_csv(output), bundle.forecast_timestamps)
    return frame, selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('challenge_data'))
    parser.add_argument('--output', type=Path, default=Path('predictions.csv'))
    parser.add_argument('--horizon', type=int, default=24)
    args = parser.parse_args()
    try:
        frame, selected = run(args.data, args.output, args.horizon)
    except (ValueError, FileNotFoundError) as error:
        parser.exit(2, f'Prediction failed: {error}\n')
    print(f'Wrote {len(frame)} validated rows to {args.output}; models: {selected}')


if __name__ == '__main__':
    main()
