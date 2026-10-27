"""Matched expanding-window multi-step backtests entirely within public training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import TARGET_COLUMNS, load_training_bundle
from .models import CANDIDATES, forecast


def origins(n: int, horizon: int) -> list[int]:
    """Official 110/24 gives [50,62,74,86]; adapt for smaller smoke files."""
    first, last = max(8, 2*horizon + 2), n-horizon
    if last < 8:
        raise ValueError(f'Need at least {8+horizon} training timestamps for backtesting')
    first = min(first, last)
    return np.unique(np.linspace(first, last, min(4, last-first+1), dtype=int)).tolist()


def evaluate(train: pd.DataFrame, horizon: int = 24):
    """Return raw predictions, summaries and deterministic per-target choices.

    Selection: require >5% pooled MAE improvement over persistence. Among
    candidates within 1% of minimum MAE choose first in simplicity order.
    This is model-development validation, not an unbiased final test estimate.
    """
    if (train.timestamp.isna().any() or train.timestamp.duplicated().any()
            or not train.timestamp.is_monotonic_increasing):
        raise ValueError('Training timestamps must be unique and sorted')
    if not np.isfinite(train[list(TARGET_COLUMNS)].to_numpy(dtype=float)).all():
        raise ValueError('Training targets must be finite')
    rows = []
    for origin in origins(len(train), horizon):
        past, actual = train.iloc[:origin], train.iloc[origin:origin+horizon]
        for target in TARGET_COLUMNS:
            # Peak threshold comes from this fold's past, never validation quantiles.
            threshold = float(past[target].quantile(.9))
            for model in CANDIDATES:
                pred = forecast(past[target], horizon, model, target)
                for h, (timestamp, truth, value) in enumerate(
                    zip(actual.timestamp, actual[target], pred), 1
                ):
                    rows.append(dict(target=target, model=model, origin=origin,
                                     horizon=h, timestamp=timestamp, actual=truth,
                                     prediction=value, error=value-truth,
                                     high_load=bool(truth > threshold)))
    detail = pd.DataFrame(rows)
    summaries = []
    for (target, model), frame in detail.groupby(['target', 'model'], sort=False):
        errors = frame.error.to_numpy()
        peak = frame.loc[frame.high_load, 'error']
        folds = frame.groupby('origin').error.apply(lambda x: float(x.abs().mean()))
        summaries.append(dict(target=target, model=model, mae=float(np.abs(errors).mean()),
                              rmse=float(np.sqrt(np.mean(errors**2))), bias=float(errors.mean()),
                              high_load_mae=float(peak.abs().mean()) if len(peak) else None,
                              high_load_n=len(peak), fold_mae_std=float(folds.std(ddof=0)),
                              worst_fold_mae=float(folds.max())))
    summary = pd.DataFrame(summaries)
    chosen = {}
    for target in TARGET_COLUMNS:
        view = summary.loc[summary.target == target].set_index('model')
        baseline = float(view.loc['persistence', 'mae'])
        best = float(view.mae.min())
        selected = 'persistence'
        if baseline > 0 and best < .95 * baseline:
            selected = next(m for m in CANDIDATES
                            if view.loc[m, 'mae'] < .95 * baseline
                            and view.loc[m, 'mae'] <= best * 1.01 + 1e-12)
        chosen[target] = selected
        mask = summary.target == target
        # Zero baseline error makes a ratio/improvement undefined, even for 0/0.
        summary.loc[mask, 'relative_mae'] = summary.loc[mask, 'mae']/baseline if baseline else np.nan
        summary.loc[mask, 'improvement_pct'] = (1-summary.loc[mask, 'mae']/baseline)*100 if baseline else np.nan
    return detail, summary, chosen


def diagnostics(detail, selected, horizon):
    """Exploratory pooled residual bands; chronological prequential coverage.

    Earlier-fold intervals use residuals whose labels are available at the next
    origin. The subsequent assessment windows can overlap.
    The selected model still used all folds, so coverage is descriptive only.
    """
    output = {}
    for target, model in selected.items():
        frame = detail[(detail.target == target) & (detail.model == model)]
        errors = frame.error.to_numpy()
        prequential = []
        for origin in sorted(frame.origin.unique())[1:]:
            old = frame[(frame.origin < origin) & (frame.origin + frame.horizon <= origin)]
            new = frame[frame.origin == origin]
            if len(old):
                radius = float(old.error.abs().quantile(.9, interpolation='higher'))
                prequential.extend((new.error.abs() <= radius).tolist())
        output[target] = dict(model=model,
            absolute_error_p90=float(np.quantile(np.abs(errors), .9, method='higher')),
            signed_error_p10=float(np.quantile(errors, .1)),
            signed_error_p90=float(np.quantile(errors, .9)),
            pooled_forecast_errors=len(errors), unique_validation_timestamps=int(frame.timestamp.nunique()),
            prequential_coverage=float(np.mean(prequential)) if prequential else None,
            prequential_n=len(prequential),
            warning='Exploratory only: overlapping folds, model selection and few origins preclude calibrated coverage')
    return output


def write_results(train, horizon, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    detail, summary, selected = evaluate(train, horizon)
    detail.to_csv(output/'backtest_predictions.csv', index=False)
    summary.to_csv(output/'model_comparison.csv', index=False)
    fold = detail.groupby(['target','model','origin']).error.agg(
        mae=lambda x: x.abs().mean(), rmse=lambda x: np.sqrt((x*x).mean()), bias='mean')
    fold.to_csv(output/'fold_metrics.csv')
    by_horizon = detail.groupby(['target','model','horizon']).error.agg(
        mae=lambda x: x.abs().mean(), rmse=lambda x: np.sqrt((x*x).mean()), bias='mean')
    by_horizon.to_csv(output/'horizon_metrics.csv')
    info = dict(selected=selected, origins=origins(len(train), horizon), horizon=horizon,
                training_steps=len(train), selection_rule='>5% MAE improvement over persistence; simplest within 1% of best',
                uncertainty=diagnostics(detail, selected, horizon))
    (output/'selection.json').write_text(json.dumps(info, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    plot_results(detail, selected, output/'figures')
    return info


def plot_results(detail, selected, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 9, 'savefig.dpi': 150})
    target = 'aggregate_cpu_utilization'
    chosen = detail[(detail.target == target) & (detail.model == selected[target])]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), squeeze=False)
    for ax, (origin, frame) in zip(axes.flat, chosen.groupby('origin')):
        ax.plot(frame.horizon*5, frame.actual, label='Actual')
        ax.plot(frame.horizon*5, frame.prediction, label=f'{selected[target]} (selected)')
        base = detail[(detail.target == target) & (detail.model == 'persistence') & (detail.origin == origin)]
        if selected[target] != 'persistence':
            ax.plot(base.horizon*5, base.prediction, '--', label='Persistence')
        ax.set(title=f'Train steps: {origin}', xlabel='Minutes from origin', ylabel='Mean CPU (%)')
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    fig.suptitle('24-step internal CPU forecasts (training labels only)')
    fig.tight_layout()
    fig.savefig(output/'06_validation_forecasts.png')
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for origin, frame in chosen.groupby('origin'):
        axes[0].plot(frame.horizon, frame.error, alpha=.8, label=f'Origin {origin}')
    axes[0].axhline(0, color='black', linewidth=.8)
    axes[0].set(title='Signed errors by fold', xlabel='Forecast step (5 min)', ylabel='Prediction - actual (pp)')
    axes[0].legend(fontsize=7)
    for model in dict.fromkeys(['persistence', selected[target]]):
        frame = detail[(detail.target == target) & (detail.model == model)]
        axes[1].plot(frame.groupby('horizon').error.apply(lambda x:x.abs().mean()), label=model)
    axes[1].set(title='Mean absolute error versus horizon', xlabel='Forecast step (5 min)', ylabel='MAE (pp)')
    axes[1].legend(fontsize=7)
    for ax in axes: ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(output/'07_residual_horizon.png')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('challenge_data'))
    parser.add_argument('--output', type=Path, default=Path('reports'))
    parser.add_argument('--horizon', type=int, default=24)
    args = parser.parse_args()
    bundle = load_training_bundle(args.data, train_steps=None, horizon=args.horizon)
    print(json.dumps(write_results(bundle.targets, args.horizon, args.output), indent=2))


if __name__ == '__main__':
    main()
