"""Small deterministic forecasting candidates; only past target vectors enter fit."""
from __future__ import annotations

import numpy as np

# Ordered by preference for simplicity. Fixed before examining backtest results.
CANDIDATES = (
    'persistence', 'mean', 'moving_average_6', 'moving_average_24',
    'ses_0.2', 'ses_0.6', 'holt_damped', 'ridge_6_1', 'ridge_6_10',
)


def forecast(values, horizon: int, model: str, target: str) -> np.ndarray:
    """Generate an entire horizon recursively or from the final fitted state.

    SES uses alpha=.2/.6; Holt uses alpha=.2, beta=.1, damping=.9.
    Ridge uses six oldest-to-newest lags, a training-only centered intercept,
    and fixed L2 penalties on raw target units (no global scaler).
    """
    y = np.asarray(values, dtype=float)
    if y.ndim != 1 or len(y) < 8 or not np.isfinite(y).all():
        raise ValueError('Need at least 8 finite training values')
    if horizon < 1 or model not in CANDIDATES:
        raise ValueError('Invalid horizon or model')
    if target not in ('active_vm_count', 'aggregate_cpu_utilization'):
        raise ValueError('Unknown target')
    upper = 100. if target == 'aggregate_cpu_utilization' else np.inf
    if model == 'persistence':
        pred = np.repeat(y[-1], horizon)
    elif model == 'mean':
        pred = np.repeat(y.mean(), horizon)
    elif model.startswith('moving_average'):
        pred = np.repeat(y[-int(model.split('_')[-1]):].mean(), horizon)
    elif model.startswith('ses'):
        alpha = float(model.split('_')[-1])
        level = y[0]
        for value in y[1:]:
            level = alpha * value + (1 - alpha) * level
        pred = np.repeat(level, horizon)
    elif model == 'holt_damped':
        level, trend = y[0], 0.
        for value in y[1:]:
            previous = level
            level = .2 * value + .8 * (level + .9 * trend)
            trend = .1 * (level - previous) + .9 * .9 * trend
        pred = level + np.cumsum(.9 ** np.arange(1, horizon + 1)) * trend
    else:
        lag, penalty = 6, float(model.split('_')[-1])
        x = np.array([y[i-lag:i] for i in range(lag, len(y))])
        z = y[lag:]
        xm, zm = x.mean(axis=0), z.mean()
        xc = x - xm
        coef = np.linalg.solve(xc.T @ xc + penalty * np.eye(lag), xc.T @ (z-zm))
        history, pred = y.tolist(), []
        for _ in range(horizon):
            value = float(np.clip(zm + (np.array(history[-lag:])-xm) @ coef, 0, upper))
            pred.append(value)
            history.append(value)  # Predictions, never horizon actuals.
    result = np.clip(np.asarray(pred, dtype=float), 0, upper)
    if not np.isfinite(result).all():
        raise ValueError(f'Nonfinite forecast from {model}')
    return result
