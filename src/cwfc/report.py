"""Regenerate the evidence-based Markdown report and exploratory forecast bands."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import write_results
from .data import load_training_bundle
from .eda import write_eda
from .predict import generate_predictions


def table(frame, columns, digits=4):
    def cell(value):
        if pd.isna(value): return 'undefined'
        return f'{value:.{digits}f}' if isinstance(value, (float, np.floating)) else str(value)
    return '| ' + ' | '.join(columns) + ' |\n|' + '|'.join(['---']*len(columns)) + '|\n' + '\n'.join(
        '| ' + ' | '.join(cell(row[c]) for c in columns) + ' |' for _, row in frame.iterrows())


def build(data, output):
    output = Path(output)
    bundle = load_training_bundle(data, train_steps=None)
    stats = write_eda(bundle, output)
    selection = write_results(bundle.targets, 24, output)
    predictions, _ = generate_predictions(bundle.targets, bundle.forecast_timestamps)
    comparison = pd.read_csv(output/'model_comparison.csv')
    detail = pd.read_csv(output/'backtest_predictions.csv')
    cpu = comparison[comparison.target == 'aggregate_cpu_utilization']
    chosen = selection['selected']['aggregate_cpu_utilization']
    best = cpu[cpu.model == chosen].iloc[0]
    errors = detail[(detail.target == 'aggregate_cpu_utilization') & (detail.model == chosen)]
    folds = pd.read_csv(output/'fold_metrics.csv')
    folds = folds[(folds.target == 'aggregate_cpu_utilization') & (folds.model == chosen)]
    horizon = pd.read_csv(output/'horizon_metrics.csv')
    horizon = horizon[(horizon.target == 'aggregate_cpu_utilization') & (horizon.model == chosen)]
    radius = selection['uncertainty']['aggregate_cpu_utilization']['absolute_error_p90']
    coverage = selection['uncertainty']['aggregate_cpu_utilization']['prequential_coverage']
    band = predictions.copy()
    band['cpu_lower_exploratory'] = np.maximum(0, band.aggregate_cpu_utilization-radius)
    band['cpu_upper_exploratory'] = np.minimum(100, band.aggregate_cpu_utilization+radius)
    band.to_csv(output/'forecast_uncertainty.csv', index=False)
    plot_forecast(bundle, band, output/'figures'/'08_final_forecast.png')
    a, b = stats['training_target_statistics'], stats['training_vm_level_statistics']
    sections = []
    for lo, hi in [(1,6),(7,12),(13,18),(19,24)]:
        part = errors[errors.horizon.between(lo,hi)]
        sections.append(dict(steps=f'{lo}-{hi}', mae=part.error.abs().mean(), bias=part.error.mean()))
    group_table = table(pd.DataFrame(sections), ['steps','mae','bias'])
    fold_table = table(folds, ['origin','mae','rmse','bias'])
    cat_table = table(pd.DataFrame([dict(category=k,vms=v['vms'],mean_cpu=v['mean_cpu'],
                                        high_cpu_pct=v['share_vm_readings_avg_cpu_ge_80']*100)
                                  for k,v in stats['categories'].items()]),
                      ['category','vms','mean_cpu','high_cpu_pct'])
    forecast_value = predictions.aggregate_cpu_utilization.iloc[0]
    text = f'''# Cloud workload forecasting: final technical report

Microsoft Azure Public Dataset V2, Atomity curated `data-v1`. All empirical
performance claims below use internal training-window validation. Regenerate
with `python -m cwfc.report --data challenge_data --output reports`.

## 1. Executive summary

The fleet's unweighted CPU mean is low but masks concentrated load. Across
{stats['training_row_count']:,} permitted training VM observations, mean fleet
CPU is **{a['aggregate_cpu_mean']:.4f}%** and the range is
{a['aggregate_cpu_min']:.4f}%-{a['aggregate_cpu_max']:.4f}%. Meanwhile,
{100*b['fraction_at_least_80pct_avg_cpu']:.2f}% of VM observations have average
CPU at least 80%. Aggregate capacity and individual VM pressure require
separate monitoring.

Nine compact candidates were compared at four expanding origins. Persistence
is selected separately for both targets: CPU validation MAE is
**{best.mae:.4f} percentage points (pp)** and RMSE **{best.rmse:.4f} pp**;
active-count MAE and RMSE are zero in this curated, constant-population sample.
Ridge slightly improves CPU RMSE but does not improve MAE. There is insufficient
evidence to justify added complexity under the stated MAE selection rule.

The final two-hour prediction is {forecast_value:.4f}% mean CPU and
{predictions.active_vm_count.iloc[0]:,.0f} active VMs at every required timestamp,
derived from the last training observations. The observed public holdout
labels have never been scored or used to choose a model. These predictions
support monitoring and cautious planning; they do not establish safe physical
capacity reductions.

## 2. Dataset and preprocessing

The release samples 10,000 complete VM traces, stratified by category and core
bucket using deterministic identifier ranking. There are 134 timestamps at
300-second cadence (1,340,000 rows); the first 110 form training and the last
24 are reserved. Source timestamps are relative trace seconds. No wall-clock,
daily or weekly features are invented. Three SHA-256 manifest entries were
verified. All training observations pass range and min/average/max consistency
checks, with zero missing measurements or duplicate VM/timestamp pairs.

The target constructor counts distinct VM IDs with non-null average CPU and
averages their `avg_cpu` percentages without weighting. Duplicate pairs are
rejected rather than arbitrarily averaged. Missing VM rows or null averages
are excluded from both targets; partial telemetry is not imputed as idle CPU.
An entirely unobserved training timestamp, invalid CPU range/order, or missing
five-minute grid point raises an explicit error. This distinguishes missing
telemetry from actual inactivity. The supplied sample needs no imputation.

The complete target series is constructed and immediately sliced; holdout
labels are discarded inside the loader. Downstream prediction accepts only a
training target frame and future timestamp coordinates. EDA receives only
training readings. Metadata CPU summaries and deletion times are excluded;
static categories and hardware buckets serve descriptive segmentation only.
No model uses VM identities or metadata, and all Ridge centering, lag features,
smoothing states and residual summaries are constructed within the appropriate
training prefix. Synthetic tests change future CPU values and potentially
leaking metadata and verify identical model choices and forecast CSV bytes.

## 3. Workload understanding

Training covers 110 intervals, with first-to-last span 9.083 hours. Every VM
has all 110 observations. There are zero observed telemetry arrivals or
departures within this cohort. Complete-trace selection explains constant
activity; it provides no evidence that real Azure fleets have no churn.
Fleet CPU standard deviation is {a['aggregate_cpu_std']:.4f} pp and lag-one
autocorrelation is {a['aggregate_cpu_lag1_autocorrelation']:.3f}. The maximum
training aggregate occurs at relative second {a['aggregate_cpu_peak_timestamp']},
and the minimum at {a['aggregate_cpu_low_timestamp']}. The short rising and
falling excursions do not establish repeatable seasonality.

![Training fleet CPU, relative elapsed time](figures/01_aggregate_cpu_train.png)

The VM-reading median is {b['avg_cpu_median']:.2f}%, P95
{b['avg_cpu_p95']:.2f}% and P99 {b['avg_cpu_p99']:.2f}%. Below-5% readings
account for {100*b['fraction_below_5pct_avg_cpu']:.2f}%. High five-minute maxima
occur in {100*b['fraction_at_least_80pct_max_cpu']:.2f}% of readings, distinct
from the {100*b['fraction_at_least_80pct_avg_cpu']:.2f}% with high averages.
Thus brief bursts and sustained saturation need separate alerts. The maximum
high-average fraction at one timestamp is
{100*b['max_high_avg_share_by_timestamp']:.2f}%.

{cat_table}

CPU columns above are percentages; `high_cpu_pct` is the percentage of segment
readings with average CPU >=80%. Interactive workloads have greater average
and high-tail demand than delay-insensitive workloads, suggesting that
latency-sensitive segments deserve separate protection. The descriptive
category labels' availability should be verified before production use.
Core and memory bucket tables are reproduced in [EDA](eda_summary.md).
Bucket labels are not exact capacity counts; no core-weighted demand is inferred.
Additional distribution and hot-VM figures are in that companion analysis.

## 4. Forecasting approach and validation

Candidates are persistence, full-history mean, trailing means of 6 and 24 steps,
SES with fixed alpha 0.2 and 0.6, damped Holt (alpha 0.2, beta 0.1, phi 0.9),
and recursive Ridge with six lags and penalties 1 and 10. SES starts at the
first observation; Holt starts at that level with zero trend. Ridge centers
its design and response within each fit and penalizes coefficients, not the
intercept. Its target uses the next observation after each six-value window.
All models generate the full horizon without consuming validation actuals;
recursive Ridge appends predictions to its own history. CPU is clipped to
[0,100] and counts to nonnegative values. These are deterministic NumPy
implementations without optimizer or random-seed dependence.

Expanding folds train on indices 0-49, 0-61, 0-73 and 0-85, then predict the
next 24 observations each. Every candidate uses identical folds and no random
split. These provide 96 forecast errors per model but only 60 unique validation
timestamps because origins overlap by 12 steps. This mirrors frequent forecast
refreshes, while correlated folds limit effective sample size and prevent
strong significance claims. Every observation is weighted equally within
pooled MAE. The final 24 public observations are outside all folds.

The explicit selection rule requires more than 5% pooled MAE improvement over
persistence; among candidates within 1% of the minimum MAE, use the first in
predeclared simplicity order. MAE is the primary loss because it measures
average absolute utilization error directly. RMSE, signed bias, peak error
and stability provide additional evidence; selecting by a different loss
could favor Ridge. The 5% gate is an engineering complexity threshold, not a
statistical significance test: a modest gain is required to justify additional
maintenance given only four dependent origins. It was not tuned to holdout
performance. Persistence is the outright MAE minimum here even without the gate.
Fixed small parameter sets reduce search
pressure; validation still informs model selection, so these are development
estimates rather than independent final-test performance.

## 5. Quantitative results

CPU metrics are in percentage points. Relative MAE divides by persistence
MAE; negative improvement means a worse result.

{table(cpu, ['model','mae','rmse','relative_mae','improvement_pct'])}

All active-count candidates have MAE/RMSE zero on these folds. Their relative
MAE and percentage improvement are **undefined** because baseline error is
zero. Persistence is the simplest tie winner, using the observed count rather
than a hard-coded VM population. High-load count observations are absent, so
count peak-error metrics are also undefined.

Selected CPU model by origin (number of available training steps):

{fold_table}

Persistence's fold MAE standard deviation is {best.fold_mae_std:.4f} pp and
its worst-fold MAE is {best.worst_fold_mae:.4f} pp. Fold-dependent bias changes
sign: an origin before an upward excursion underpredicts, while a relatively
high origin followed by lower load overpredicts. A flat forecast cannot
anticipate either event. Smoothing delays response to such shifts; recursive
Ridge regresses toward its estimated local equilibrium and can miss peaks.

![Matched internal 24-step CPU forecasts](figures/06_validation_forecasts.png)

## 6. Failure analysis and uncertainty

Signed error is prediction minus actual. Pooled selected-model bias is
{best.bias:+.4f} pp. High load means actual CPU exceeds the **fold training**
90th percentile, with the threshold fixed before that forecast. Across
{int(best.high_load_n)} such forecast instances, MAE is
{best.high_load_mae:.4f} pp, larger than overall MAE. Overlap means these
instances are not independent extreme events. Per-step errors and every
candidate's per-fold metrics are available in the CSV appendices.

{group_table}

Grouped horizon metrics above use equal numbers of forecast instances per
step. Performance does not have to deteriorate monotonically: the realized
shape of these few validation trajectories matters more than nominal horizon.

![Signed residuals and horizon-specific MAE](figures/07_residual_horizon.png)

The pooled 90th percentile of absolute selected CPU residuals is
{radius:.4f} pp. Adding/subtracting this fixed radius gives an exploratory
sensitivity band of {band.cpu_lower_exploratory.iloc[0]:.4f}%-
{band.cpu_upper_exploratory.iloc[0]:.4f}% for the final point forecast.
`forecast_uncertainty.csv` stores these bounds separately from submission
predictions. A chronological diagnostic calibrates using earlier folds whose
labels have already arrived at each subsequent origin; its descriptive
coverage is {coverage*100:.2f}% over 72 forecast instances. That is not proof
of nominal coverage: errors overlap, only four origins exist, model selection
used all folds, and future regimes may differ. The bounds pool horizons and
do not model horizon-dependent risk. A zero active-count residual band
reflects sample design and must not imply zero future fleet uncertainty.

## 7. Capacity recommendation

Use the point forecast as a near-term aggregate monitoring reference. Keep
existing operational headroom until placement, true core demand, memory,
latency and scaling costs are known. As an illustrative warning trigger,
compare observed fleet CPU with forecast plus the {radius:.4f} pp historical
error radius, and investigate persistent exceedances rather than automatically
adding hosts. This margin is a utilization discrepancy buffer, not a measured
number of spare cores. A breach may indicate drift or changed telemetry.

At the same time, monitor per-VM average and maximum CPU, hot-VM fraction,
queue depth and latency separately. Repeated average CPU >=80% is an
illustrative investigation threshold, not a validated service-level limit.
Prioritize interactive workloads and deployment/tenant hotspots; defer
flexible batch work when segment pressure and latency rise together.
Do not consolidate hosts simply because the unweighted fleet mean is low.
Under-provisioning risks queue buildup and missed SLOs; over-provisioning buys
headroom at recurring cost. Production actions should include cooldowns,
hysteresis, minimum capacity and rollback, with thresholds established from
costs and service behavior rather than this short curated trace.

## 8. Limitations and next work

This sample excludes incomplete traces and is stratified rather than a random
fleet snapshot. It cannot validate real churn, rare failures or fleet-wide
rates. Roughly nine hours of training cannot identify daily/weekly cycles,
regime changes or reliable tail probabilities. Static classification may be
retrospective; future lifecycle timestamps and whole-trace summaries are
particularly unsafe. Missing telemetry can reduce an apparent active count
without a VM actually stopping.

A production policy needs longer representative history, exact vCPU/memory
allocations, VM-host placement and oversubscription, deployment/tenant context,
actual lifecycle events, demand and latency SLOs, request queues, scaling lead
times, allocation constraints, cost and incident labels. Collect these before
claiming savings or calibrated uncertainty. With more history, reserve a
separate chronological test period, compare segment forecasts, use nested
model selection and evaluate interval coverage across regimes. The honest
conclusion here is that persistence is a strong, reproducible baseline under
the chosen loss; generalization beyond this window remains unmeasured.

## Reproduction and evidence appendix

- `eda_stats.json`, `eda_summary.md`: training-only statistics and hardware buckets.
- `model_comparison.csv`: all targets, MAE/RMSE, bias, high-load error and fold stability.
- `backtest_predictions.csv`: every internal actual, forecast, origin and horizon.
- `fold_metrics.csv`, `horizon_metrics.csv`: detailed failure analysis.
- `selection.json`: model choices, rule, folds and uncertainty diagnostics.
- `forecast_uncertainty.csv`: final exploratory bounds without holdout labels.
- `audit.md`: initial audit, validation commands and implementation changes.

All report values are generated from these artifacts by `cwfc.report`. The
submission CSV is independently regenerated by `cwfc.predict`; no manual
prediction editing or public holdout scoring is performed.
'''
    (output/'final_report.md').write_text(text, encoding='utf-8')
    return text


def plot_forecast(bundle, band, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t0 = bundle.targets.timestamp.iloc[0]
    fig, ax = plt.subplots(figsize=(10,3.5))
    ax.plot((bundle.targets.timestamp-t0)/3600,bundle.targets.aggregate_cpu_utilization,label='Training actual')
    x = (band.timestamp-t0)/3600
    ax.plot(x,band.aggregate_cpu_utilization,label='Forecast')
    ax.fill_between(x,band.cpu_lower_exploratory,band.cpu_upper_exploratory,alpha=.2,label='Exploratory error band')
    ax.set(title='Final CPU forecast; no holdout labels shown',xlabel='Hours since first observation',ylabel='Mean CPU (%)')
    ax.legend(fontsize=8)
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(path,dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=Path('challenge_data'))
    parser.add_argument('--output',type=Path,default=Path('reports'))
    args = parser.parse_args()
    build(args.data,args.output)
    print(f'Regenerated EDA, backtests, bounds and {args.output / "final_report.md"}')


if __name__ == '__main__': main()
