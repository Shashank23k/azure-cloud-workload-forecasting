"""Reproducible *training-only* Azure VM workload exploration.

Usage:
    python -m cwfc.eda --data challenge_data --output reports

Only the first 110 timestamps' CPU observations may enter summaries/plots.
Source timestamps are relative seconds, not real wall-clock datetimes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import TrainingBundle, load_training_bundle


def compute_eda(bundle: TrainingBundle) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Return summary statistics and two segment views (static metadata only)."""
    frame = bundle.readings
    targets = bundle.targets
    cpu = targets.aggregate_cpu_utilization
    joined = frame.merge(bundle.metadata, on="vm_id", how="left", validate="many_to_one")
    if joined.vm_category.isna().any():
        raise ValueError("Missing metadata categories after join")

    category = (
        joined.groupby("vm_category", observed=True)
        .agg(
            vm_count=("vm_id", "nunique"),
            mean_cpu=("avg_cpu", "mean"),
            median_cpu=("avg_cpu", "median"),
            p90_cpu=("avg_cpu", lambda x: x.quantile(.9)),
            high_cpu_share=("avg_cpu", lambda x: (x >= 80).mean()),
        )
        .sort_values("vm_count", ascending=False)
    )
    category["vm_fraction"] = category.vm_count / category.vm_count.sum()

    by_core = (
        joined.groupby("virtual_core_bucket", observed=True)
        .agg(vm_count=("vm_id", "nunique"), mean_cpu=("avg_cpu", "mean"))
        .sort_values("vm_count", ascending=False)
    )
    by_memory = joined.groupby('memory_gb_bucket', observed=True).agg(
        vm_count=('vm_id', 'nunique'), mean_cpu=('avg_cpu', 'mean'))
    populations = [set(part.vm_id) for _, part in frame.groupby('timestamp', sort=True)]
    entries = [len(b-a) for a, b in zip(populations, populations[1:])]
    exits = [len(a-b) for a, b in zip(populations, populations[1:])]

    vms_per_ts = frame.groupby("timestamp", sort=True).vm_id.nunique()
    train_vm_coverage = frame.groupby("vm_id", observed=True).timestamp.nunique()
    vm_avg = frame.groupby("vm_id", observed=True).avg_cpu.mean()
    by_time = joined.groupby(["timestamp", "vm_category"], observed=True).avg_cpu.mean().unstack()
    # Count high-CPU VMs without conflating five-minute mean with max-in-interval.
    high_by_time = frame.groupby("timestamp").agg(
        high_avg_share=("avg_cpu", lambda x: (x >= 80).mean()),
        high_max_share=("max_cpu", lambda x: (x >= 80).mean()),
    )
    out = {
        **bundle.audit,
        "training_target_statistics": {
            "active_vm_min": int(vms_per_ts.min()),
            "active_vm_max": int(vms_per_ts.max()),
            "active_vm_unique_values": int(vms_per_ts.nunique()),
            "aggregate_cpu_mean": float(cpu.mean()),
            "aggregate_cpu_std": float(cpu.std()),
            "aggregate_cpu_min": float(cpu.min()),
            "aggregate_cpu_max": float(cpu.max()),
            "aggregate_cpu_p95": float(cpu.quantile(.95)),
            "aggregate_cpu_peak_timestamp": int(targets.loc[cpu.idxmax(), "timestamp"]),
            "aggregate_cpu_low_timestamp": int(targets.loc[cpu.idxmin(), "timestamp"]),
            "aggregate_cpu_lag1_autocorrelation": float(cpu.autocorr(lag=1)) if len(cpu) > 2 else None,
        },
        "training_vm_level_statistics": {
            "avg_cpu_median": float(frame.avg_cpu.median()),
            "avg_cpu_p90": float(frame.avg_cpu.quantile(.90)),
            "avg_cpu_p95": float(frame.avg_cpu.quantile(.95)),
            "avg_cpu_p99": float(frame.avg_cpu.quantile(.99)),
            "fraction_below_5pct_avg_cpu": float((frame.avg_cpu < 5).mean()),
            "fraction_at_least_80pct_avg_cpu": float((frame.avg_cpu >= 80).mean()),
            "fraction_at_least_80pct_max_cpu": float((frame.max_cpu >= 80).mean()),
            "mean_high_avg_share_by_timestamp": float(high_by_time.high_avg_share.mean()),
            "max_high_avg_share_by_timestamp": float(high_by_time.high_avg_share.max()),
            "vm_mean_cpu_p95": float(vm_avg.quantile(.95)),
            "min_observations_per_training_vm": int(train_vm_coverage.min()),
            "max_observations_per_training_vm": int(train_vm_coverage.max()),
        },
        "categories": {
            str(idx): {
                "vms": int(r.vm_count),
                "vm_share": float(r.vm_fraction),
                "mean_cpu": float(r.mean_cpu),
                "median_cpu": float(r.median_cpu),
                "p90_cpu": float(r.p90_cpu),
                "share_vm_readings_avg_cpu_ge_80": float(r.high_cpu_share),
            }
            for idx, r in category.iterrows()
        },
        "memory_buckets": {str(k): {'vms': int(r.vm_count), 'mean_cpu': float(r.mean_cpu)}
                           for k, r in by_memory.iterrows()},
        "observed_activity": {'entries_after_first_timestamp': sum(entries),
                              'exits_before_last_timestamp': sum(exits),
                              'note': 'Telemetry appearances/disappearances, not lifecycle creation/deletion'},
        "time_handling": "Relative seconds from source trace; not UTC or local datetimes",
        "leakage_policy": (
            "Descriptive CPU analysis uses only first training timestamps; "
            "future timestamp coordinates are used only to construct the forecast template; "
            "metadata whole-trace statistics/deletion times are excluded"
        ),
    }
    return out, by_time, by_core


def create_figures(bundle: TrainingBundle, by_category_time: pd.DataFrame, output_dir: Path) -> None:
    """Write reviewer-friendly PNGs with no exposure to the holdout target."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    frame = bundle.readings
    target = bundle.targets
    elapsed = (target.timestamp - target.timestamp.iloc[0]) / 3600
    plt.rcParams.update({"font.size": 10, "figure.dpi": 130, "savefig.dpi": 160})

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(elapsed, target.aggregate_cpu_utilization, linewidth=1.6, label="Observed training CPU")
    ax.axhline(target.aggregate_cpu_utilization.mean(), linestyle="--", linewidth=1.3,
               label="Training mean")
    ax.set(title="Fleet mean CPU utilization — training window only",
           xlabel="Hours since first observation", ylabel="Mean CPU (%)")
    ax.grid(alpha=.22)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "01_aggregate_cpu_train.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(frame.avg_cpu, bins=np.linspace(0, 100, 101), log=True,
            edgecolor="none", label="VM five-minute mean CPU")
    ax.axvline(frame.avg_cpu.median(), linewidth=1.5, linestyle="--", label="VM observation median")
    ax.axvline(frame.avg_cpu.quantile(.95), linewidth=1.5, linestyle=":", label="VM observation P95")
    ax.set(title="VM-level CPU utilization is highly skewed — train only",
           xlabel="Five-minute average CPU utilization (%)", ylabel="Number of observations (log scale)")
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "02_vm_cpu_distribution.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    x = (by_category_time.index.to_numpy() - target.timestamp.iloc[0]) / 3600
    for category in by_category_time.columns:
        ax.plot(x, by_category_time[category], linewidth=1.45, label=str(category))
    ax.set(title="Mean CPU utilization by VM workload category — train only",
           xlabel="Hours since first observation", ylabel="Category mean CPU (%)")
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "03_cpu_by_category.png")
    plt.close(fig)

    high_by_time = frame.groupby("timestamp", sort=True).agg(
        high_avg_share=("avg_cpu", lambda x: (x >= 80).mean()),
        high_max_share=("max_cpu", lambda x: (x >= 80).mean()),
    )
    x = (high_by_time.index.to_numpy() - target.timestamp.iloc[0]) / 3600
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, high_by_time.high_avg_share.to_numpy() * 100,
            label="Five-minute average CPU ≥ 80%", linewidth=1.5)
    ax.plot(x, high_by_time.high_max_share.to_numpy() * 100,
            label="Five-minute maximum CPU ≥ 80%", linewidth=1.5)
    ax.set(title="Hot VM fraction: aggregate mean conceals localized risk",
           xlabel="Hours since first observation", ylabel="Share of VMs per timestamp (%)")
    ax.legend(fontsize=8)
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(output_dir / "04_hot_vm_fraction.png")
    plt.close(fig)

    vm_mean = frame.groupby("vm_id", observed=True).avg_cpu.mean()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(vm_mean, bins=np.arange(0, 101, 2))
    ax.set(title="Distribution of per-VM mean CPU utilization — train only",
           xlabel="Mean CPU across training observations (%)", ylabel="Number of VMs")
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(output_dir / "05_vm_mean_load.png")
    plt.close(fig)


def render_summary(stats: dict, by_core: pd.DataFrame) -> str:
    a = stats["training_target_statistics"]
    b = stats["training_vm_level_statistics"]
    categories = stats["categories"]
    table = "\n".join(
        f"| {label} | {info['vms']:,} | {info['mean_cpu']:.2f}% | "
        f"{info['p90_cpu']:.2f}% | {info['share_vm_readings_avg_cpu_ge_80']*100:.2f}% |"
        for label, info in categories.items()
    )
    core_rows = "\n".join(
        f"| {idx} | {int(row.vm_count):,} | {row.mean_cpu:.2f}% |"
        for idx, row in by_core.iterrows()
    )
    return f"""# EDA: Azure cloud workload, training data only

> Reproduce with `python -m cwfc.eda --data challenge_data --output reports`.
> This is an exploration report, **not** the required 4–6 page final modeling report.

## Scope and validation

- Full source file: {stats['total_rows']:,} CPU observations across {stats['unique_vms']:,} VMs and {stats['all_timestamp_count']} timestamps; {stats['metadata_rows']:,} metadata rows.
- **Training-only CPU analysis**: {stats['training_row_count']:,} observations at the first {stats['training_timestamp_count']} timestamps. The final {stats['forecast_timestamp_count']} timestamp *coordinates* are reserved for the forecasting template; their CPU measurements are excluded from all plots, summaries, and feature construction.
- Timestamp increment: {stats['cadence_seconds']} seconds. Source timestamps are **relative seconds**, not real dates; avoid fabricated hour-of-day or weekday features.
- Training missing measurements: {stats['training_missing_measurements']}; training duplicate `(timestamp, vm_id)` pairs: {stats['training_duplicate_vm_timestamp_pairs']}.
- Per-VM training coverage: {b['min_observations_per_training_vm']}–{b['max_observations_per_training_vm']} timestamps. CPU measurements satisfy `[0, 100]` and min ≤ avg ≤ max.

## Aggregate targets

- Active VM count is **{a['active_vm_min']:,}–{a['active_vm_max']:,}** in the training window ({a['active_vm_unique_values']} unique value). This is explained by the dataset's deliberate selection of complete VM traces, **not evidence of zero churn in Azure generally**.
- Fleet average CPU: **{a['aggregate_cpu_mean']:.4f}%**, standard deviation {a['aggregate_cpu_std']:.4f} percentage points; range **{a['aggregate_cpu_min']:.4f}%–{a['aggregate_cpu_max']:.4f}%**.
- Peak aggregate CPU in the training window: {a['aggregate_cpu_max']:.4f}% at relative timestamp {a['aggregate_cpu_peak_timestamp']} seconds; minimum {a['aggregate_cpu_min']:.4f}% at {a['aggregate_cpu_low_timestamp']} seconds.
- Lag-1 autocorrelation: {a['aggregate_cpu_lag1_autocorrelation']:.3f}. Short-lag persistence may help a statistical baseline, but one ~9-hour training window cannot demonstrate daily/weekly seasonality.

![Training aggregate CPU](figures/01_aggregate_cpu_train.png)

## VM-level heterogeneity and localized demand

- Median VM observation CPU: **{b['avg_cpu_median']:.2f}%**, while the P95 and P99 are **{b['avg_cpu_p95']:.2f}%** and **{b['avg_cpu_p99']:.2f}%**.
- **{b['fraction_below_5pct_avg_cpu']*100:.2f}%** of VM observations have average CPU below 5%, yet **{b['fraction_at_least_80pct_avg_cpu']*100:.2f}%** have five-minute average CPU at least 80%.
- **{b['fraction_at_least_80pct_max_cpu']*100:.2f}%** of observations have an intra-interval maximum at least 80%. Mean and maximum measure different risks and must not be equated.
- The aggregate mean cannot guarantee low resource pressure for every VM, host, deployment or tenant.

![VM CPU distribution](figures/02_vm_cpu_distribution.png)

![VM mean CPU distribution](figures/05_vm_mean_load.png)

![Fraction of high-CPU VMs](figures/04_hot_vm_fraction.png)

## Workload segments (training readings; static VM categories)

| VM category | VMs | Mean CPU | P90 CPU | Share of readings avg ≥80% |
|---|---:|---:|---:|---:|
{table}

![Category CPU time series](figures/03_cpu_by_category.png)

Core buckets are labels, **not exact provisioned core counts**; using these buckets directly as exact capacity weights would be unjustified.

| Virtual-core bucket | VMs | Mean CPU |
|---|---:|---:|
{core_rows}

Memory buckets (labels, not exact allocated GB):

| Memory bucket | VMs | Mean CPU |
|---|---:|---:|
{chr(10).join(f"| {k} | {v['vms']:,} | {v['mean_cpu']:.2f}% |" for k,v in stats['memory_buckets'].items())}

Observed training telemetry entries/exits after/before boundary timestamps:
{stats['observed_activity']['entries_after_first_timestamp']} /
{stats['observed_activity']['exits_before_last_timestamp']}. These are telemetry
transitions, not inferred lifecycle events.

## Operational interpretation

1. Use a naive/short-history statistical baseline as the initial forecasting reference: aggregate load is smooth relative to VM-level load, and only 110 training steps are available.
2. Active VM count forecasting can start with persistence, but fixed VM count is a **sampling-design artifact**. The data cannot support a strong churn forecast for a live cloud fleet.
3. Evaluate tail risk independently of the mean: a steady 9% fleet average can coincide with a small but important set of VMs at ≥80% CPU.
4. Any scaling/headroom policy will need a buffer calibrated using **rolling-origin errors**, not a constant margin asserted from this EDA alone.

## Explicit limitations and leakage controls

- Training EDA excludes observed final-holdout CPU values; the unseen holdout target must not drive feature selection, training, or model comparison.
- Metadata fields `avg_cpu`, `max_cpu`, `p95_max_cpu`, and `timestamp_vm_deleted` are intentionally **not used**: they can encode information from beyond the forecast origin.
- VM population is a curated, stratified subset of long-enough complete traces; neither churn nor true fleet-wide rates are representative of all Azure VMs.
- Mean VM utilization is an **unweighted percentage**. It is not total CPU-core demand or a rigorous host capacity measure; virtual-core buckets are not exact core counts.
- The time grid is relative seconds. **No claims of daily/weekly patterns** are defensible from this short window alone.

## Modeling results

See `final_report.md` for matched 24-step expanding-window backtests, model selection, residual diagnostics and operational recommendations.
"""


def write_eda(bundle: TrainingBundle, output_dir: str | Path) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stats, by_category_time, by_core = compute_eda(bundle)
    create_figures(bundle, by_category_time, output_dir / "figures")
    (output_dir / "eda_stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    (output_dir / "eda_summary.md").write_text(render_summary(stats, by_core), encoding="utf-8")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Run leakage-safe training EDA")
    parser.add_argument("--data", type=Path, default=Path("challenge_data"))
    parser.add_argument("--output", type=Path, default=Path("reports"))
    parser.add_argument("--train-steps", type=int, default=110)
    parser.add_argument("--horizon", type=int, default=24)
    args = parser.parse_args()
    bundle = load_training_bundle(args.data, train_steps=args.train_steps, horizon=args.horizon)
    stats = write_eda(bundle, args.output)
    print(f"EDA completed: {args.output / 'eda_summary.md'}")
    print(f"Training mean CPU: {stats['training_target_statistics']['aggregate_cpu_mean']:.4f}%")
    print(f"Active VM count range: "
          f"{stats['training_target_statistics']['active_vm_min']}–"
          f"{stats['training_target_statistics']['active_vm_max']}")
    print("All EDA measurements are restricted to the training portion.")


if __name__ == "__main__":
    main()
