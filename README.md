# Azure Cloud Workload Forecasting

Atomity ML take-home assessment using the **Microsoft Azure Public Dataset V2**, curated `data-v1` release.

Forecast the next **24 five-minute timestamps (two hours)** for the number of VMs with valid telemetry and the unweighted mean CPU utilization. The implementation includes training-only EDA, nine forecasting candidates, matched rolling-origin evaluation, error analysis, exploratory uncertainty bounds, tests, and a reproducible prediction CLI.

## Results

| Target / selected model | Internal MAE | Internal RMSE |
|---|---:|---:|
| Active VM count / persistence | 0 VMs | 0 VMs |
| Aggregate CPU / persistence | 0.2056 percentage points | 0.2540 percentage points |
| Aggregate CPU / recursive Ridge, penalty 1 | 0.2062 percentage points | 0.2414 percentage points |

Persistence is selected separately for both targets. Ridge improves CPU RMSE but does not improve the primary MAE loss. No alternative clears the requirement of more than 5% MAE improvement. Count errors are zero because the release deliberately selects complete traces; relative improvement against that zero-error baseline is undefined. These are **internal model-development validation results**, with overlapping folds, rather than public holdout test scores.

The 5% gate is a modest maintenance-cost hurdle given four dependent validation origins, not a significance test or a holdout-tuned parameter. Persistence is the outright CPU MAE minimum even without this gate.

The generated [predictions.csv](predictions.csv) has 24 rows: 10,000 active VMs and 9.1646% CPU at every timestamp, derived from the last permitted observations. No forecast values are manually edited. Read the [final technical report](reports/final_report.md), [EDA](reports/eda_summary.md), [comparison table](reports/model_comparison.csv), and [audit](reports/audit.md).

![Internal validation forecasts](reports/figures/06_validation_forecasts.png)

## Installation

Python **3.11+**. Execution was verified with the existing Python **3.13.7** environment, NumPy 2.5.4, pandas 3.1.0, and matplotlib 3.11.2. Runtime forecasting needs only NumPy and pandas; report figures require the analysis extra.

For a new checkout, Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,analysis]"
```

Reuse an existing environment when available. Activation is optional; direct interpreter paths avoid PowerShell activation-policy issues. On Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,analysis]"
```

## Dataset

Obtain the fixed [Atomity data-v1 release](https://github.com/atomityhq/forecasting-challenge/releases/tag/data-v1). The challenge is derived from [Microsoft Azure Public Dataset V2](https://github.com/Azure/AzurePublicDataset); preserve this attribution. No raw data is included in Git.

With the supplied archive in the repository root:

```powershell
New-Item -ItemType Directory -Force challenge_data
tar -xzf data-v1.tar.gz -C challenge_data
```

Expected files directly under `challenge_data/`:

| File | Schema / purpose |
|---|---|
| `vm_cpu_readings.csv.gz` | `timestamp,vm_id,min_cpu,max_cpu,avg_cpu` |
| `vm_metadata.csv.gz` | VM ID; subscription/deployment IDs; creation/deletion times; source CPU summaries; category; core/memory buckets |
| `DATA_CARD.md` | Curated sample design and provenance |
| `SHA256SUMS` | Three-file integrity manifest |

See [DATASET.md](DATASET.md) for exact fields and [SUBMISSION.md](SUBMISSION.md) for the output contract. Source timestamps are **relative seconds**, not verified datetimes.

## Reproduce the submission

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe -m cwfc.verify_data --data challenge_data
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m cwfc.predict --data challenge_data --output predictions.csv
```

With the environment activated, the portable prediction command is:

```bash
python -m cwfc.predict --data challenge_data --output predictions.csv
```

The CLI loads and validates the files, constructs targets, removes holdout labels, compares candidates within training, fits the selected methods and validates the written CSV. Required schema:

```text
timestamp,active_vm_count,aggregate_cpu_utilization
```

Output timestamps exactly match the final 24 source timestamps in sorted order. Predictions are finite, counts are nonnegative and CPU is bounded to [0,100]. No index column is written. Forecast counts may be fractional for alternative methods; the contract permits nonnegative numeric values.

For alternate datasets, the CLI uses all timestamps except the final horizon as training. `--horizon 4` supports small smoke tests. At least `8 + horizon` training observations, hence `8 + 2*horizon` total timestamps, are needed for model comparison. The horizon defaults to 24; VM population size is unrestricted. Cadence must remain 300 seconds. Too-short or malformed files produce validation errors.

Regenerate all analysis and the numerical report:

```powershell
$env:MPLCONFIGDIR = Join-Path (Get-Location) '.mpl_cache'
.\.venv\Scripts\python.exe -m cwfc.report --data challenge_data --output reports
```

Individual commands are also available:

```bash
python -m cwfc.eda --data challenge_data --output reports
python -m cwfc.backtest --data challenge_data --output reports
```

`cwfc.report` regenerates EDA, backtests, selection, bounds and Markdown; `cwfc.predict` writes the submission independently. The notebook is a companion to the script-based analysis. Pytest creates a fresh ignored `.test_tmp_*/` directory per run and disables its cache to avoid Windows account ownership conflicts. Scratch directories may be removed after testing when accessible.

## Methodology and leakage prevention

The official dataset has 10,000 VMs, 134 timestamps and 1,340,000 readings. The first 110 timestamps are training; final 24 target labels are discarded inside the loader after full target construction. Downstream forecasting receives training targets and future timestamps only. No public holdout scores are computed.

- Active count uses distinct VM IDs with non-null average CPU at each timestamp. CPU is the unweighted mean of those readings.
- Duplicate VM/timestamp pairs are rejected. Missing averages and absent VM readings are excluded, not imputed as idle. Entire training-timestamp gaps or invalid training CPU raise errors.
- Whole-trace metadata CPU statistics and deletion times are excluded. Static category/core/memory labels are used only for EDA; models use only past aggregate targets.
- Internal origins are 50, 62, 74 and 86 training observations, each predicting a genuine 24-step future trajectory. No future actuals are fed into recursive forecasts.
- Fixed candidates: persistence, full mean, 6/24-step trailing averages, SES alpha .2/.6, damped Holt and six-lag recursive Ridge with penalties 1/10. No unsupported calendar seasonality or deep-learning complexity is introduced.
- Selection requires >5% MAE improvement over persistence; favor the simplest candidate within 1% of the best MAE. The full rule and parameters are documented in source and report.

## Workload findings and limits

Training-only mean fleet CPU is **9.4141%**, ranging **9.1116%-9.9122%**. All training timestamps contain 10,000 VMs, with zero duplicate/missing measurements. VM-reading median CPU is **5.12%**, while **1.27%** of readings average at least 80% CPU and **2.95%** have an interval maximum at least 80%. Static categories and hardware bucket summaries are in the EDA report.

Low average CPU can hide individual VM saturation. Complete-trace sampling suppresses churn, and roughly nine hours of training cannot demonstrate daily cycles or generalize to a production fleet. An unweighted VM percentage cannot establish physical core demand. Residual-based bounds are exploratory, with overlapping folds and too few origins for calibrated coverage. Production scaling requires true capacity/placement, latency SLOs, queues, lifecycle events and longer representative history.

## Repository layout

```text
src/cwfc/data.py          Contract validation; target construction; leakage boundary
src/cwfc/models.py        Deterministic NumPy forecasting candidates
src/cwfc/baselines.py     Preserved starter baseline API
src/cwfc/validation.py    Preserved temporal split API (relative timestamps retained)
src/cwfc/backtest.py      Matched folds; metrics; selection; diagnostic figures
src/cwfc/predict.py       Submission CLI and output contract checks
src/cwfc/eda.py           Training-only workload and segment analysis
src/cwfc/report.py        Reproducible evidence-based report and forecast bounds
src/cwfc/verify_data.py   Portable SHA-256 checker
reports/                 Final report, EDA, CSV/JSON evidence and eight figures
notebooks/               Companion training-only EDA notebook
tests/                   Unit, leakage-isolation and synthetic CLI integration tests
predictions.csv          Required 24-row submission
.github/workflows/       Dataset-free CI for Python 3.11-3.13
```

## Submission readiness

All 46 local tests pass (the original 45 plus a selection-gate regression test), three checksums match, and predictions were regenerated and checked against the official timestamp and value contract. Tests include CLI reproduction using two synthetic VMs and changed holdout labels. CI is configured; remote CI has not been run in this session. See the [pre-submission audit](reports/pre_submission_audit.md) for final installation, rendering and Git checks.

Commit source, tests, documentation, notebook, small analysis artifacts and `predictions.csv`. `.gitignore` excludes raw data/archive, environment, caches, package/build artifacts and common local secret files. No remote repository is created or pushed by the implementation. Review the [assignment](ASSIGNMENT.md) and [submission contract](SUBMISSION.md) before publishing. The existing MIT [license](LICENSE) is preserved.
