# Implementation audit and verification

> Historical implementation record. See [pre_submission_audit.md](pre_submission_audit.md)
> for the final quality gate and local Git preparation results.

Date: 27 October 2026 (Asia/Calcutta).

## Initial audit

Read `ASSIGNMENT.md`, `DATASET.md`, `SUBMISSION.md`, `README.md`, `pyproject.toml`,
all existing `src/cwfc/` modules and six tests, the EDA notebook, previous EDA
statistics/report, and the supplied `DATA_CARD.md` before substantial edits.
No Git repository was initially present. Existing source-integrity checking,
train-only bundle boundary, static metadata filtering and EDA were preserved.
No dataset download, environment replacement or external data upload occurred.

The literal initial `python -m pytest -q` run passed two tests but encountered
four fixture errors because Windows denied access to the preexisting system
temporary pytest directory. A workspace-local basetemp/cache run passed all six
original tests. Pytest now creates a fresh ignored local scratch directory per run and disables
its cache, so the requested ordinary command works across account contexts.

## Changes and review decisions

- Corrected active target construction to count only non-null average readings.
  Absent rows or null averages are excluded from both target aggregates; entire
  missing training timestamps and invalid remaining measurements fail clearly.
- Enforced 300-second grid, finite relative timestamps and adaptive CLI split.
- Preserved original baseline and rolling-split APIs. Fixed relative timestamp
  conversion and nonpositive step handling in the starter split helper.
- Changed normalized MAE with zero baseline error to undefined (NaN).
- Added fixed model candidates, matched multi-step backtests, explicit selection,
  detailed residual/peak/fold/horizon artifacts, and prediction contract checks.
- Training-only EDA now includes memory buckets and telemetry transition counts.
- Added reproducible report generation and exploratory uncertainty diagnostics;
  pooled bounds are not described as formally calibrated prediction intervals.
- Added forecast, constant-target, malformed-input, missing-observation, schema,
  deterministic regeneration, smaller synthetic CLI and holdout-poisoning tests.
- Updated README and Windows guide to describe the completed workflow.

Potential leakage review: the loader constructs full aggregate targets as the
assignment specifies, but returns no held-out label vector. Model fitting,
centering, recursive lags, selection and residual analysis operate exclusively
on training prefixes. Only public holdout timestamp coordinates reach the final
predictor. Whole-trace metadata summaries and deletion times are never model
inputs. Structural file checks necessarily inspect identifiers and timestamps;
no holdout CPU summaries are logged, plotted or scored. Training diagnostics
exclude the final holdout. Complete-trace bias is explicitly documented.

## Executed checks

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m cwfc.verify_data --data challenge_data
.\.venv\Scripts\python.exe -m cwfc.eda --data challenge_data --output reports
.\.venv\Scripts\python.exe -m cwfc.backtest --data challenge_data --output reports
.\.venv\Scripts\python.exe -m cwfc.predict --data challenge_data --output predictions.csv
$env:MPLCONFIGDIR = Join-Path (Get-Location) '.mpl_cache'
.\.venv\Scripts\python.exe -m cwfc.report --data challenge_data --output reports
```

Results: **45 tests passed**; all three manifest entries match. Nine candidates
were evaluated on four identical folds per target. The prediction CLI generated
24 rows and validated the written CSV; a separate source-timestamp-only check
confirms exact matching timestamps, schema, finite values and bounds. Repeated
prediction generation is byte-identical. A CLI integration test uses two VMs,
40 timestamps and a four-step horizon, then poisons held-out labels and metadata
and confirms identical forecasts and model choices. All tests are dataset-free.

Verified environment: Python 3.13.7, NumPy 2.5.4, pandas 3.1.0, matplotlib 3.11.2.
No clean-environment installation or remote GitHub Actions execution is claimed.
An existing matplotlib user-cache permission warning was resolved by directing
its cache to ignored workspace `.mpl_cache/` for report reproduction.

## Limitations

The four overlapping origins provide 96 forecast instances but 60 unique
validation timestamps. Model-development results are not an independent test
estimate. The report is delivered in Markdown; its final pagination depends on
the reviewer's renderer. Public holdout performance is deliberately unmeasured.
No production autoscaling, remote publishing or cost-savings claim is made.

## Local Git and final review

Git is initialized on `main`; there are no commits and nothing has been staged.
`git status --short` was reviewed, and `git check-ignore` confirmed exclusion of
the virtual environment, raw CPU data, archive, pytest scratch, matplotlib cache
and generated egg-info. Small source/report artifacts and predictions remain
ready to commit. No remote or global Git configuration was changed.

Git detected different Windows ownership; inspection used a command-scoped
`-c safe.directory=<this workspace>` setting. After initialization, the normal
sandbox command runner failed during setup; an approved elevated execution path
completed final verification. A reused sandbox-owned pytest directory initially
failed from the other account; fresh per-run scratch directories resolved it.
The final literal `python -m pytest -q` run passed all 45 tests. These environment
failures did not indicate forecasting test failures.

`submission_verification.json` records the independent timestamp-only source
check and SHA-256 of the byte-identically regenerated submission. No holdout
CPU labels were read for that independent check. The companion EDA notebook was
retained from the initial project; scripts regenerate current authoritative
statistics and figures, without requiring notebook execution dependencies.
