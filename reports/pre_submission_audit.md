# Final pre-submission quality audit

27 October 2026, Asia/Calcutta. Local audit only; no remote was created or used.

## Scope and findings

Read the assignment, submission contract, completed README and final report,
all forecasting/data/EDA/report modules, and all tests. The README describes
the completed solution and commands, not the earlier Phase 1 deliverable.

The loader constructs full targets as required and discards holdout labels
before returning. Fitting, centering, lag generation, smoothing, model comparison
and uncertainty calculations receive training prefixes only. Prediction accepts
only training targets and future timestamp coordinates. Whole-trace metadata
statistics and deletion times are excluded; static categories are descriptive
only. No public holdout score was calculated.

A stronger official-data check redacted every CPU field at all final 24
timestamps in memory and changed potentially leaking metadata fields. It
confirmed identical training values **and dtypes**, comparison metrics, model
choices and final forecasts. The source data files were not modified.

Two genuine edge cases were corrected:

1. The simplicity tie-break must itself satisfy the >5% improvement gate, rather
   than only requiring that the best candidate satisfies it. Added one regression
   test, bringing the original 45 tests to **46**.
2. Empty future aggregate groups could upcast training active counts to floats.
   Explicit training count dtype restoration now prevents this label-dependent
   preprocessing effect. Extended the existing holdout-isolation regression.

These fixes do not alter any official-data numerical results or predictions.
Documentation now explains that the 5% gate is a modest maintenance-cost hurdle
with only four dependent origins, not statistical significance or a holdout-tuned
parameter. Persistence is the outright MAE minimum even without this gate.

## Executed forecasting and submission checks

All commands below ran successfully from the repository root using the existing
`.venv` interpreter. For plotting, `MPLCONFIGDIR` was set to the ignored
workspace `.mpl_cache_preaudit` directory.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m cwfc.verify_data --data challenge_data
.\.venv\Scripts\python.exe -m cwfc.backtest --data challenge_data --output reports
.\.venv\Scripts\python.exe -m cwfc.predict --data challenge_data --output predictions.csv
.\.venv\Scripts\python.exe -m cwfc.report --data challenge_data --output reports
```

Results: **46 passed**; **three of three SHA-256 checks matched**. Report
regeneration also reran training-only EDA. All commands were rerun after the
implementation fixes. Six artifacts reproduced byte-for-byte: `predictions.csv`,
`model_comparison.csv`, `backtest_predictions.csv`, `fold_metrics.csv`,
`horizon_metrics.csv`, and `selection.json`.

CPU persistence MAE is **0.2056140901 pp**, RMSE **0.2540461313 pp**. Ridge
with penalty 1 has MAE 0.2062031597 pp and RMSE 0.2413715200 pp. Thus Ridge
improves RMSE while losing on the documented primary MAE criterion. All count
candidates have zero error; normalized improvement is undefined. No holdout
information is needed for either choice.

Independent source checking read **only the timestamp column**, then validated
exact columns/order, 24 rows, timestamps 33000 through 39900 at 300-second
cadence, uniqueness, sorted order, no missing/nonfinite numbers, nonnegative
counts and CPU in [0,100]. Forecast regeneration is byte-identical. The CSV
contains 10,000 active VMs and 9.164556174985146% mean CPU at every horizon.
`submission_verification.json` contains the CSV SHA-256 and contract results.

The extra audit helpers below were executed from ignored scratch space:

```powershell
.\.venv\Scripts\python.exe .audit_work/poison_holdout.py
.\.venv\Scripts\python.exe .audit_work/run_quality.py
```

## Clean-source installation and smoke reproducibility

Exported the actual staged index into `.audit_work/checkout/` using
`git checkout-index --all --force --prefix=.audit_work/checkout/`. This is a
clean staged-source export, not a remote checkout. Created a separate fresh
Python 3.13 virtual environment at `.audit_work/venv/`; the original `.venv`
was not changed. Runtime/test dependencies and audit rendering packages were
installed there before installing the project's declared extras:

```powershell
.\.venv\Scripts\python.exe -m venv .audit_work/venv
.\.audit_work\venv\Scripts\python.exe -m pip install --disable-pip-version-check --index-url https://pypi.org/simple -e '.audit_work/checkout[dev,analysis]'
.\.venv\Scripts\python.exe .audit_work/check_clean.py
```

The clean-check helper ran the following with the audit interpreter and
`.audit_work/checkout` as its working directory:

```text
python -m pip check
python -m pytest -q
python -m cwfc.predict --data <absolute workspace>/challenge_data --output <absolute workspace>/.audit_work/clean_predictions.csv
```

`pip check`: **No broken requirements found**. Imported `cwfc.__file__` was
asserted to resolve inside the staged-source export, not the original source.
**All 46 tests passed** there, including the two-VM, 40-timestamp, four-step
synthetic CLI test and redacted-label reproduction. Actual challenge predictions
were byte-identical to the submission. No raw data was copied into the export.
Installation and tests were verified on Windows/Python 3.13 only; remote CI and
other Python/OS matrix combinations were not executed.

## Report completeness and rendered length

The final report covers all nine requested topics: executive summary; data
understanding/preprocessing; workload analysis; forecasting; temporal validation;
results versus baselines; error/failure analysis; capacity recommendation; and
limitations/future work. Forecasting and validation share one section.

Rendered the complete Markdown and its three figures into an ignored audit PDF
using ReportLab, A4, Helvetica 10.5pt body with 14pt leading, 50pt side margins,
and 40/42pt top/bottom margins. The appendix starts on a separate page.
Result: **five main pages plus one appendix page**, within the 4-6 page target.
All six rasterized pages were visually inspected: readable figures and tables,
no overlapping or clipped text. Other Markdown renderers may paginate differently.

```powershell
.\.audit_work\venv\Scripts\python.exe .audit_work/render_report.py
```

The audit-only PDF, page images, virtual environment, source export and logs are
ignored under `.audit_work/`; they are not submission artifacts or staged files.

## Git and repository hygiene

Reviewed the actual staged-file list and staged blobs, not just `.gitignore`.
The final index includes only source/tests, original challenge documentation,
README/setup docs, CI/editor recommendations, notebook, predictions, reports
and small figures. Every staged file is below 5 MB. No dataset, archive,
virtual environment, credentials, package cache, audit scratch or build files
are staged. Recognizable private-key, GitHub-token and AWS-key patterns were
checked in staged text; none were found. This is a limited pattern check, not a
claim of exhaustive secret detection.

Executed `git diff --check` and `git diff --cached --check`; both passed.
`git check-ignore` confirmed raw data, archive, environment, test scratch,
matplotlib caches, egg-info and audit scratch exclusions. Git branch is `main`;
no remote is configured. The authorized local commit uses message:

```text
Complete leakage-safe Azure workload forecasting challenge
```

Its identifier is available from `git log -1 --format=%H` and the final chat
status. Git operations used a command-scoped
`-c safe.directory=<absolute workspace>` because Windows ownership differs.
No global Git or system security settings were changed. The normal sandbox
command runner failed during setup, so an approved execution path was used.

## Remaining limitations

No unresolved submission-quality check remains. Public holdout performance is
intentionally unmeasured. Four overlapping origins, complete-trace selection,
short history and exploratory uncertainty bounds limit generalization. Remote
GitHub Actions, other Python/OS combinations and production autoscaling were
not tested. There was no remote publication.
