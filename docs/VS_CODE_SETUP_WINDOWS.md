# Windows / VS Code reproduction

The complete forecasting workflow is documented in [README.md](../README.md).
Use the existing `.venv` when present. For a clean checkout only:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,analysis]"
```

Select `.venv\Scripts\python.exe` as the VS Code Python interpreter. No activation
or execution-policy change is necessary. Obtain the fixed Atomity `data-v1`
archive and extract its files directly into `challenge_data/` as described in
the README; raw data must stay outside Git.

From the project root:

```powershell
.\.venv\Scripts\python.exe -m cwfc.verify_data --data challenge_data
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m cwfc.predict --data challenge_data --output predictions.csv
$env:MPLCONFIGDIR = Join-Path (Get-Location) '.mpl_cache'
.\.venv\Scripts\python.exe -m cwfc.report --data challenge_data --output reports
```

Inspect `reports/final_report.md` and `predictions.csv`. The EDA notebook can be
opened in VS Code using the same kernel. See `reports/audit.md` for verification
results and `reports/interview_preparation.md` for discussion points.

Before making a local commit, inspect `git status` and `git diff --cached --stat`.
Keep `.venv/`, `challenge_data/`, archives and credentials untracked. Remote
creation and pushing are separate actions for the repository owner.
