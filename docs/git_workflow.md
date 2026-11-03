# Safe Git workflow (run these yourself; nothing was committed or pushed for you)

Repository: https://github.com/samirkumarguin502-hub/aiml-01-fraud-detection
Never commit: the PaySim CSV, `.env`, `.kaggle\kaggle.json`, `.venv*`, `ml\artifacts\`, `mlruns\`, databases, `node_modules`, `frontend\dist`. All are git-ignored; verify anyway.

```
cd C:\Projects\AIML-01-Fraud-Detection
git status                                   # review every path
git status --ignored --short | Select-String "ml/data|artifacts|\.db|\.env|venv|node_modules"   # ignored items should be listed here as '!!', never as '??' or 'A'
git ls-files | Select-String "\.csv$|\.joblib$|\.db$|^\.env$|kaggle|node_modules"                 # must print nothing
git add README.md docs backend ml monitoring streaming frontend feature_repo deploy scripts tests *.txt *.ini *.yml Dockerfile .gitignore .env.example .dockerignore .github
git status                                   # confirm only intended files are staged
git diff --cached --stat
git commit -m "Pass 6: feature sets, evaluation outputs, real-data training workflow"
git remote -v                                # check the remote before pushing
git push origin main                         # only when you are satisfied
```
If you accidentally staged something sensitive: `git restore --staged <path>` before committing. If it was already committed (not pushed): `git reset --soft HEAD~1`, fix, recommit.
If a secret was ever pushed, rotate it first; deleting the file does not remove it from history.

## Model artifacts and results
- Artifacts (`ml\artifacts\<run_id>\`) stay out of Git. Keep the folder you serve from (`primary\`) on your laptop; to share it, zip that one folder and attach it to a GitHub Release (or
  use Git LFS) after checking its size. An XGBoost JSON model is typically small, but check `Get-ChildItem -Recurse ml\artifacts\<run_id> | Measure-Object Length -Sum` first.
- Results you want in the repository: copy only `report.md`, `summary.json` and `plots\` into `docs\results\<run_id>\` (aggregate statistics only; no raw rows). Review them for anything you do not want public.
- `docs\eda\dataset_stats.json` contains the first five dataset rows (simulated). Delete the `head` field or do not commit that file if you prefer.
