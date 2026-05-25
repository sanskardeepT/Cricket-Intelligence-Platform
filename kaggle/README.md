# Cricket Intelligence Platform IPL Artifacts

This dataset package contains generated local artifacts for the Cricket Intelligence Platform:

- Cricsheet IPL raw CSV zip
- cleaned IPL deliveries CSV
- train/test feature matrices
- trained model artifacts
- JSON metric summaries

The repository intentionally does not track these large generated files. Build or download them locally, then upload this Kaggle package.

## Publish

1. Create a Kaggle API token from your Kaggle account settings.
2. Save it at `%USERPROFILE%\.kaggle\kaggle.json`.
3. Run from the repo root:

```powershell
python scripts\prepare_kaggle_dataset.py
kaggle datasets create -p kaggle\cricket-intelligence-platform-ipl-artifacts
```

For updates after the dataset already exists:

```powershell
python scripts\prepare_kaggle_dataset.py
kaggle datasets version -p kaggle\cricket-intelligence-platform-ipl-artifacts -m "Update Cricket Intelligence artifacts"
```
