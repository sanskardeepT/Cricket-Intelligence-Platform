"""Prepare generated data/model artifacts for Kaggle upload."""

from __future__ import annotations

import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "kaggle" / "cricket-intelligence-platform-ipl-artifacts"
METADATA = ROOT / "kaggle" / "dataset-metadata.json"

FILES = [
    ROOT / "data" / "raw" / "ipl_csv2.zip",
    ROOT / "data" / "processed" / "ipl_deliveries.csv",
    ROOT / "data" / "features" / "X_train.csv",
    ROOT / "data" / "features" / "X_test.csv",
    ROOT / "data" / "features" / "y_train.csv",
    ROOT / "data" / "features" / "y_test.csv",
    ROOT / "data" / "features" / "metadata.csv",
    ROOT / "data" / "features" / "feature_metadata.json",
    ROOT / "artifacts" / "models" / "win_probability_baseline.joblib",
    ROOT / "artifacts" / "models" / "training_metrics.json",
    ROOT / "artifacts" / "models" / "next_ball_outcome.joblib",
    ROOT / "artifacts" / "models" / "next_ball_metrics.json",
    ROOT / "artifacts" / "models" / "toss_decision.joblib",
    ROOT / "artifacts" / "models" / "toss_metrics.json",
    ROOT / "artifacts" / "models" / "player_runs.joblib",
    ROOT / "artifacts" / "models" / "player_runs_metrics.json",
]


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if METADATA.exists():
        shutil.copy2(METADATA, OUTPUT / "dataset-metadata.json")
    copied: list[dict[str, object]] = []
    missing: list[str] = []
    for source in FILES:
        if not source.exists():
            missing.append(str(source.relative_to(ROOT)))
            continue
        relative = source.relative_to(ROOT)
        target = OUTPUT / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied.append({"path": str(relative).replace("\\", "/"), "bytes": source.stat().st_size})
    manifest = {"copied": copied, "missing": missing}
    (OUTPUT / "artifact-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"prepared Kaggle dataset folder: {OUTPUT}")
    print(f"copied={len(copied)}, missing={len(missing)}")
    if missing:
        print("missing files:")
        for item in missing:
            print(f"- {item}")


if __name__ == "__main__":
    main()
