"""CLI for training baseline win-probability models."""

from __future__ import annotations

import argparse

from src.models.training import train_baselines


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Cricket Intelligence win-probability models")
    parser.add_argument("--feature-dir", default="data/features", help="Directory containing X/y train-test CSVs")
    parser.add_argument("--artifact-dir", default="artifacts/models", help="Directory for model artifact and metrics")
    parser.add_argument("--folds", type=int, default=5, help="TimeSeriesSplit folds")
    args = parser.parse_args()

    summary = train_baselines(args.feature_dir, args.artifact_dir, args.folds)
    print(
        f"trained: best={summary.best_model}, train_rows={summary.train_rows}, "
        f"test_rows={summary.test_rows}, features={summary.features}"
    )
    print(f"artifact={summary.artifact_path}")
    print(f"metrics={summary.metrics_path}")


if __name__ == "__main__":
    main()
