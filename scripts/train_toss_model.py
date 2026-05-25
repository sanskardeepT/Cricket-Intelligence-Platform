"""CLI for training the toss-decision model."""

from __future__ import annotations

import argparse

from src.models.toss_model import train_toss_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Cricket Intelligence toss-decision model")
    parser.add_argument("cricsheet_zip", help="Path to Cricsheet CSV zip with *_info.csv files")
    parser.add_argument("deliveries_csv", help="Path to cleaned deliveries CSV")
    parser.add_argument("--artifact-dir", default="artifacts/models", help="Directory for model artifacts")
    parser.add_argument("--test-fraction", type=float, default=0.2, help="Chronological holdout fraction")
    parser.add_argument("--folds", type=int, default=3, help="TimeSeriesSplit folds")
    args = parser.parse_args()

    summary = train_toss_model(
        args.cricsheet_zip,
        args.deliveries_csv,
        artifact_dir=args.artifact_dir,
        test_fraction=args.test_fraction,
        folds=args.folds,
    )
    print(
        f"trained toss model: best={summary.best_model}, train_rows={summary.train_rows}, "
        f"test_rows={summary.test_rows}, features={summary.features}, "
        f"accuracy={summary.test_accuracy:.4f}, log_loss={summary.test_log_loss:.4f}"
    )
    print(f"artifact={summary.artifact_path}")
    print(f"metrics={summary.metrics_path}")


if __name__ == "__main__":
    main()
