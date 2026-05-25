"""CLI for training the player runs model."""

from __future__ import annotations

import argparse

from src.models.player_runs import train_player_runs_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Cricket Intelligence batter runs model")
    parser.add_argument("deliveries_csv", help="Path to cleaned deliveries CSV")
    parser.add_argument("--artifact-dir", default="artifacts/models", help="Directory for model artifacts")
    parser.add_argument("--test-fraction", type=float, default=0.2, help="Chronological holdout fraction")
    parser.add_argument("--folds", type=int, default=4, help="TimeSeriesSplit folds")
    args = parser.parse_args()

    summary = train_player_runs_model(
        args.deliveries_csv,
        artifact_dir=args.artifact_dir,
        test_fraction=args.test_fraction,
        folds=args.folds,
    )
    print(
        f"trained player runs model: best={summary.best_model}, train_rows={summary.train_rows}, "
        f"test_rows={summary.test_rows}, features={summary.features}, "
        f"mae={summary.test_mae:.3f}, rmse={summary.test_rmse:.3f}"
    )
    print(f"artifact={summary.artifact_path}")
    print(f"metrics={summary.metrics_path}")


if __name__ == "__main__":
    main()
