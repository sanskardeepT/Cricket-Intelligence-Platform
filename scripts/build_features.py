"""CLI for creating train-test feature matrices."""

from __future__ import annotations

import argparse

from src.features.matrix import write_feature_matrices


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Cricket Intelligence feature matrices")
    parser.add_argument("deliveries_csv", help="Path to cleaned deliveries CSV")
    parser.add_argument("--output-dir", default="data/features", help="Directory for X/y train-test CSVs")
    parser.add_argument("--test-fraction", type=float, default=0.2, help="Chronological holdout fraction")
    args = parser.parse_args()

    summary = write_feature_matrices(args.deliveries_csv, args.output_dir, args.test_fraction)
    print(
        f"features built: rows={summary.rows}, train={summary.train_rows}, "
        f"test={summary.test_rows}, columns={summary.features}, output={summary.output_dir}"
    )


if __name__ == "__main__":
    main()
