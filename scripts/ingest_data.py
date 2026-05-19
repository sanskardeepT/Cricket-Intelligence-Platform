"""CLI for loading cricket delivery data into PostgreSQL."""

from __future__ import annotations

import argparse

from src.data.ingest import ingest_deliveries, summarize_source


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest Cricket Intelligence deliveries into PostgreSQL")
    parser.add_argument("path", help="Path to a deliveries CSV or Cricsheet CSV zip")
    parser.add_argument("--format", default="IPL", choices=["IPL", "T20I", "ODI"], help="Cricket format tag")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print counts without writing")
    args = parser.parse_args()

    summary = summarize_source(args.path, args.format) if args.dry_run else ingest_deliveries(args.path, args.format)
    mode = "dry-run" if summary.dry_run else "loaded"
    print(
        f"{mode}: matches={summary.matches}, deliveries={summary.deliveries}, "
        f"players={summary.players}, venues={summary.venues}"
    )


if __name__ == "__main__":
    main()
