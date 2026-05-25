"""Download and restore Kaggle-hosted Cricket Intelligence artifacts."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = "sanskardeept/cricket-intelligence-platform-ipl-artifacts"
DEFAULT_DOWNLOAD_DIR = ROOT / "kaggle" / "downloaded-artifacts"


def _copy_tree(source: Path, target: Path) -> int:
    copied = 0
    for item in source.rglob("*"):
        if not item.is_file() or item.name in {"dataset-metadata.json", "artifact-manifest.json"}:
            continue
        relative = item.relative_to(source)
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, destination)
        copied += 1
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and restore Kaggle artifacts")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Kaggle dataset slug")
    parser.add_argument("--download-dir", default=str(DEFAULT_DOWNLOAD_DIR), help="Temporary download directory")
    parser.add_argument("--skip-download", action="store_true", help="Restore from an already downloaded folder")
    args = parser.parse_args()

    download_dir = Path(args.download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_download:
        command = [
            sys.executable,
            "-m",
            "kaggle",
            "datasets",
            "download",
            "-d",
            args.dataset,
            "-p",
            str(download_dir),
            "--unzip",
        ]
        subprocess.run(command, check=True)
    copied = _copy_tree(download_dir, ROOT)
    print(f"restored {copied} files from {download_dir}")


if __name__ == "__main__":
    main()
