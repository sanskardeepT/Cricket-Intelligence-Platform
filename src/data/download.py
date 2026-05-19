"""Download free cricket datasets from Cricsheet."""

from __future__ import annotations

from pathlib import Path
from urllib.error import URLError
from urllib.request import urlretrieve


CRICSHEET_URLS = {
    "ipl": "https://cricsheet.org/downloads/ipl_csv2.zip",
    "t20s": "https://cricsheet.org/downloads/t20s_csv2.zip",
    "odis": "https://cricsheet.org/downloads/odis_csv2.zip",
}


def download_cricsheet_dataset(kind: str, output_dir: str | Path = "data/raw") -> Path:
    """Download a Cricsheet zip and return the saved path."""

    if kind not in CRICSHEET_URLS:
        raise ValueError(f"kind must be one of {sorted(CRICSHEET_URLS)}")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{kind}_csv2.zip"
    if target.exists() and target.stat().st_size > 0:
        return target
    try:
        urlretrieve(CRICSHEET_URLS[kind], target)
    except URLError as exc:
        raise RuntimeError(f"Could not download {kind} from Cricsheet: {exc}") from exc
    if target.stat().st_size == 0:
        raise RuntimeError(f"Downloaded file is empty: {target}")
    return target


def download_all_cricsheet(output_dir: str | Path = "data/raw") -> dict[str, Path]:
    """Download IPL, T20I, and ODI Cricsheet CSV zips."""

    return {kind: download_cricsheet_dataset(kind, output_dir) for kind in CRICSHEET_URLS}

