"""Load cleaned cricket datasets into PostgreSQL."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.preprocessor import load_cricsheet_zip, load_deliveries_csv
from src.db.database import connect, database_url, initialize_schema


@dataclass(frozen=True)
class IngestionSummary:
    """Counts produced by a dataset ingestion run."""

    matches: int
    deliveries: int
    players: int
    venues: int
    dry_run: bool


def stable_id(prefix: str, value: str) -> str:
    """Create a deterministic short ID for natural-key entities."""

    digest = hashlib.sha1(value.strip().lower().encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def _read_source(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if source.suffix.lower() == ".zip":
        return load_cricsheet_zip(source)
    return load_deliveries_csv(source)


def _first_existing(row: pd.Series, columns: list[str], default: Any = None) -> Any:
    for column in columns:
        if column in row and pd.notna(row[column]):
            return row[column]
    return default


def build_match_rows(deliveries: pd.DataFrame, cricket_format: str = "IPL") -> list[dict[str, Any]]:
    """Infer minimal match rows required by the normalized schema."""

    required = {"match_id", "batting_team", "bowling_team"}
    missing = required - set(deliveries.columns)
    if missing:
        raise ValueError(f"deliveries missing required columns: {sorted(missing)}")

    rows: list[dict[str, Any]] = []
    for match_id, group in deliveries.groupby("match_id", sort=False):
        first = group.iloc[0]
        teams = sorted(set(group["batting_team"].dropna()) | set(group["bowling_team"].dropna()))
        if len(teams) < 2:
            teams = [str(first["batting_team"]), str(first["bowling_team"])]
        venue = _first_existing(first, ["venue", "ground"], "Unknown Venue")
        rows.append(
            {
                "match_id": str(match_id),
                "match_date": _first_existing(first, ["date", "start_date"], None),
                "format": cricket_format,
                "team1": str(teams[0]),
                "team2": str(teams[1]),
                "venue": str(venue),
                "toss_winner": _first_existing(first, ["toss_winner"], None),
                "toss_decision": _normalize_toss_decision(_first_existing(first, ["toss_decision"], None)),
                "winner": _first_existing(first, ["winner", "match_winner"], None),
                "result_margin": _first_existing(first, ["result_margin"], None),
            }
        )
    return rows


def build_venue_rows(match_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build venue dimension rows from match metadata."""

    seen: dict[str, dict[str, Any]] = {}
    for match in match_rows:
        name = match["venue"] or "Unknown Venue"
        venue_id = stable_id("venue", name)
        seen[venue_id] = {
            "venue_id": venue_id,
            "name": name,
            "city": None,
            "pitch_type": "balanced",
        }
    return list(seen.values())


def build_player_rows(deliveries: pd.DataFrame) -> list[dict[str, Any]]:
    """Build player dimension rows from batter and bowler columns."""

    names: set[str] = set()
    for column in ["batter", "bowler", "non_striker", "player_dismissed"]:
        if column in deliveries.columns:
            names.update(str(value) for value in deliveries[column].dropna().unique() if str(value).strip())
    return [{"player_id": stable_id("player", name), "name": name} for name in sorted(names)]


def build_delivery_rows(deliveries: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert cleaned deliveries to schema-shaped dictionaries."""

    rows: list[dict[str, Any]] = []
    for _, row in deliveries.iterrows():
        rows.append(
            {
                "match_id": str(row["match_id"]),
                "innings": int(_first_existing(row, ["innings", "inning"], 1) or 1),
                "over_number": int(float(row["over"])),
                "ball_number": min(max(int(float(row["ball"])), 1), 6),
                "batting_team": str(row["batting_team"]),
                "bowling_team": str(row["bowling_team"]),
                "batter": str(_first_existing(row, ["batter"], "Unknown Batter")),
                "bowler": str(_first_existing(row, ["bowler"], "Unknown Bowler")),
                "runs": int(float(row["runs"])),
                "extras": int(float(_first_existing(row, ["extras", "extras_total"], 0) or 0)),
                "is_wicket": bool(int(float(_first_existing(row, ["is_wicket"], 0) or 0))),
                "wicket_type": _first_existing(row, ["wicket_type", "kind"], None),
            }
        )
    return rows


def _normalize_toss_decision(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().lower()
    if text in {"bat", "field"}:
        return text
    if text in {"bowl", "bowling"}:
        return "field"
    return None


def summarize_source(path: str | Path, cricket_format: str = "IPL") -> IngestionSummary:
    """Return row counts without requiring PostgreSQL."""

    deliveries = _read_source(path)
    match_rows = build_match_rows(deliveries, cricket_format)
    return IngestionSummary(
        matches=len(match_rows),
        deliveries=len(deliveries),
        players=len(build_player_rows(deliveries)),
        venues=len(build_venue_rows(match_rows)),
        dry_run=True,
    )


def ingest_deliveries(path: str | Path, cricket_format: str = "IPL", dry_run: bool = False) -> IngestionSummary:
    """Load a deliveries CSV/Cricsheet zip into PostgreSQL."""

    deliveries = _read_source(path)
    match_rows = build_match_rows(deliveries, cricket_format)
    venue_rows = build_venue_rows(match_rows)
    player_rows = build_player_rows(deliveries)
    delivery_rows = build_delivery_rows(deliveries)

    if dry_run or not database_url():
        return IngestionSummary(
            matches=len(match_rows),
            deliveries=len(delivery_rows),
            players=len(player_rows),
            venues=len(venue_rows),
            dry_run=True,
        )

    initialize_schema()
    with connect() as conn:
        with conn.cursor() as cursor:
            cursor.executemany(
                """
                INSERT INTO venues (venue_id, name, city, pitch_type)
                VALUES (%(venue_id)s, %(name)s, %(city)s, %(pitch_type)s)
                ON CONFLICT (venue_id) DO UPDATE SET
                    name = EXCLUDED.name,
                    city = COALESCE(EXCLUDED.city, venues.city),
                    pitch_type = EXCLUDED.pitch_type
                """,
                venue_rows,
            )
            cursor.executemany(
                """
                INSERT INTO matches (
                    match_id, match_date, format, team1, team2, venue,
                    toss_winner, toss_decision, winner, result_margin
                )
                VALUES (
                    %(match_id)s, %(match_date)s, %(format)s, %(team1)s, %(team2)s, %(venue)s,
                    %(toss_winner)s, %(toss_decision)s, %(winner)s, %(result_margin)s
                )
                ON CONFLICT (match_id) DO UPDATE SET
                    match_date = EXCLUDED.match_date,
                    format = EXCLUDED.format,
                    team1 = EXCLUDED.team1,
                    team2 = EXCLUDED.team2,
                    venue = EXCLUDED.venue,
                    toss_winner = EXCLUDED.toss_winner,
                    toss_decision = EXCLUDED.toss_decision,
                    winner = EXCLUDED.winner,
                    result_margin = EXCLUDED.result_margin
                """,
                match_rows,
            )
            cursor.executemany(
                """
                INSERT INTO players (player_id, name)
                VALUES (%(player_id)s, %(name)s)
                ON CONFLICT (player_id) DO UPDATE SET name = EXCLUDED.name
                """,
                player_rows,
            )
            cursor.executemany(
                """
                INSERT INTO deliveries (
                    match_id, innings, over_number, ball_number, batting_team, bowling_team,
                    batter, bowler, runs, extras, is_wicket, wicket_type
                )
                VALUES (
                    %(match_id)s, %(innings)s, %(over_number)s, %(ball_number)s, %(batting_team)s,
                    %(bowling_team)s, %(batter)s, %(bowler)s, %(runs)s, %(extras)s,
                    %(is_wicket)s, %(wicket_type)s
                )
                """,
                delivery_rows,
            )
        conn.commit()

    return IngestionSummary(
        matches=len(match_rows),
        deliveries=len(delivery_rows),
        players=len(player_rows),
        venues=len(venue_rows),
        dry_run=False,
    )
