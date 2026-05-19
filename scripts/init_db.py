"""Initialize the Cricket Intelligence PostgreSQL schema."""

from __future__ import annotations

from src.db.database import initialize_schema


if __name__ == "__main__":
    created = initialize_schema()
    print("database schema initialized" if created else "DATABASE_URL not configured; skipped")
