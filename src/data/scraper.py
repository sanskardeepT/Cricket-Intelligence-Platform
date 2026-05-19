"""Live score scraping helpers with defensive parsing."""

from __future__ import annotations

from dataclasses import dataclass

import requests
from bs4 import BeautifulSoup


@dataclass(frozen=True)
class LiveScore:
    """Minimal live score snapshot parsed from a score page."""

    title: str
    score_text: str
    status: str


def scrape_scorecard(url: str) -> LiveScore:
    """Scrape a public score page and extract a robust text snapshot."""

    if not url.startswith(("http://", "https://")):
        raise ValueError("url must be absolute")
    response = requests.get(url, timeout=20, headers={"User-Agent": "CricketIntelligence/1.0"})
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    title = soup.title.get_text(strip=True) if soup.title else "Live cricket match"
    visible_text = " ".join(soup.get_text(" ", strip=True).split())
    if not visible_text:
        raise RuntimeError("score page did not contain visible text")
    return LiveScore(title=title, score_text=visible_text[:500], status="parsed")

