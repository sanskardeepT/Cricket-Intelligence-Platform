"""OpenWeatherMap integration for venue context."""

from __future__ import annotations

import os
from dataclasses import dataclass

import requests


@dataclass(frozen=True)
class WeatherSnapshot:
    """Weather features used by toss and live prediction explainers."""

    city: str
    temperature_c: float
    humidity: float
    wind_speed: float
    dew_probability: float


def estimate_dew_probability(humidity: float, temperature_c: float) -> float:
    """Estimate dew likelihood from humidity and temperature."""

    humidity_component = max(0.0, min(1.0, (humidity - 45.0) / 45.0))
    temp_component = max(0.0, min(1.0, (32.0 - temperature_c) / 14.0))
    return round((humidity_component * 0.75 + temp_component * 0.25) * 100.0, 2)


def fetch_weather(city: str, api_key: str | None = None) -> WeatherSnapshot:
    """Fetch current weather from OpenWeatherMap."""

    key = api_key or os.getenv("OPENWEATHER_API_KEY")
    if not key:
        raise RuntimeError("OPENWEATHER_API_KEY is required for live weather")
    response = requests.get(
        "https://api.openweathermap.org/data/2.5/weather",
        params={"q": city, "appid": key, "units": "metric"},
        timeout=15,
    )
    response.raise_for_status()
    payload = response.json()
    temp = float(payload["main"]["temp"])
    humidity = float(payload["main"]["humidity"])
    wind = float(payload.get("wind", {}).get("speed", 0.0))
    return WeatherSnapshot(
        city=city,
        temperature_c=temp,
        humidity=humidity,
        wind_speed=wind,
        dew_probability=estimate_dew_probability(humidity, temp),
    )

