"""Live conditions with no API key: Open-Meteo weather and the CAMS air quality model.

CAMS (Copernicus Atmosphere Monitoring Service) is a global air quality model run
by the European Centre for Medium-Range Weather Forecasts. It is a ~10–40 km grid
estimate, not a station measurement, so the app labels it "CAMS model estimate"
and uses it as an independent reference next to VayuVaani's station forecasts.
"""

from __future__ import annotations

import threading
import time

from .http import get

AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WX_URL = "https://api.open-meteo.com/v1/forecast"
CACHE_SECONDS = 15 * 60
FAIL_SECONDS = 2 * 60  # after a failed fetch, don't retry (and stall the page) for a while

_cache: dict = {}
_lock = threading.Lock()


def _cached(key, fn):
    with _lock:
        hit = _cache.get(key)
        if hit and not isinstance(hit[1], Exception) and time.time() - hit[0] < CACHE_SECONDS:
            return hit[1]
        if hit and isinstance(hit[1], Exception) and time.time() - hit[0] < FAIL_SECONDS:
            raise hit[1]
    try:
        value = fn()
    except Exception as e:
        with _lock:
            _cache[key] = (time.time() - CACHE_SECONDS + FAIL_SECONDS, e)
        raise
    with _lock:
        _cache[key] = (time.time(), value)
    return value


def weather_now(lat: float, lon: float, tz: str = "Asia/Kolkata") -> dict:
    def fetch():
        params = {
            "latitude": lat, "longitude": lon, "timezone": tz, "forecast_days": 2,
            "current": "temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,"
                       "weather_code,wind_speed_10m,wind_direction_10m,cloud_cover",
            "hourly": "temperature_2m,precipitation_probability,wind_speed_10m",
            "daily": "sunrise,sunset,temperature_2m_max,temperature_2m_min",
        }
        return get(WX_URL, params=params, timeout=20).json()
    return _cached(("wx", round(lat, 2), round(lon, 2)), fetch)


def air_quality(lat: float, lon: float, tz: str = "Asia/Kolkata") -> dict:
    """CAMS current values plus hourly PM2.5/PM10 for yesterday to 3 days ahead."""
    def fetch():
        params = {
            "latitude": lat, "longitude": lon, "timezone": tz,
            "past_days": 1, "forecast_days": 3,
            "current": "pm2_5,pm10,nitrogen_dioxide,ozone,carbon_monoxide,sulphur_dioxide,us_aqi",
            "hourly": "pm2_5,pm10",
        }
        return get(AQ_URL, params=params, timeout=20).json()
    return _cached(("aq", round(lat, 2), round(lon, 2)), fetch)


WEATHER_CODES = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast", 45: "Fog", 48: "Fog",
    51: "Light drizzle", 53: "Drizzle", 55: "Heavy drizzle", 61: "Light rain", 63: "Rain",
    65: "Heavy rain", 80: "Rain showers", 81: "Rain showers", 82: "Heavy showers",
    95: "Thunderstorm", 96: "Thunderstorm with hail", 99: "Thunderstorm with hail",
}
