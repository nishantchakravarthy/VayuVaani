"""Small HTTP helper with retries and rate-limit handling."""

from __future__ import annotations

import logging
import time

import requests

log = logging.getLogger(__name__)

_session = requests.Session()
_session.headers["User-Agent"] = "VayuVaani/0.1 (student project)"


def get(url: str, params=None, headers=None, retries: int = 4, timeout: int = 60) -> requests.Response:
    """GET with retries. Waits and retries on 429 and 5xx responses."""
    delay = 2.0
    for attempt in range(1, retries + 1):
        try:
            resp = _session.get(url, params=params, headers=headers, timeout=timeout)
        except requests.RequestException as exc:
            if attempt == retries:
                raise
            log.warning("Request failed (%s), retrying in %.0fs", exc, delay)
            time.sleep(delay)
            delay *= 2
            continue
        if resp.status_code == 429 or resp.status_code >= 500:
            if attempt == retries:
                resp.raise_for_status()
            wait = float(resp.headers.get("Retry-After", delay))
            log.warning("HTTP %s from %s, retrying in %.0fs", resp.status_code, url, wait)
            time.sleep(wait)
            delay *= 2
            continue
        resp.raise_for_status()
        return resp
    raise RuntimeError("unreachable")
