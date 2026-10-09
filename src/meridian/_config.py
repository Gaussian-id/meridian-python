from __future__ import annotations

import math
from urllib.parse import urlsplit

import httpx

from ._errors import MeridianError, config_error

DEFAULT_TIMEOUT = 60.0
DEFAULT_MAX_RETRIES = 2


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def sane_timeout(value: object) -> float:
    number = _finite_number(value)
    return number if number is not None and number > 0 else DEFAULT_TIMEOUT


def sane_retries(value: object) -> int:
    number = _finite_number(value)
    return math.floor(number) if number is not None and number >= 0 else DEFAULT_MAX_RETRIES


def url_refusal(url: str | None, what: str) -> MeridianError | None:
    try:
        parts = urlsplit(url or '')
        parts.port  # noqa: B018  (reading it validates the port)
        httpx.URL(url or '')
    except (ValueError, httpx.InvalidURL):
        return config_error(f'{what} must be an absolute http(s) URL.')
    if parts.scheme in ('http', 'https') and parts.netloc:
        return None
    return config_error(f'{what} must be an absolute http(s) URL.')
