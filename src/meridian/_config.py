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


# The client and transport classes each mode takes
SYNC_IO = (httpx.Client, httpx.BaseTransport)
ASYNC_IO = (httpx.AsyncClient, httpx.AsyncBaseTransport)


def io_refusal(http_client: object, transport: object, kinds: tuple[type, type]) -> MeridianError | None:
    """Why `http_client` and `transport` cannot be used together or at all (`kinds`: the client and transport classes
    this mode takes), or None."""
    client_kind, transport_kind = kinds
    if http_client is not None and transport is not None:
        return config_error('Pass http_client or transport, not both: transport would be ignored.')
    if http_client is not None and not isinstance(http_client, client_kind):
        return config_error(f'http_client must be an httpx.{client_kind.__name__}.')
    if transport is not None and not isinstance(transport, transport_kind):
        return config_error(f'transport must be an httpx.{transport_kind.__name__}.')
    return None


def url_refusal(url: object, what: str) -> MeridianError | None:
    if not isinstance(url, str):
        return config_error(f'{what} must be an absolute http(s) URL.')
    try:
        parts = urlsplit(url or '')
        parts.port  # noqa: B018  (reading it validates the port)
        httpx.URL(url or '')
    except (ValueError, httpx.InvalidURL):
        return config_error(f'{what} must be an absolute http(s) URL.')
    if parts.scheme in ('http', 'https') and parts.netloc:
        return None
    return config_error(f'{what} must be an absolute http(s) URL.')
