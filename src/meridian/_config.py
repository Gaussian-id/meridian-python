from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

import httpx

from ._errors import MeridianError, config_error

DEFAULT_TIMEOUT = 60.0
DEFAULT_MAX_RETRIES = 2


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def given_timeout(value: object) -> float | None:
    """The caller's timeout when it is a usable number of seconds (finite, > 0, not a bool); otherwise None."""
    number = _finite_number(value)
    return number if number is not None and number > 0 else None


def sane_timeout(value: object) -> float:
    return given_timeout(value) or DEFAULT_TIMEOUT


def sane_retries(value: object) -> int:
    number = _finite_number(value)
    return math.floor(number) if number is not None and number >= 0 else DEFAULT_MAX_RETRIES


# The client and transport classes each mode takes
SYNC_IO = (httpx.Client, httpx.BaseTransport)
ASYNC_IO = (httpx.AsyncClient, httpx.AsyncBaseTransport)


@dataclass(frozen=True)
class Listener:
    """Which engine listener a client talks to: where its settings come from and how a refusal names them."""

    url_env: str
    key_env: str
    url_name: str
    key_name: str


PUBLIC = Listener('MERIDIAN_URL', 'MERIDIAN_KEY', 'The Meridian URL', 'A Meridian project key')
ADMIN = Listener('MERIDIAN_ADMIN_URL', 'MERIDIAN_INTEGRATOR_KEY', 'The Meridian admin URL', 'A Meridian integrator key')


def resolve_settings(url: object, key: object, listener: Listener) -> tuple[Any, Any]:
    """The URL and key as given, each falling back to its environment variable only when it is None (so an explicit
    `''` beats the environment)."""
    return (
        url if url is not None else os.environ.get(listener.url_env, ''),
        key if key is not None else os.environ.get(listener.key_env, ''),
    )


def refusal(
    url: object, key: object, http_client: object, transport: object, listener: Listener, kinds: tuple[type, type]
) -> MeridianError | None:
    """The configuration error a client answers every call with, or None when the settings are usable (`kinds`: the
    client and transport classes this mode takes)."""
    # [1] An absolute http(s) URL and a non-empty key, both strings
    if refused := url_refusal(url, listener.url_name):
        return refused
    if not isinstance(key, str) or not key:
        return config_error(f'{listener.key_name} is required: pass key= or set {listener.key_env}.')

    # [2] At most one of http_client and transport, and each of this mode's kind
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
