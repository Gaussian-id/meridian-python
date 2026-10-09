from __future__ import annotations

import json
import random
import re
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

import httpx

from . import _clock
from ._config import DEFAULT_MAX_RETRIES, DEFAULT_TIMEOUT, sane_retries, sane_timeout
from ._errors import Err, MeridianError, Ok, Result, client_error, config_error, error_from_body
from ._version import __version__

_NO_BODY: Any = object()

__all__ = [
    'DEFAULT_MAX_RETRIES',
    'DEFAULT_TIMEOUT',
    'MAX_RETRY_AFTER',
    'AsyncRefusedTransport',
    'AsyncTransport',
    'Call',
    'RefusedTransport',
    'SyncTransport',
    'new_idempotency_key',
    'next_wait',
    'retry_after_seconds',
]

BASE_DELAY = 0.5
MAX_JITTER = 0.25
MAX_RETRY_AFTER = 30.0
_SECONDS = re.compile(r'\d+(?:\.\d+)?')
# What building or sending a request may raise that becomes an Err (RecursionError: params too deep to encode)
_SEND_ERRORS = (httpx.HTTPError, httpx.InvalidURL, UnicodeError, RecursionError)


@dataclass(frozen=True)
class Call:
    method: str
    path: str
    query: Mapping[str, str | int | None] | None = None
    body: Any = _NO_BODY
    headers: Mapping[str, str] = field(default_factory=lambda: {})
    retry: bool = True  # False: not safe to repeat once it may have reached the server
    object_body: bool = False  # True: a 2xx body that is not a JSON object is a transport failure


def new_idempotency_key() -> str:
    return str(uuid.uuid4())


def retry_after_seconds(value: str | None) -> float | None:
    """`Retry-After` as plain decimal seconds or an HTTP date; None when absent, unreadable, zero or past."""
    if not value or not value.strip():
        return None
    text = value.strip()
    if _SECONDS.fullmatch(text):
        seconds = float(text)
    else:
        try:
            when = parsedate_to_datetime(text)
        except (TypeError, ValueError):
            return None
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        seconds = (when - datetime.now(timezone.utc)).total_seconds()
    return seconds if seconds > 0 else None


def next_wait(tried: int, error: MeridianError, retry_after: str | None, call: Call, max_retries: int) -> float | None:
    """Seconds to wait before the next attempt, or None to stop and return the error."""
    # [1] A final error, a spent budget, or a call unsafe to repeat (unless the server refused it before running it)
    if not error.retryable or tried >= max_retries:
        return None
    if not call.retry and error.code != 'rate_limited':
        return None

    # [2] A server asking for a long wait gets its answer now
    wait = retry_after_seconds(retry_after)
    if wait is not None:
        return None if wait > MAX_RETRY_AFTER else wait

    # [3] Exponential backoff with jitter
    return BASE_DELAY * 2**tried + random.uniform(0, MAX_JITTER)


class _Shared:
    """Everything the sync and async transports have in common: only the I/O differs."""

    def __init__(self, base_url: str, key: str | None, timeout: object, max_retries: object) -> None:
        self.base = base_url.rstrip('/')
        self.key = key
        self.timeout = sane_timeout(timeout)
        self.max_retries = sane_retries(max_retries)

    def request(self, client: httpx.Client | httpx.AsyncClient, call: Call) -> httpx.Request:
        # [1] Query encoded once; undefined values left out
        params = {k: str(v) for k, v in (call.query or {}).items() if v is not None}
        headers = {
            'accept': 'application/json, text/event-stream',
            'user-agent': f'meridian-python/{__version__}',
            **call.headers,
        }

        # [2] JSON body and bearer key
        content = None
        if call.body is not _NO_BODY:
            headers['content-type'] = 'application/json'
            content = json.dumps(call.body).encode()
        if self.key:
            headers['authorization'] = f'Bearer {self.key}'
        return client.build_request(
            call.method,
            self.base + call.path,
            params=params or None,
            headers=headers,
            content=content,
            timeout=self.timeout,
        )

    def failure(self, exc: BaseException) -> MeridianError:
        if not isinstance(exc, httpx.HTTPError):  # a URL or header httpx cannot encode: our setup, not the wire
            return config_error(f'The request could not be built: {exc}')
        if isinstance(exc, httpx.TimeoutException):
            return client_error('network_error', f'No response within {self.timeout:g} s.')
        return client_error('network_error', 'The network request failed.')

    @staticmethod
    def error_status(response: httpx.Response, text: str) -> tuple[MeridianError, str | None]:
        error = error_from_body(response.status_code, response.headers.get('x-request-id'), text)
        return error, response.headers.get('retry-after')

    @staticmethod
    def dropped(response: httpx.Response) -> Err:
        message = 'The connection dropped while reading the response body.'
        return Err(client_error('network_error', message, response.status_code))

    @staticmethod
    def decode(call: Call, response: httpx.Response, text: str) -> Result[Any]:
        try:
            data: Any = json.loads(text)
        except (ValueError, RecursionError):  # RecursionError: nested deeper than the parser allows
            return Err(client_error('network_error', 'The response was not valid JSON.', response.status_code))
        if call.object_body and not isinstance(data, dict):
            return Err(client_error('network_error', 'The response was not a JSON object.', response.status_code))
        return Ok[Any](data, request_id=response.headers.get('x-request-id'))


class SyncTransport:
    def __init__(
        self,
        base_url: str,
        key: str | None,
        *,
        timeout: object = None,
        max_retries: object = None,
        http_client: httpx.Client | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._shared = _Shared(base_url, key, timeout, max_retries)
        self._owns = http_client is None
        self._client = http_client or httpx.Client(transport=transport)

    @property
    def max_retries(self) -> int:
        return self._shared.max_retries

    def _attempt(self, call: Call) -> tuple[httpx.Response | None, MeridianError | None, str | None]:
        # [1] Send, keeping the body unread
        try:
            response = self._client.send(self._shared.request(self._client, call), stream=True)
        except _SEND_ERRORS as exc:
            return None, self._shared.failure(exc), None
        if response.is_success:
            return response, None, None

        # [2] An error status: read its body once, then close
        try:
            text = response.read().decode('utf-8', 'replace')
        except httpx.HTTPError:
            text = ''
        finally:
            response.close()
        error, retry_after = self._shared.error_status(response, text)
        return None, error, retry_after

    def open(self, call: Call) -> Result[httpx.Response]:
        # [1] Attempt, backing off between retryable failures; the same Call (and key) every time
        tried = 0
        while True:
            response, error, retry_after = self._attempt(call)
            if response is not None:
                return Ok(response)
            if error is None:  # unreachable, but never an assert: python -O must still not raise
                return Err(client_error('network_error', 'The request failed.'))
            wait = next_wait(tried, error, retry_after, call, self._shared.max_retries)
            if wait is None:
                return Err(error)
            _clock.sleep(wait)
            tried += 1

    def json(self, call: Call) -> Result[Any]:
        # [1] The response, or the failure to get one
        opened = self.open(call)
        if opened.error is not None:
            return opened
        response = opened.data
        try:
            # [2] No content is None data
            if response.status_code == 204:
                return Ok(None)

            # [3] A body that drops mid-read, or is not JSON, is a transport failure
            try:
                text = response.read().decode('utf-8', 'replace')
            except httpx.HTTPError:
                return self._shared.dropped(response)
            return self._shared.decode(call, response, text)
        finally:
            response.close()

    def close(self) -> None:
        if self._owns:
            self._client.close()


class AsyncTransport:
    """SyncTransport with awaited I/O. A CancelledError is never caught: it propagates after the response closes."""

    def __init__(
        self,
        base_url: str,
        key: str | None,
        *,
        timeout: object = None,
        max_retries: object = None,
        http_client: httpx.AsyncClient | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._shared = _Shared(base_url, key, timeout, max_retries)
        self._owns = http_client is None
        self._client = http_client or httpx.AsyncClient(transport=transport)

    @property
    def max_retries(self) -> int:
        return self._shared.max_retries

    async def _attempt(self, call: Call) -> tuple[httpx.Response | None, MeridianError | None, str | None]:
        # [1] Send, keeping the body unread
        try:
            response = await self._client.send(self._shared.request(self._client, call), stream=True)
        except _SEND_ERRORS as exc:
            return None, self._shared.failure(exc), None
        if response.is_success:
            return response, None, None

        # [2] An error status: read its body once, then close
        try:
            text = (await response.aread()).decode('utf-8', 'replace')
        except httpx.HTTPError:
            text = ''
        finally:
            await response.aclose()
        error, retry_after = self._shared.error_status(response, text)
        return None, error, retry_after

    async def open(self, call: Call) -> Result[httpx.Response]:
        # [1] Attempt, backing off between retryable failures; a CancelledError in the wait propagates
        tried = 0
        while True:
            response, error, retry_after = await self._attempt(call)
            if response is not None:
                return Ok(response)
            if error is None:  # unreachable, but never an assert: python -O must still not raise
                return Err(client_error('network_error', 'The request failed.'))
            wait = next_wait(tried, error, retry_after, call, self._shared.max_retries)
            if wait is None:
                return Err(error)
            await _clock.asleep(wait)
            tried += 1

    async def json(self, call: Call) -> Result[Any]:
        # [1] The response, or the failure to get one
        opened = await self.open(call)
        if opened.error is not None:
            return opened
        response = opened.data
        try:
            # [2] No content is None data
            if response.status_code == 204:
                return Ok(None)

            # [3] A body that drops mid-read, or is not JSON, is a transport failure
            try:
                text = (await response.aread()).decode('utf-8', 'replace')
            except httpx.HTTPError:
                return self._shared.dropped(response)
            return self._shared.decode(call, response, text)
        finally:
            await response.aclose()

    async def aclose(self) -> None:
        if self._owns:
            await self._client.aclose()


class RefusedTransport:
    """Every call answers the same configuration error; nothing is sent."""

    max_retries = 0

    def __init__(self, error: MeridianError) -> None:
        self._error = error

    def open(self, call: Call) -> Result[Any]:
        return Err(self._error)

    def json(self, call: Call) -> Result[Any]:
        return Err(self._error)

    def close(self) -> None:
        return None


class AsyncRefusedTransport:
    """The async twin of RefusedTransport."""

    max_retries = 0

    def __init__(self, error: MeridianError) -> None:
        self._error = error

    async def open(self, call: Call) -> Result[Any]:
        return Err(self._error)

    async def json(self, call: Call) -> Result[Any]:
        return Err(self._error)

    async def aclose(self) -> None:
        return None
