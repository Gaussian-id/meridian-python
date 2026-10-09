from __future__ import annotations

import json
import random
import re
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, cast

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
# An error body is read this far (an envelope is a few hundred bytes); a buffered reply this far. A body over its cap
# is refused, never truncated: the cap plus one byte is read, and no more (meridian-go's limits)
MAX_ERROR_BODY_BYTES = 1 << 20
MAX_REPLY_BYTES = 64 << 20
_SECONDS = re.compile(r'\d+(?:\.\d+)?')
_REQUEST_ID = re.compile(r'[ -~]{1,255}')  # a request id fit to show: 1-255 printable ASCII characters
# What building or sending a request may raise that becomes an Err: a body or URL that cannot be encoded (TypeError,
# ValueError, which includes UnicodeError), params too deep to encode (RecursionError)
_SEND_ERRORS = (httpx.HTTPError, httpx.InvalidURL, TypeError, ValueError, RecursionError)


@dataclass(frozen=True)
class Call:
    method: str
    path: str
    query: Mapping[str, str | int | None] | None = None
    body: Any = _NO_BODY
    headers: Mapping[str, str] = field(default_factory=lambda: {})
    # True: repeating the call is harmless although it is a write with no idempotency key (see `may_repeat`)
    repeatable: bool = False
    object_body: bool = False  # True: a 2xx body that is not a JSON object is a transport failure
    no_content: bool = False  # True: the operation answers 204 with no body (a DELETE); no other call accepts a 204


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


def may_repeat(call: Call) -> bool:
    """Whether a call that may have reached the server may be sent again. The one place this is decided.

    A read or a delete can be repeated; so can any call that carries an idempotency key (the engine replays it) or is
    marked `repeatable`. Every other call (a write the engine would run twice) is safe by default: it is repeated only
    after a rate limit, which means the engine did not run it.
    """
    return (
        call.repeatable
        or call.method.upper() in ('GET', 'DELETE')
        or any(name.lower() == 'idempotency-key' for name in call.headers)
    )


def next_wait(tried: int, error: MeridianError, retry_after: str | None, call: Call, max_retries: int) -> float | None:
    """Seconds to wait before the next attempt, or None to stop and return the error."""
    # [1] A final error, a spent budget, or a call unsafe to repeat (unless the server refused it before running it)
    if not error.retryable or tried >= max_retries:
        return None
    if not may_repeat(call) and error.code != 'rate_limited':
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
        # [1] Query encoded once; undefined and empty values left out
        params = {k: str(v) for k, v in (call.query or {}).items() if v is not None and v != ''}
        headers = {
            'accept': 'application/json, text/event-stream',
            'user-agent': f'meridian-python/{__version__}',
            **call.headers,
        }

        # [2] JSON body (strict: NaN and infinity are no JSON, so they are a ValueError) and bearer key
        content = None
        if call.body is not _NO_BODY:
            headers['content-type'] = 'application/json'
            content = json.dumps(call.body, allow_nan=False).encode()
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
        if isinstance(exc, UnicodeError):  # the message would quote a character of the request, maybe of a secret
            return config_error('The request could not be built: it holds text that cannot be encoded.')
        if not isinstance(exc, httpx.HTTPError):  # a URL, header or body httpx cannot encode: our setup, not the wire
            return config_error(f'The request could not be built: {exc}')
        if isinstance(exc, httpx.TimeoutException):
            return client_error('network_error', f'No response within {self.timeout:g} s.')
        return client_error('network_error', 'The network request failed.')

    @staticmethod
    def error_status(response: httpx.Response, raw: bytes | None) -> tuple[MeridianError, str | None]:
        # A body over its cap (None) is no envelope: a network error with the real status
        text = '' if raw is None else raw.decode('utf-8', 'replace')
        error = error_from_body(response.status_code, response.headers.get('x-request-id'), text)
        return error, response.headers.get('retry-after')

    @staticmethod
    def dropped(response: httpx.Response) -> Err:
        message = 'The connection dropped while reading the response body.'
        return Err(client_error('network_error', message, response.status_code))

    @staticmethod
    def too_large(response: httpx.Response) -> Err:
        message = 'The reply was larger than the SDK accepts.'
        return Err(client_error('network_error', message, response.status_code))

    @staticmethod
    def take(data: bytearray, piece: bytes, cap: int) -> bool:
        """Add a piece of a body being read; False once it holds more than `cap` bytes (stop reading)."""
        data += piece
        return len(data) <= cap

    @staticmethod
    def no_content(call: Call, response: httpx.Response) -> Result[Any]:
        """A 204: nothing to read, and an answer only for the call that expects it."""
        if call.no_content:
            return Ok(None, request_id=response.headers.get('x-request-id'))
        message = 'The engine answered with no content, which this call does not expect.'
        return Err(client_error('network_error', message, response.status_code))

    @staticmethod
    def decode(call: Call, response: httpx.Response, raw: bytes | None) -> Result[Any]:
        # [1] Over the cap, not JSON, or nested deeper than the parser allows (RecursionError): a transport failure
        status, header_id = response.status_code, response.headers.get('x-request-id')
        if raw is None:
            return _Shared.too_large(response)
        try:
            data: Any = json.loads(raw.decode('utf-8', 'replace'))
        except (ValueError, RecursionError):
            return Err(client_error('network_error', 'The response was not valid JSON.', status))
        if data is None:  # a literal null is nothing: no call takes it for a result
            return Err(client_error('network_error', 'The response was empty (JSON null).', status))

        # [2] A reply that says `error` is no result, whatever else it holds: every result is an object without one.
        # The error's own request id is kept when it is fit to show, else the response's
        error: Any = cast('dict[str, Any]', data).get('error') if isinstance(data, dict) else None
        if error is not None:
            own: Any = cast('dict[str, Any]', error).get('request_id') if isinstance(error, dict) else None
            usable = own if isinstance(own, str) and _REQUEST_ID.fullmatch(own) else header_id
            message = "The engine's reply carried an error instead of a result."
            return Err(replace(client_error('network_error', message, status), request_id=usable))
        if call.object_body and not isinstance(data, dict):
            return Err(client_error('network_error', 'The response was not a JSON object.', status))
        return Ok[Any](None if call.no_content else data, request_id=header_id)


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

        # [2] An error status: read its body once (up to its cap), then close
        try:
            raw = self._read(response, MAX_ERROR_BODY_BYTES)
        except httpx.HTTPError:
            raw = b''
        finally:
            response.close()
        error, retry_after = self._shared.error_status(response, raw)
        return None, error, retry_after

    @staticmethod
    def _read(response: httpx.Response, cap: int) -> bytes | None:
        """The (decoded) body, or None once it is over `cap` bytes: the rest is never read."""
        data = bytearray()
        for piece in response.iter_bytes():
            if not _Shared.take(data, piece, cap):
                return None
        return bytes(data)

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
            # [2] No content: None for the call that expects it, a failure for every other
            if response.status_code == 204:
                return self._shared.no_content(call, response)

            # [3] A body that drops mid-read, is too large or is not JSON is a transport failure
            try:
                raw = self._read(response, MAX_REPLY_BYTES)
            except httpx.HTTPError:
                return self._shared.dropped(response)
            return self._shared.decode(call, response, raw)
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

        # [2] An error status: read its body once (up to its cap), then close
        try:
            raw = await self._read(response, MAX_ERROR_BODY_BYTES)
        except httpx.HTTPError:
            raw = b''
        finally:
            await response.aclose()
        error, retry_after = self._shared.error_status(response, raw)
        return None, error, retry_after

    @staticmethod
    async def _read(response: httpx.Response, cap: int) -> bytes | None:
        """The (decoded) body, or None once it is over `cap` bytes: the rest is never read."""
        data = bytearray()
        async for piece in response.aiter_bytes():
            if not _Shared.take(data, piece, cap):
                return None
        return bytes(data)

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
            # [2] No content: None for the call that expects it, a failure for every other
            if response.status_code == 204:
                return self._shared.no_content(call, response)

            # [3] A body that drops mid-read, is too large or is not JSON is a transport failure
            try:
                raw = await self._read(response, MAX_REPLY_BYTES)
            except httpx.HTTPError:
                return self._shared.dropped(response)
            return self._shared.decode(call, response, raw)
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
