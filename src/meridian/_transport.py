from __future__ import annotations

import json
import random
import re
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, cast

import httpx

from . import _clock
from ._config import DEFAULT_MAX_RETRIES, DEFAULT_TIMEOUT, given_timeout, sane_retries
from ._errors import (
    NOT_JSON,
    Err,
    MeridianError,
    Ok,
    Result,
    client_error,
    config_error,
    error_from_body,
    error_member,
    load_json,
    valid_request_id,
)
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
    'retry_after_seconds',
]

BASE_DELAY = 0.5
MAX_JITTER = 0.25
MAX_RETRY_AFTER = 30.0
CHAT_READ_TIMEOUT = 300.0  # a buffered chat reply may take minutes to write: its default wait to read
# An error body is read this far (an envelope is a few hundred bytes); a buffered reply this far. A body over its cap
# is refused, never truncated: the cap plus one byte is read, and no more (meridian-go's limits)
MAX_ERROR_BODY_BYTES = 1 << 20
MAX_REPLY_BYTES = 64 << 20
_SECONDS = re.compile(r'\d+(?:\.\d+)?')
_DELTA_SECONDS = re.compile(r'\d+')
# What reading an HTTP date may raise: a malformed one (TypeError, ValueError), a year no datetime holds (OverflowError,
# from the parser or from the arithmetic on a date at the edge of the range)
_BAD_DATE = (TypeError, ValueError, OverflowError)
# The engine's rule for an idempotency key (meridian-domain ids.rs): 1-255 characters, each visible ASCII 0x21-0x7E
_IDEMPOTENCY_KEY = re.compile(r'[!-~]{1,255}')
# What building or sending a request may raise that becomes an Err: a body or URL that cannot be encoded (TypeError,
# ValueError, which includes UnicodeError), params too deep to encode (RecursionError)
_SEND_ERRORS = (httpx.HTTPError, httpx.InvalidURL, TypeError, ValueError, RecursionError)
# Every call on a closed client (the SDK's, or the httpx client the caller passed in) answers this; nothing is sent
CLOSED = config_error('The client is closed.')


@dataclass(frozen=True)
class Call:
    method: str
    path: str
    query: Mapping[str, str | int | None] | None = None
    body: Any = _NO_BODY
    # Sent as the `idempotency-key` header on every attempt: the engine replays a repeat instead of running it twice
    idempotency_key: str | None = None
    # True: repeating the call is harmless although it is a write with no idempotency key (see `may_repeat`)
    repeatable: bool = False
    object_body: bool = False  # True: a 2xx body that is not a JSON object is a transport failure
    no_content: bool = False  # True: the operation answers 204 with no body (a DELETE); no other call accepts a 204
    read_timeout: float | None = None  # this call's own default wait to read (a caller's timeout overrides it)


def new_idempotency_key() -> str:
    return str(uuid.uuid4())


def valid_idempotency_key(value: object) -> bool:
    """The one rule for an idempotency key, the caller's and a browser's: what the engine accepts."""
    return isinstance(value, str) and _IDEMPOTENCY_KEY.fullmatch(value) is not None


def valid_retry_after(value: str | None) -> str | None:
    """A `Retry-After` header worth handing on, trimmed: delta-seconds or an HTTP date. Anything else is None."""
    text = (value or '').strip()
    if not valid_request_id(text):  # printable ASCII only, so it can never break a header
        return None
    if _DELTA_SECONDS.fullmatch(text):
        return text
    try:
        parsedate_to_datetime(text)
    except _BAD_DATE:
        return None
    return text


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
            if when.tzinfo is None:
                when = when.replace(tzinfo=timezone.utc)
            seconds = (when - datetime.now(timezone.utc)).total_seconds()
        except _BAD_DATE:
            return None
    return seconds if seconds > 0 else None


def may_repeat(call: Call) -> bool:
    """Whether a call that may have reached the server may be sent again. The one place this is decided.

    A read or a delete can be repeated; so can any call that carries an idempotency key (the engine replays it) or is
    marked `repeatable`. Every other call (a write the engine would run twice) is safe by default: it is repeated only
    after a rate limit, which means the engine did not run it.
    """
    return call.repeatable or call.idempotency_key is not None or call.method.upper() in ('GET', 'DELETE')


class _Attempts:
    """The retry state of one call, without any I/O: how many attempts failed, and the last failure. Each failure goes
    in through `after`, which answers the seconds to wait before the next attempt, or the error to return."""

    def __init__(self, call: Call, max_retries: int) -> None:
        self._call = call
        self._max_retries = max_retries
        self._tried = 0
        self._previous: MeridianError | None = None

    def after(self, error: MeridianError, retry_after: str | None) -> float | MeridianError:
        # [1] A retry that finds its own key in use: the attempt before it reached the engine, which still holds the
        # key, and failed; that failure is the answer. Not after a rate limit: the engine releases a key it refused
        # before running the attempt, so a conflict then is a real one
        previous, self._previous = self._previous, error
        if (
            previous is not None
            and self._call.idempotency_key is not None
            and error.code == 'idempotency_conflict'
            and previous.code != 'rate_limited'
        ):
            return previous

        # [2] A final error, a spent budget, or a call unsafe to repeat (unless the server refused it before running it)
        if not error.retryable or self._tried >= self._max_retries:
            return error
        if not may_repeat(self._call) and error.code != 'rate_limited':
            return error

        # [3] A server asking for a long wait gets its answer now; otherwise its wait, or backoff with jitter
        wait = retry_after_seconds(retry_after)
        if wait is not None and wait > MAX_RETRY_AFTER:
            return error
        if wait is None:
            wait = BASE_DELAY * 2**self._tried + random.uniform(0, MAX_JITTER)
        self._tried += 1
        return wait


class _Shared:
    """Everything the sync and async transports have in common: only the I/O differs."""

    def __init__(self, base_url: str, key: str | None, timeout: object, max_retries: object) -> None:
        self.base = base_url.rstrip('/')
        self.key = key
        # The caller's timeout applies to every call; without one, each call has its own default (CHAT_READ_TIMEOUT
        # to read a buffered chat reply, DEFAULT_TIMEOUT for everything else)
        self.given_timeout = given_timeout(timeout)
        self.max_retries = sane_retries(max_retries)

    def timeout(self, call: Call) -> httpx.Timeout:
        if self.given_timeout is not None:
            return httpx.Timeout(self.given_timeout)
        return httpx.Timeout(DEFAULT_TIMEOUT, read=call.read_timeout or DEFAULT_TIMEOUT)

    def request(self, client: httpx.Client | httpx.AsyncClient, call: Call) -> httpx.Request:
        # [1] Query encoded once; undefined and empty values left out
        params = {k: str(v) for k, v in (call.query or {}).items() if v is not None and v != ''}
        headers = {'accept': 'application/json, text/event-stream', 'user-agent': f'meridian-python/{__version__}'}
        if call.idempotency_key is not None:
            headers['idempotency-key'] = call.idempotency_key

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
            timeout=self.timeout(call),
        )

    def failure(self, exc: BaseException, call: Call) -> MeridianError:
        if isinstance(exc, UnicodeError):  # the message would quote a character of the request, maybe of a secret
            return config_error('The request could not be built: it holds text that cannot be encoded.')
        if not isinstance(exc, httpx.HTTPError):  # a URL, header or body httpx cannot encode: our setup, not the wire
            return config_error(f'The request could not be built: {exc}')
        if isinstance(exc, httpx.TimeoutException):
            timeout = self.timeout(call)
            seconds = timeout.read if isinstance(exc, httpx.ReadTimeout) else timeout.connect
            return client_error('network_error', f'No response within {seconds or DEFAULT_TIMEOUT:g} s.')
        return client_error('network_error', 'The network request failed.')

    @staticmethod
    def error_status(response: httpx.Response, raw: bytes | None) -> tuple[MeridianError, str | None]:
        # A body over its cap (None) is no envelope: a network error with the real status
        text = '' if raw is None else raw.decode('utf-8', 'replace')
        error = error_from_body(response.status_code, response.headers.get('x-request-id'), text)
        retry_after = response.headers.get('retry-after')
        return replace(error, retry_after=valid_retry_after(retry_after)), retry_after

    @staticmethod
    def unusable(response: httpx.Response, message: str) -> Err:
        """A reply the SDK cannot use: a network error with its status and its header's request id (when valid)."""
        request_id = valid_request_id(response.headers.get('x-request-id'))
        return Err(client_error('network_error', message, response.status_code, request_id))

    @staticmethod
    def dropped(response: httpx.Response) -> Err:
        return _Shared.unusable(response, 'The connection dropped while reading the response body.')

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
        return _Shared.unusable(response, 'The engine answered with no content, which this call does not expect.')

    @staticmethod
    def decode(call: Call, response: httpx.Response, raw: bytes | None) -> Result[Any]:
        # [1] Over the cap, not JSON, or nested deeper than the parser allows (RecursionError): a transport failure.
        # The message does not say which of the last two: a body too deep to decode is not known to be invalid JSON
        status, header_id = response.status_code, response.headers.get('x-request-id')
        if raw is None:
            return _Shared.unusable(response, 'The reply was larger than the SDK accepts.')
        data: Any = load_json(raw.decode('utf-8', 'replace'))
        if data is NOT_JSON:
            return _Shared.unusable(response, 'The response could not be decoded.')
        if data is None:  # a literal null is nothing: no call takes it for a result
            return _Shared.unusable(response, 'The response was empty (JSON null).')

        # [2] A reply that says `error` is no result, whatever else it holds: every result is an object without one,
        # and a case variant of a member the SDK reads is never judged. The error's own request id is kept when it
        # is fit to show, else the response's
        error = error_member(data)
        if error is not None:
            own: Any = cast('dict[str, Any]', error).get('request_id') if isinstance(error, dict) else None
            message = "The engine's reply carried an error instead of a result."
            return Err(
                client_error('network_error', message, status, valid_request_id(own) or valid_request_id(header_id))
            )
        if call.object_body and not isinstance(data, dict):
            return _Shared.unusable(response, 'The response was not a JSON object.')
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
        self._closed = False

    @property
    def is_closed(self) -> bool:
        return self._closed

    def _attempt(self, call: Call) -> tuple[httpx.Response | None, MeridianError | None, str | None]:
        # [1] Send, keeping the body unread; httpx raises RuntimeError for a send on a client that is closed
        try:
            response = self._client.send(self._shared.request(self._client, call), stream=True)
        except _SEND_ERRORS as exc:
            return None, self._shared.failure(exc, call), None
        except RuntimeError:
            if not self._client.is_closed:
                raise
            return None, CLOSED, None
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
        # [1] Nothing is sent on a closed client
        if self._closed or self._client.is_closed:
            return Err(CLOSED)

        # [2] Attempt, and wait as _Attempts says between failures; the same Call (and key) every time
        attempts = _Attempts(call, self._shared.max_retries)
        while True:
            response, error, retry_after = self._attempt(call)
            if response is not None:
                return Ok(response)
            if error is None:  # unreachable, but never an assert: python -O must still not raise
                return Err(client_error('network_error', 'The request failed.'))
            wait = attempts.after(error, retry_after)
            if isinstance(wait, MeridianError):
                return Err(wait)
            _clock.sleep(wait)

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
        self._closed = True
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
        self._closed = False

    @property
    def is_closed(self) -> bool:
        return self._closed

    async def _attempt(self, call: Call) -> tuple[httpx.Response | None, MeridianError | None, str | None]:
        # [1] Send, keeping the body unread; httpx raises RuntimeError for a send on a client that is closed
        try:
            response = await self._client.send(self._shared.request(self._client, call), stream=True)
        except _SEND_ERRORS as exc:
            return None, self._shared.failure(exc, call), None
        except RuntimeError:
            if not self._client.is_closed:
                raise
            return None, CLOSED, None
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
        # [1] Nothing is sent on a closed client
        if self._closed or self._client.is_closed:
            return Err(CLOSED)

        # [2] Attempt, and wait as _Attempts says between failures; a CancelledError in the wait propagates
        attempts = _Attempts(call, self._shared.max_retries)
        while True:
            response, error, retry_after = await self._attempt(call)
            if response is not None:
                return Ok(response)
            if error is None:  # unreachable, but never an assert: python -O must still not raise
                return Err(client_error('network_error', 'The request failed.'))
            wait = attempts.after(error, retry_after)
            if isinstance(wait, MeridianError):
                return Err(wait)
            await _clock.asleep(wait)

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
        self._closed = True
        if self._owns:
            await self._client.aclose()


class RefusedTransport:
    """Every call answers the same configuration error (once closed, CLOSED); nothing is sent."""

    def __init__(self, error: MeridianError) -> None:
        self._error = error
        self.is_closed = False

    def open(self, call: Call) -> Result[Any]:
        return Err(CLOSED if self.is_closed else self._error)

    def json(self, call: Call) -> Result[Any]:
        return self.open(call)

    def close(self) -> None:
        self.is_closed = True


class AsyncRefusedTransport:
    """The async twin of RefusedTransport."""

    def __init__(self, error: MeridianError) -> None:
        self._refused = RefusedTransport(error)

    @property
    def is_closed(self) -> bool:
        return self._refused.is_closed

    async def open(self, call: Call) -> Result[Any]:
        return self._refused.open(call)

    async def json(self, call: Call) -> Result[Any]:
        return self._refused.open(call)

    async def aclose(self) -> None:
        self._refused.close()
