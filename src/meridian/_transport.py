from __future__ import annotations

import json
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import httpx

from ._config import DEFAULT_MAX_RETRIES, DEFAULT_TIMEOUT, sane_retries, sane_timeout
from ._errors import Err, MeridianError, Ok, Result, client_error, config_error, error_from_body
from ._version import __version__

_NO_BODY: Any = object()

__all__ = ['DEFAULT_MAX_RETRIES', 'DEFAULT_TIMEOUT', 'Call', 'RefusedTransport', 'SyncTransport', 'new_idempotency_key']


@dataclass(frozen=True)
class Call:
    method: str
    path: str
    query: Mapping[str, str | int | None] | None = None
    body: Any = _NO_BODY
    headers: Mapping[str, str] = field(default_factory=lambda: {})
    retry: bool = True  # False: not safe to repeat once it may have reached the server


def new_idempotency_key() -> str:
    return str(uuid.uuid4())


class _Shared:
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
        if isinstance(exc, httpx.TimeoutException):
            return client_error('network_error', f'No response within {self.timeout:g} s.')
        return client_error('network_error', 'The network request failed.')


def _decode_json(response: httpx.Response, text: str) -> Result[Any]:
    try:
        return Ok(json.loads(text))
    except ValueError:
        return Err(client_error('network_error', 'The response was not valid JSON.', response.status_code))


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
        except httpx.HTTPError as exc:
            return None, self._shared.failure(exc), None
        except (httpx.InvalidURL, UnicodeError) as exc:  # a URL or header httpx cannot encode: our setup, not the wire
            return None, config_error(f'The request could not be built: {exc}'), None
        if response.is_success:
            return response, None, None

        # [2] An error status: read its body once, then close
        try:
            text = response.read().decode('utf-8', 'replace')
        except httpx.HTTPError:
            text = ''
        finally:
            response.close()
        error = error_from_body(response.status_code, response.headers.get('x-request-id'), text)
        return None, error, response.headers.get('retry-after')

    def open(self, call: Call) -> Result[httpx.Response]:
        response, error, _ = self._attempt(call)
        if response is not None:
            return Ok(response)
        assert error is not None
        return Err(error)

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
                return Err(
                    client_error(
                        'network_error', 'The connection dropped while reading the response body.', response.status_code
                    )
                )
            return _decode_json(response, text)
        finally:
            response.close()

    def close(self) -> None:
        if self._owns:
            self._client.close()


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
