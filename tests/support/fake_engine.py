"""An in-memory Meridian engine for httpx.MockTransport. It speaks the engine's wire shapes."""

from __future__ import annotations

import asyncio
import itertools
import json
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from typing import Any

import httpx

FAKE_PROJECT_KEY = 'gmr_' + '0' * 64
FAKE_INTEGRATOR_KEY = 'gmi_' + '1' * 64
PUBLIC_URL = 'http://engine.test'
ADMIN_URL = 'http://admin.engine.test'

STATUS = {
    'invalid_request': 400,
    'invalid_api_key': 401,
    'key_revoked': 401,
    'key_expired': 401,
    'insufficient_credit': 402,
    'permission_denied': 403,
    'model_not_allowed': 403,
    'project_unavailable': 403,
    'organisation_unavailable': 403,
    'byok_required': 403,
    'project_host_mismatch': 403,
    'entitlement_missing': 403,
    'not_found': 404,
    'method_not_allowed': 405,
    'rate_limited': 429,
    'hard_limit_reached': 429,
    'internal': 500,
    'upstream_unavailable': 502,
    'upstream_error': 502,
    'service_unavailable': 503,
    'deadline_exceeded': 504,
}
TYPE = {
    'invalid_request': 'invalid_request_error',
    'invalid_api_key': 'authentication_error',
    'key_revoked': 'authentication_error',
    'key_expired': 'authentication_error',
    'insufficient_credit': 'rate_limit_error',
    'rate_limited': 'rate_limit_error',
    'hard_limit_reached': 'rate_limit_error',
    'permission_denied': 'permission_error',
    'model_not_allowed': 'permission_error',
    'project_unavailable': 'permission_error',
    'organisation_unavailable': 'permission_error',
    'byok_required': 'permission_error',
    'project_host_mismatch': 'permission_error',
    'entitlement_missing': 'permission_error',
    'not_found': 'invalid_request_error',
    'method_not_allowed': 'invalid_request_error',
}
_ids = itertools.count(1)


@dataclass
class Scripted:
    status: int = 200
    headers: dict[str, str] = field(default_factory=lambda: {})
    json: Any = None
    text: str | None = None
    chunks: list[bytes | str] | None = None
    drop_after: int | None = None  # raise ReadError after this many chunks
    fail_after: int | None = None  # raise a RuntimeError (not an httpx error) after this many chunks
    hang: bool = False  # async only: the body never finishes
    network_error: bool = False
    timeout: bool = False


def envelope(code: str, request_id: str | None = None) -> dict[str, Any]:
    rid = request_id or f'req_fake_{next(_ids)}'
    return {'error': {'message': f'fake {code}', 'type': TYPE.get(code, 'api_error'), 'code': code, 'request_id': rid}}


def engine_error(code: str, headers: dict[str, str] | None = None) -> Scripted:
    body = envelope(code)
    return Scripted(
        status=STATUS.get(code, 500),
        headers={'x-request-id': body['error']['request_id'], **(headers or {})},
        json=body,
    )


def frame(value: Any) -> str:
    return f'data: {value if isinstance(value, str) else json.dumps(value)}\n\n'


def chunk(content: str, model: str = 'gemini-3.8-flash') -> dict[str, Any]:
    return {
        'id': 'chatcmpl-fake',
        'object': 'chat.completion.chunk',
        'created': 0,
        'model': model,
        'choices': [{'index': 0, 'delta': {'content': content}, 'finish_reason': None}],
    }


def usage_chunk(prompt: int = 3, completion: int = 2) -> dict[str, Any]:
    return {
        'id': 'chatcmpl-fake',
        'object': 'chat.completion.chunk',
        'created': 0,
        'model': 'gemini-3.8-flash',
        'choices': [],
        'usage': {'prompt_tokens': prompt, 'completion_tokens': completion, 'total_tokens': prompt + completion},
    }


def completion(content: str = 'hello', model: str = 'gemini-3.8-flash', **usage: Any) -> dict[str, Any]:
    return {
        'id': 'chatcmpl-fake',
        'object': 'chat.completion',
        'created': 0,
        'model': model,
        'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': content}, 'finish_reason': 'stop'}],
        'usage': usage or {'prompt_tokens': 3, 'completion_tokens': 1, 'total_tokens': 4},
    }


class _Body(httpx.SyncByteStream, httpx.AsyncByteStream):
    def __init__(
        self,
        parts: list[bytes],
        drop_after: int | None,
        hang: bool,
        fail_after: int | None = None,
        hanging: asyncio.Event | None = None,
    ) -> None:
        self._parts, self._drop_after, self._hang, self._fail_after = parts, drop_after, hang, fail_after
        self._hanging = hanging  # set once a hanging body waits for a part that never comes
        self.closed = False  # the upstream connection was closed by the client
        self.read = 0  # parts handed to the client so far

    def __iter__(self) -> Iterator[bytes]:
        for i, part in enumerate(self._parts):
            if self._drop_after is not None and i >= self._drop_after:
                raise httpx.ReadError('connection dropped')
            if self._fail_after is not None and i >= self._fail_after:
                raise RuntimeError('boom')
            self.read += 1
            yield part

    def close(self) -> None:
        self.closed = True

    async def aclose(self) -> None:
        self.closed = True

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for i, part in enumerate(self._parts):
            if self._drop_after is not None and i >= self._drop_after:
                raise httpx.ReadError('connection dropped')
            if self._fail_after is not None and i >= self._fail_after:
                raise RuntimeError('boom')
            self.read += 1
            yield part
        if self._hang:
            if self._hanging is not None:
                self._hanging.set()
            await asyncio.sleep(3600)


class FakeEngine:
    public_url = PUBLIC_URL
    admin_url = ADMIN_URL
    project_key = FAKE_PROJECT_KEY
    integrator_key = FAKE_INTEGRATOR_KEY
    is_fake = True

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self._queue: list[Scripted] = []
        self._bodies: list[_Body] = []
        # Set when a hanging stream (Scripted.hang) has handed over its parts and waits: a read is now blocked in it
        self.hanging = asyncio.Event()
        self.transport = httpx.MockTransport(self._handle)

    def next(self, *answers: Scripted) -> None:
        self._queue.extend(answers)

    def reset(self) -> None:
        """Forget every recorded request, scripted answer and streamed body, as if the engine had just started."""
        self.requests.clear()
        self._queue.clear()
        self._bodies.clear()

    def upstream_closed(self) -> bool:
        """True when the client has closed every streamed response it opened (and opened at least one)."""
        return bool(self._bodies) and all(body.closed for body in self._bodies)

    def parts_read(self) -> int:
        """How many scripted stream parts the client has read, across every streamed response."""
        return sum(body.read for body in self._bodies)

    def bodies(self) -> list[Any]:
        return [json.loads(r.content) if r.content else None for r in self.requests]

    def _handle(self, request: httpx.Request) -> httpx.Response:
        # [1] Record, then answer with the next scripted reply (a 200 completion by default)
        request.read()
        self.requests.append(request)
        answer = self._queue.pop(0) if self._queue else Scripted(json=completion())
        if answer.network_error:
            raise httpx.ConnectError('fake network failure', request=request)
        if answer.timeout:
            raise httpx.ReadTimeout('fake timeout', request=request)

        # [2] Streamed, JSON, text, or empty body
        headers = dict(answer.headers)
        if answer.chunks is not None:
            parts = [c.encode() if isinstance(c, str) else c for c in answer.chunks]
            headers.setdefault('content-type', 'text/event-stream')
            body = _Body(parts, answer.drop_after, answer.hang, answer.fail_after, self.hanging)
            self._bodies.append(body)
            return httpx.Response(answer.status, headers=headers, stream=body, request=request)
        if answer.json is not None:
            headers.setdefault('content-type', 'application/json')
            return httpx.Response(
                answer.status, headers=headers, content=json.dumps(answer.json).encode(), request=request
            )
        return httpx.Response(answer.status, headers=headers, content=(answer.text or '').encode(), request=request)
