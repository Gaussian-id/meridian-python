"""The framework-neutral forwarding route: the browser calls it, the server adds the key. Frameworks only adapt I/O.

The route is public and spends the app's credit, so it is the SDK's security boundary: the app's `authorize`
decides who may use it, only allowed fields reach the engine, and nothing about the app's own key, account or
credit is ever shown to the browser.
"""

from __future__ import annotations

import contextlib
import inspect
import json
import math
import re
from collections.abc import (
    AsyncGenerator,
    AsyncIterable,
    AsyncIterator,
    Awaitable,
    Callable,
    Generator,
    Iterable,
    Iterator,
    Sequence,
)
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field
from types import GenericAlias
from typing import TYPE_CHECKING, Any, Generic, Literal, Protocol, TypeVar, cast, runtime_checkable

import anyio  # httpx's own dependency (its async I/O), so no dependency of the SDK's own; works under asyncio and trio

from ._client import AsyncMeridian, Meridian, async_shared_client, shared_client
from ._errors import (
    NOT_JSON,
    STATUS_BY_CODE,
    MeridianError,
    Result,
    client_error,
    config_error,
    error_member,
    load_json,
    valid_request_id,
)
from ._stream import AsyncChatStream, ChatStream
from ._transport import valid_idempotency_key

if TYPE_CHECKING:
    from .types import ChatParams

# What an adapter or an app needs. The constants (APP_CREDENTIAL_CODES, GENERIC_MESSAGE, MAX_BODY_BYTES,
# MAX_BODY_DEPTH, MAX_EVENT_BYTES) stay importable for anyone who wants to read them
__all__ = [
    'DEFAULT_MAX_TOKENS_CAP',
    'AllowRule',
    'AsyncForwarder',
    'AsyncReplyBody',
    'Forwarder',
    'Reply',
    'ReplyBody',
]

R = TypeVar('R')
C = TypeVar('C')

DEFAULT_MAX_TOKENS_CAP = 4096
GENERIC_MESSAGE = 'The service is temporarily unavailable.'

# A browser's request body may be this many bytes (the engine's own request limit, and meridian-go's)
MAX_BODY_BYTES = 8 << 20

# A browser's JSON may nest containers this deep (a chat body is shallow)
MAX_BODY_DEPTH = 64

# One streamed event may be this many bytes before the stream is failed closed (the parser's own cap, in bytes)
MAX_EVENT_BYTES = 16 * 1024 * 1024

# How long closing the engine's stream may take once the relay ends (the adapters use the same bound)
CLOSE_WITHIN = 5

# The app's own key, account or credit (owner, 10 Oct): never described to the browser
APP_CREDENTIAL_CODES = frozenset(
    {
        'invalid_api_key',
        'key_revoked',
        'key_expired',
        'project_unavailable',
        'organisation_unavailable',
        'project_host_mismatch',
        'byok_required',
        'insufficient_credit',
        'hard_limit_reached',
        'entitlement_missing',
    }
)

# Fields the route sets itself or never takes: `options` can never copy them raw (max_completion_tokens would lift the
# token cap, and `n` would multiply what one request spends)
_OWNED_FIELDS = frozenset(
    {'model', 'stream', 'stream_options', 'messages', 'max_tokens', 'max_completion_tokens', 'temperature', 'n'}
)
_JSON_TYPE = re.compile(r'application/json\s*(?:;|$)', re.IGNORECASE)
_SSE_HEADERS = {'content-type': 'text/event-stream; charset=utf-8', 'cache-control': 'no-cache'}


@dataclass(frozen=True)
class AllowRule:
    """What the browser may set. `model`: True for any, a list for only those, default none (Meridian decides)."""

    model: bool | Sequence[str] = False
    max_tokens_cap: int = DEFAULT_MAX_TOKENS_CAP
    options: Sequence[str] = ()


@runtime_checkable
class ReplyBody(Protocol):
    """A streaming body for a sync framework: iterate it for the bytes, and `close()` it on every exit path."""

    def __iter__(self) -> Iterator[bytes]: ...

    def __next__(self) -> bytes: ...

    def close(self) -> None: ...


@runtime_checkable
class AsyncReplyBody(Protocol):
    """A streaming body for an async framework: `async for` it for the bytes, and `await aclose()` it on every exit
    path."""

    def __aiter__(self) -> AsyncIterator[bytes]: ...

    async def __anext__(self) -> bytes: ...

    async def aclose(self) -> None: ...


B = TypeVar('B', bound='ReplyBody | AsyncReplyBody')


@dataclass(frozen=True)
class Reply(Generic[B]):
    """What a framework adapter sends back. `passthrough` is set (and nothing else) when `authorize` answered with
    the app's own response.

    `Forwarder.handle` returns a `Reply[ReplyBody]` and `AsyncForwarder.ahandle` a `Reply[AsyncReplyBody]`: the body is
    `bytes`, or a streaming body of that kind, so `isinstance(reply.body, bytes)` is all a type checker needs. A
    streaming body holds the engine's response open from the moment the Reply exists, so the caller must close it on
    every exit path: `reply.close()` (sync) or `await reply.aclose()` (async), or the body's own `close()` /
    `aclose()`, even when it never started iterating (the browser left before the first read). Closing is idempotent,
    a finished body allows it, and closing a reply whose body is `bytes` does nothing.

    Under pyright strict, a bare local `reply = Reply(200)` is a `Reply[Unknown]` (the body type cannot be inferred
    from the defaults): annotate it (`reply: Reply[ReplyBody] = Reply(200)`) or return it straight away.
    """

    status: int
    headers: dict[str, str] = field(default_factory=lambda: {})
    body: bytes | B = b''
    passthrough: object | None = None

    def close(self) -> None:
        """Close a sync streaming body; nothing to do for any other."""
        if isinstance(self.body, ReplyBody):
            self.body.close()

    async def aclose(self) -> None:
        """Close a streaming body of either kind; nothing to do for `bytes`."""
        if isinstance(self.body, AsyncReplyBody):
            await self.body.aclose()
        else:
            self.close()


# --- Replies --------------------------------------------------------------------------------------------------------


def _dumps(value: Any) -> bytes:
    # Strict: NaN and infinity are no JSON (ValueError), and a value nested past the encoder's depth is a RecursionError
    return json.dumps(value, separators=(',', ':'), allow_nan=False).encode()


def _safe_id(value: object) -> str | None:
    """An id from the engine, fit for a header and for the body: 1-255 printable ASCII characters, so never a line
    break (header injection). Anything else is None, and the id is left out."""
    return valid_request_id(value)


def _hides(code: str, *, sdk_side: bool = False) -> bool:
    """The one decision about what a browser may not learn: a failure on the SDK's side, a code this SDK does not
    recognise (a newer engine's code could be about the app's account), or the app's own key, account or credit."""
    return sdk_side or code not in STATUS_BY_CODE or code in APP_CREDENTIAL_CODES


def _json_reply(
    status: int, value: Any, request_id: str | None = None, extra: dict[str, str] | None = None
) -> Reply[Any]:
    headers = {'content-type': 'application/json', **(extra or {})}
    if request_id := _safe_id(request_id):
        headers['x-request-id'] = request_id
    return Reply[Any](status, headers, _dumps(value))


def _generic(request_id: str | None) -> dict[str, Any]:
    error: dict[str, Any] = {
        'message': GENERIC_MESSAGE,
        'type': 'api_error',
        'code': 'service_unavailable',
        'retryable': False,
    }
    if request_id := _safe_id(request_id):
        error['request_id'] = request_id
    return {'error': error}


def _generic_frame(request_id: str | None) -> bytes:
    return b'data: ' + _dumps(_generic(request_id)) + b'\n\n'


def _browser_error(error: MeridianError) -> Reply[Any]:
    # [1] What the browser may not learn (see _hides): the generic 503, request id kept, never a Retry-After
    if _hides(error.engine_code or error.code, sdk_side=error.type == 'client_error'):
        return _json_reply(503, _generic(error.request_id), error.request_id)

    # [2] Anything about the end user's own request keeps its real status and code; a rate limit keeps the engine's
    # valid Retry-After, so the browser waits as long as the engine asked
    body: dict[str, Any] = {'message': error.message, 'type': error.type, 'code': error.code}
    if request_id := _safe_id(error.request_id):
        body['request_id'] = request_id
    wait = {'retry-after': error.retry_after} if error.code == 'rate_limited' and error.retry_after else None
    return _json_reply(error.status if error.status > 0 else 502, {'error': body}, error.request_id, wait)


def _refuse(status: int, code: str, message: str, extra: dict[str, str] | None = None) -> Reply[Any]:
    kind = 'permission_error' if code == 'permission_denied' else 'invalid_request_error'
    return _json_reply(status, {'error': {'message': message, 'type': kind, 'code': code}}, None, extra)


def _denied() -> Reply[Any]:
    return _refuse(403, 'permission_denied', 'Not allowed to use this route.')


def _unavailable() -> Reply[Any]:
    return _browser_error(config_error('The Meridian client could not be created.'))


def _unreadable() -> Reply[Any]:
    return _refuse(400, 'invalid_request', 'The body could not be read.')


# --- Request checks -------------------------------------------------------------------------------------------------


def _precheck(method: str, content_type: str | None) -> Reply[Any] | None:
    """POST and JSON only, checked before authorize: a cross-site form can send neither."""
    if method.upper() != 'POST':
        return _refuse(405, 'method_not_allowed', 'Use POST.', {'allow': 'POST'})
    if not _JSON_TYPE.match(content_type or ''):
        return _refuse(415, 'invalid_request', 'The content-type must be application/json.')
    return None


def _verdict(value: object, response_type: type | tuple[type, ...]) -> Reply[Any] | None:
    """Only the literal True proceeds. An instance of `response_type` is the app's own response, returned untouched.
    Anything else (a truthy user object, 0, '', a coroutine, ...) is a 403: it must never be taken for a yes."""
    if value is True:
        return None
    if isinstance(value, response_type):
        return Reply[Any](0, passthrough=value)
    return _denied()


def _browser_key(value: str | None) -> str | None:
    """The browser's idempotency key when the engine would accept it (the client's own rule); anything else is None,
    and the client makes a key of its own."""
    return value if valid_idempotency_key(value) else None


def _client_for(option: object, kind: type[C]) -> C | None:
    """This request's client: the instance given, or a fresh one from the factory. None when there is none to use."""
    try:
        made = option() if callable(option) else option
    except Exception:
        return None
    return made if isinstance(made, kind) else None


# --- The body and what reaches the engine ---------------------------------------------------------------------------


def _too_large() -> Reply[Any]:
    return _refuse(413, 'body_too_large', 'The request body is too large.')


def _take(gathered: bytearray, piece: object) -> Reply[Any] | None:
    """Add one chunk of the browser's body: a refusal for a chunk that is not bytes, or once the body is over
    MAX_BODY_BYTES (nothing more is read); otherwise None."""
    if not isinstance(piece, (bytes, bytearray)):
        return _unreadable()
    gathered += piece
    return _too_large() if len(gathered) > MAX_BODY_BYTES else None


def _close_quietly(*things: object) -> None:
    """Close each of these that has a `close()`, once each: best-effort, a close that raises changes nothing."""
    seen: list[object] = []
    for thing in things:
        if not any(thing is done for done in seen):
            seen.append(thing)
            with contextlib.suppress(Exception):  # a `close` that cannot even be looked up changes nothing either
                close = getattr(thing, 'close', None)
                if callable(close):
                    close()


async def _aclose_quietly(*things: object) -> None:
    """`_close_quietly` for async sources: each one's `aclose()`."""
    seen: list[object] = []
    for thing in things:
        if not any(thing is done for done in seen):
            seen.append(thing)
            with contextlib.suppress(Exception):
                aclose = getattr(thing, 'aclose', None)
                if callable(aclose):
                    await cast('Callable[[], Awaitable[object]]', aclose)()


def _read(read_body: Callable[[], object]) -> bytes | Reply[Any]:
    """The browser's body from `read_body`: its bytes, or a chunk source read up to the cap and closed (the iterable
    it returned and its iterator, whichever has a `close()`). A raise, or anything that gives no bytes, is a 400."""
    try:
        # [1] Plain bytes, already in memory
        source = read_body()
        if isinstance(source, (bytes, bytearray)):
            return _too_large() if len(source) > MAX_BODY_BYTES else bytes(source)

        # [2] Chunks, read only until the cap is passed; the source is closed whatever happens
        pieces: object = None
        gathered = bytearray()
        try:
            pieces = iter(cast('Iterable[object]', source))
            for piece in pieces:
                if refused := _take(gathered, piece):
                    return refused
        finally:
            _close_quietly(pieces, source)
        return bytes(gathered)
    except Exception:
        return _unreadable()


async def _aread(read_body: Callable[[], object]) -> bytes | Reply[Any]:
    """`_read` for async: the bytes, or an async chunk source (a sync iterable or an awaitable gives no bytes)."""
    try:
        # [1] Plain bytes, already in memory; a coroutine is no source, and is closed so it is never left un-awaited
        source = read_body()
        if isinstance(source, (bytes, bytearray)):
            return _too_large() if len(source) > MAX_BODY_BYTES else bytes(source)
        if inspect.iscoroutine(source):
            source.close()
            return _unreadable()

        # [2] Chunks, read only until the cap is passed; the source is closed whatever happens
        pieces: object = None
        gathered = bytearray()
        try:
            pieces = aiter(cast('AsyncIterable[object]', source))
            async for piece in pieces:
                if refused := _take(gathered, piece):
                    return refused
        finally:
            await _aclose_quietly(pieces, source)
        return bytes(gathered)
    except Exception:
        return _unreadable()


def _names(value: object) -> tuple[str, ...]:
    """A bare string is one name, never its letters; anything but a collection of strings names nothing."""
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (Sequence, AbstractSet)):
        return tuple(name for name in cast('Iterable[object]', value) if isinstance(name, str))
    return ()


def _cap(value: object) -> int:
    """A positive int (never a bool) is a cap; anything else falls back to the default."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return DEFAULT_MAX_TOKENS_CAP
    return value


def _number(value: object) -> int | float | None:
    return None if isinstance(value, bool) or not isinstance(value, (int, float)) else value


@dataclass(frozen=True)
class _Allowed:
    """An AllowRule made safe once, at construction."""

    cap: int
    any_model: bool
    models: tuple[str, ...]
    options: tuple[str, ...]

    @classmethod
    def of(cls, rule: AllowRule) -> _Allowed:
        options = tuple(name for name in _names(rule.options) if name not in _OWNED_FIELDS)
        return cls(_cap(rule.max_tokens_cap), rule.model is True, _names(rule.model), options)


@dataclass(frozen=True)
class _Ask:
    params: dict[str, Any]
    stream: bool
    idempotency_key: str | None


def _too_deep(body: dict[str, Any]) -> bool:
    """True when containers nest deeper than MAX_BODY_DEPTH (the body itself is level 1), counted without recursion."""
    pending: list[tuple[Any, int]] = [(body, 1)]
    while pending:
        container, level = pending.pop()
        if level > MAX_BODY_DEPTH:
            return True
        children = cast('Iterable[Any]', container.values() if isinstance(container, dict) else container)
        for child in children:
            if isinstance(child, (dict, list)):
                pending.append((cast('Any', child), level + 1))
    return False


def _parse(raw: bytes) -> dict[str, Any] | Reply[Any]:
    # [1] Strict UTF-8 JSON (no UTF-16 or UTF-32, no BOM; no NaN, Infinity or 1e999, which would not re-encode as
    # JSON). The bytes are already within MAX_BODY_BYTES (_read)
    try:
        body: Any = load_json(raw.decode('utf-8'))
    except UnicodeDecodeError:
        body = NOT_JSON
    if body is NOT_JSON:
        return _refuse(400, 'invalid_request', 'The body must be JSON.')

    # [2] An object with a non-empty message list, nested no deeper than a chat body ever is
    if not isinstance(body, dict):
        return _refuse(400, 'invalid_request', 'messages must be a non-empty list.')
    fields = cast('dict[str, Any]', body)
    if not isinstance(fields.get('messages'), list) or not fields['messages']:
        return _refuse(400, 'invalid_request', 'messages must be a non-empty list.')
    if _too_deep(fields):
        return _refuse(400, 'invalid_request', 'The body is nested too deeply.')
    return fields


def _params(body: dict[str, Any], allowed: _Allowed) -> dict[str, Any]:
    # [1] Messages always; temperature when it is a number; max_tokens clamped to [1, cap], and the cap when absent
    params: dict[str, Any] = {'messages': body['messages']}
    temperature = _number(body.get('temperature'))
    if temperature is not None:
        params['temperature'] = temperature
    tokens = _number(body.get('max_tokens'))
    params['max_tokens'] = math.floor(min(max(1, tokens), allowed.cap)) if tokens is not None else allowed.cap

    # [2] A model only when the app allows it
    model = body.get('model')
    if isinstance(model, str) and (allowed.any_model or model in allowed.models):
        params['model'] = model

    # [3] Further options only by name and only when present (the owned fields were struck from the rule)
    for name in allowed.options:
        if name in body:
            params[name] = body[name]
    return params


def _ask(raw: bytes, allowed: _Allowed, idempotency_key: str | None) -> _Ask | Reply[Any]:
    body = _parse(raw)
    if isinstance(body, Reply):
        return body
    return _Ask(_params(body, allowed), body.get('stream') is True, _browser_key(idempotency_key))


# --- The stream guard -----------------------------------------------------------------------------------------------

# A blank line in any SSE line ending. A CR counts alone only when no LF follows, so a CRLF is never read as two
_LINE_END = rb'(?:\r\n|\r(?!\n)|\n)'
_BLANK_LINE = re.compile(_LINE_END + _LINE_END)
_LINE_BREAK = re.compile(r'\r\n|\r|\n')
_BOM = chr(0xFEFF)  # a byte order mark opening the stream is not part of the first event
_BACK_OFF = 3  # a blank line is at most 4 bytes: the last 3 are searched again when the next piece arrives
_Verdict = Literal['pass', 'hide', 'terminal']
_Judgement = Literal['pass', 'hide', 'terminal', 'no_data', 'unjudgeable']
_NUMBER = object()  # stands for any number in a judged event: its value never matters, but it is never null


def _skip(_: str) -> object:
    return _NUMBER


def _carried_id(error: Any) -> str | None:
    """The request id an `error` value carries (only an object can), if it is fit to be shown."""
    return _safe_id(cast('dict[str, Any]', error).get('request_id')) if isinstance(error, dict) else None


def _error_verdict(value: Any) -> tuple[_Verdict, str | None]:
    """What a parsed reply or event says through an `error` key, with the request id its envelope carries, if any:
    nothing (pass); an envelope about the app's own key, account or credit, or with a code this SDK does not know
    (hide); or a key that is set but is no object with a usable code (terminal)."""
    # [1] No error member, or a null one, is an ordinary reply (the engine builds its chunks and completions without
    # one); content below the top level (such as choices[].delta) is never inspected
    member = error_member(value)
    if member is None:
        return 'pass', None

    # [2] A member that is set must be an object with a non-empty string code, and no member the guard reads may come
    # in a case variant (error_member: MALFORMED); anything else cannot be read: terminal
    carried = _carried_id(cast('dict[str, Any]', value).get('error'))
    code = cast('dict[str, Any]', member).get('code') if isinstance(member, dict) else None
    if not (isinstance(code, str) and code):
        return 'terminal', carried

    # [3] A code this SDK knows that is not about the app's own key, account or credit passes as it is
    return ('hide', carried) if _hides(code) else ('pass', None)


def _judge(event: bytes, first: bool) -> tuple[_Judgement, str | None]:
    """One event, as a browser would read it: passable, hidden or terminal (each with its request id), without any
    data (comments and ids only), or unjudgeable."""
    # [1] Too large to read; otherwise the lines in any line ending, the stream's BOM ignored
    if len(event) > MAX_EVENT_BYTES:
        return 'unjudgeable', None
    text = event.decode('utf-8', 'replace')
    if first:
        text = text.removeprefix(_BOM)

    # [2] Only SSE fields the guard understands: data, a message event, id, retry and comments. Anything else (another
    # event name, an unknown field, a body that is not SSE at all) cannot be judged
    data: list[str] = []
    for line in _LINE_BREAK.split(text):
        if not line or line.startswith(':'):
            continue
        name, _, value = line.partition(':')
        value = value.removeprefix(' ')  # a value's one leading space is dropped
        if name == 'data':
            data.append(value)
        elif name == 'event':
            if value not in ('', 'message'):
                return 'terminal', None
        elif name not in ('id', 'retry'):
            return 'terminal', None
    if not data:
        return 'no_data', None
    if data == ['[DONE]']:
        return 'pass', None

    # [3] The data must be a JSON object (not JSON, NaN or Infinity, or too deep to parse, is terminal). Numbers are
    # never converted, so no number can make a frame unreadable here yet readable in a browser
    parsed: Any = load_json('\n'.join(data), parse_int=_skip, parse_float=_skip)
    if not isinstance(parsed, dict):
        return 'terminal', None

    # [4] An error key decides: ordinary, hidden, or terminal
    return _error_verdict(parsed)


class _Guard:
    """Judges the engine's events before any byte reaches the browser; the same logic for sync and async.

    Only whole events go out: as they came, or the generic frame for a hidden code. An event that cannot be judged
    (too large, too deep, or cut off; an event name other than message; data that is not a JSON object; an unknown
    field) or whose `error` key is no envelope with a code ends the stream closed instead: `closed` is set and nothing
    more may be read. Comment, `id` and `retry` lines are allowed and forwarded as they came; a body that ends with
    content holding no `data` line is not SSE and also ends closed.
    """

    def __init__(self, request_id: str | None) -> None:
        self._request_id = _safe_id(request_id)  # the response header's id, for a frame that carries none
        self._buffer = bytearray()
        self._searched = 0
        self._first = True
        self.closed = False

    def feed(self, piece: bytes) -> bytes:
        # [1] Cut every complete event, searching only what is new (plus the few bytes a blank line may straddle)
        buffer = self._buffer
        buffer += piece
        out = bytearray()
        begin = 0
        while (match := _BLANK_LINE.search(buffer, self._searched)) is not None:
            event = bytes(buffer[begin : match.end()])
            begin = self._searched = match.end()

            # [2] Each one judged whole: an unjudgeable or terminal one fails the stream closed
            verdict, request_id = _judge(event, self._first)
            self._first = False
            if verdict in ('unjudgeable', 'terminal'):
                return bytes(out) + self.fail(request_id)
            out += _generic_frame(request_id or self._request_id) if verdict == 'hide' else event
        del buffer[:begin]

        # [3] What is left is one unfinished event: held back unjudged, and never past the cap
        if len(buffer) > MAX_EVENT_BYTES:
            return bytes(out) + self.fail()
        self._searched = max(0, len(buffer) - _BACK_OFF)
        return bytes(out)

    def end(self) -> bytes:
        # [1] The engine finished: a last event without its blank line is judged as it is
        rest = bytes(self._buffer)
        self._buffer.clear()
        if not rest.strip(b'\r\n'):
            return rest
        verdict, request_id = _judge(rest, self._first)
        if verdict == 'pass':
            return rest

        # [2] A hidden one is replaced; one that cannot be read (cut off, unjudgeable, terminal, or with no data line
        # at the end of a body, such as a body that is not SSE) fails closed
        return _generic_frame(request_id or self._request_id) if verdict == 'hide' else self.fail(request_id)

    def fail(self, request_id: str | None = None) -> bytes:
        """Drop the unjudged bytes and end the stream with the generic frame (carrying `request_id`, else the
        response's), after a blank line closing any event."""
        self._buffer.clear()
        self.closed = True
        return b'\n\n' + _generic_frame(request_id or self._request_id)


def _relay(stream: ChatStream) -> Generator[bytes, None, None]:
    guard = _Guard(stream.request_id)
    upstream = stream.iter_bytes()
    try:
        # [1] Every piece through the guard; a stream failed closed is closed upstream before its last frame goes out
        for piece in upstream:
            out = guard.feed(piece)
            if guard.closed:
                upstream.close()
            if out:
                yield out
            if guard.closed:
                return

        # [2] A body that broke mid-stream ends with the generic frame, so no client takes the cut for a success
        out = guard.fail() if stream.error is not None else guard.end()
        if out:
            yield out
    finally:
        # [3] Finished, failed, or abandoned by the browser: upstream is closed
        upstream.close()


async def _arelay(stream: AsyncChatStream) -> AsyncGenerator[bytes, None]:
    guard = _Guard(stream.request_id)
    upstream = stream.aiter_bytes()
    try:
        # [1] Every piece through the guard; a stream failed closed is closed upstream before its last frame goes out
        async for piece in upstream:
            out = guard.feed(piece)
            if guard.closed:
                await upstream.aclose()
            if out:
                yield out
            if guard.closed:
                return

        # [2] A body that broke mid-stream ends with the generic frame, so no client takes the cut for a success
        out = guard.fail() if stream.error is not None else guard.end()
        if out:
            yield out
    finally:
        # [3] Finished, failed, cancelled, or abandoned by the browser: upstream is closed, shielded (a cancelled scope
        #     would stop the close at its first checkpoint) and bounded (a close that hangs is given up)
        with anyio.move_on_after(CLOSE_WITHIN, shield=True):
            await upstream.aclose()


class _Body:
    """The relay as a `ReplyBody` the caller can always close. The engine's response is open before the first read, and
    a generator closed before it started never runs its `finally`, so the unstarted case is closed here instead."""

    def __init__(self, stream: ChatStream) -> None:
        self._stream = stream
        self._relay: Generator[bytes, None, None] | None = None
        self._closed = False

    def __iter__(self) -> Iterator[bytes]:
        return self

    def __next__(self) -> bytes:
        if self._closed:
            raise StopIteration
        if self._relay is None:
            self._relay = _relay(self._stream)
        return next(self._relay)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._relay is not None:
            try:
                self._relay.close()  # its `finally` closes upstream
                return
            except (RuntimeError, ValueError):
                pass  # a read is running in another thread, so the relay cannot be closed from here
        self._stream.cancel()  # never started, or mid-read: close upstream directly


class _AsyncBody:
    """`_Body` as an `AsyncReplyBody`: `aclose()` is safe at any point and any number of times."""

    def __init__(self, stream: AsyncChatStream) -> None:
        self._stream = stream
        self._relay: AsyncGenerator[bytes, None] | None = None
        self._closed = False

    def __aiter__(self) -> AsyncIterator[bytes]:
        return self

    async def __anext__(self) -> bytes:
        if self._closed:
            raise StopAsyncIteration
        if self._relay is None:
            self._relay = _arelay(self._stream)
        return await anext(self._relay)

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._relay is not None:
            try:
                await self._relay.aclose()  # its `finally` closes upstream
                return
            except (RuntimeError, ValueError):
                pass  # a read is waiting in another task, so the relay cannot be closed from here
        await self._stream.cancel()  # never started, or mid-read: close upstream directly


def _streaming(request_id: str | None, body: ReplyBody | AsyncReplyBody) -> Reply[Any]:
    # Headers are the route's own: the engine's content-encoding and content-length never describe these bytes
    headers = dict(_SSE_HEADERS)
    if request_id := _safe_id(request_id):
        headers['x-request-id'] = request_id
    return Reply[Any](200, headers, body)


def _buffered(result: Result[Any]) -> Reply[Any]:
    # A 200 that carries an error member is no result: the client already made it a network_error (keeping the error's
    # own valid request id), which is the generic 503 here like every SDK-side failure
    if result.error is not None:
        return _browser_error(result.error)

    # A reply the client decoded may still not encode again for the browser (nested past the encoder's depth on a
    # Python whose decoder allows more, or holding a NaN): it fails like any SDK-side failure, keeping the reply's id
    try:
        return _json_reply(200, result.data, result.request_id)
    except (ValueError, RecursionError):
        return _browser_error(client_error('network_error', 'The reply could not be encoded.', 200, result.request_id))


# --- The two helpers ------------------------------------------------------------------------------------------------


def _require(authorize: object) -> None:
    if not callable(authorize):
        raise TypeError(
            'A forwarding helper needs an authorize(request) function: it decides who may spend your credit.'
        )


# Verdicts that are never a response: a response_type that takes any of them for one is refused
_NOT_RESPONSES = (True, False, 0, 1, None)


def _takes_a_verdict(kinds: tuple[type, ...]) -> bool:
    try:
        return any(isinstance(probe, kinds) for probe in _NOT_RESPONSES)
    except Exception:  # a class whose instance check raises cannot be trusted to tell a response either
        return True


def _response_types(value: object) -> type | tuple[type, ...]:
    """`response_type` as given, once checked: a class, or a non-empty tuple of classes (never a generic alias such as
    `list[int]`, which 3.10 still calls a class), none of which takes a verdict that is no response (`True`, `False`,
    `0`, `1`, `None`) for the app's own response: so never `object`, `bool`, `int`, `numbers.Number`, `Hashable`, ...
    Like a missing `authorize`, a bad one is a mistake in the app's set-up and raises `TypeError` at construction."""
    kinds = cast('tuple[object, ...]', value) if isinstance(value, tuple) else (value,)
    classes = tuple(kind for kind in kinds if isinstance(kind, type) and not isinstance(kind, GenericAlias))
    if not kinds or len(classes) != len(kinds) or _takes_a_verdict(classes):
        raise TypeError(
            'response_type must be a class, or a non-empty tuple of classes, that no verdict such as True, False, 0, '
            "1 or None is an instance of (so not object, bool or int): it says what counts as the app's own response."
        )
    return cast('type | tuple[type, ...]', value)


class Forwarder(Generic[R]):
    """The forwarding route for a sync framework.

    `client` is a Meridian, or a function returning one. The function is called for every request, never at creation,
    and the forwarder never closes a client, given or returned: the function should return one long-lived shared
    client, built once and closed by the app at shutdown, never a new client per request (each one would hold its own
    connections open). Without a `client`, the forwarder uses one shared client per `MERIDIAN_URL` / `MERIDIAN_KEY`,
    read at each request and built on first use; settings that cannot work are a generic 503 and are never kept, so
    an environment fixed after start-up is picked up.

    `authorize(request)` decides who may spend the app's credit: only the literal `True` proceeds. An instance of
    `response_type` (a `Reply` by default; an adapter passes its framework's response class) is the app's own
    response, returned untouched as `Reply.passthrough`. Anything else, or a raise, is a 403 and the engine is
    never called.
    """

    def __init__(
        self,
        *,
        client: Meridian | Callable[[], Meridian] | None = None,
        authorize: Callable[[R], object],
        allow: AllowRule | None = None,
        response_type: type | tuple[type, ...] = Reply,
    ) -> None:
        _require(authorize)
        self._client = shared_client if client is None else client
        self._authorize, self._allowed = authorize, _Allowed.of(allow or AllowRule())
        self._response_type = _response_types(response_type)

    def _decide(self, request: R) -> Reply[Any] | None:
        # A raising authorize is a refusal, never a crash and never a way through
        try:
            outcome = self._authorize(request)

            # An async authorize given to the sync helper is never awaited here, so it is never a yes
            if inspect.iscoroutine(outcome):
                outcome.close()
            return _verdict(outcome, self._response_type)
        except Exception:
            return _denied()

    def handle(
        self,
        request: R,
        *,
        method: str,
        content_type: str | None,
        idempotency_key: str | None,
        read_body: Callable[[], bytes | bytearray | Iterable[bytes]],
    ) -> Reply[ReplyBody]:
        """Answer one browser request. `read_body` is called only after `authorize` said yes. It returns the body's
        chunks (any iterable of bytes, such as a generator over the framework's input stream), which are read only up
        to MAX_BODY_BYTES: past it the answer is 413, nothing more is read, and the source is closed (the iterable
        returned and its iterator, each that has a `close()`; a close that raises is ignored). Plain bytes already in
        memory are fine too. An adapter should pass a generator over `stream.read(65536)`, not the file-like object
        itself: iterating a file by lines can deliver the whole body as one huge chunk before the cap is checked.
        A streaming body must be closed by the caller (see Reply)."""
        # [1] POST and JSON only, then the app's own decision
        refused = _precheck(method, content_type) or self._decide(request)
        if refused is not None:
            return refused

        # [2] A valid body within the cap, of which only the allowed fields go on
        raw = _read(read_body)
        if isinstance(raw, Reply):
            return raw
        ask = _ask(raw, self._allowed, idempotency_key)
        if isinstance(ask, Reply):
            return ask

        # [3] The client is built now, for this request only
        client = _client_for(self._client, Meridian)
        if client is None:
            return _unavailable()

        # [4] Stream or buffer; every refusal is translated for the browser
        params = cast('ChatParams', ask.params)
        if ask.stream:
            opened = client.chat.stream(params, idempotency_key=ask.idempotency_key)
            if opened.error is not None:
                return _browser_error(opened.error)
            return _streaming(opened.data.request_id, _Body(opened.data))
        return _buffered(client.chat.create(params, idempotency_key=ask.idempotency_key))


class AsyncForwarder(Generic[R]):
    """Forwarder for an async framework; see `Forwarder`. `client` is an AsyncMeridian or a function returning one
    (the function should return one long-lived shared client, closed by the app at shutdown; the forwarder never closes
    it); without one, a shared client per settings and per running asyncio event loop is used. Under another async
    runtime (trio) there is no asyncio loop, so that default builds a new client for every request and nothing closes
    it: pass `client=` there. `authorize` may be async: it is awaited once, and an awaitable it returns is a 403, not
    a yes."""

    def __init__(
        self,
        *,
        client: AsyncMeridian | Callable[[], AsyncMeridian] | None = None,
        authorize: Callable[[R], object],
        allow: AllowRule | None = None,
        response_type: type | tuple[type, ...] = Reply,
    ) -> None:
        _require(authorize)
        self._client = async_shared_client if client is None else client
        self._authorize, self._allowed = authorize, _Allowed.of(allow or AllowRule())
        self._response_type = _response_types(response_type)

    async def _decide(self, request: R) -> Reply[Any] | None:
        # A raising or rejecting authorize is a refusal, never a crash and never a way through
        try:
            outcome = self._authorize(request)
            if inspect.isawaitable(outcome):
                outcome = await outcome

            # An awaitable answered by the awaited one is no response either: closed, so it is never left un-awaited
            if inspect.iscoroutine(outcome):
                outcome.close()
            return _verdict(outcome, self._response_type)
        except Exception:
            return _denied()

    async def ahandle(
        self,
        request: R,
        *,
        method: str,
        content_type: str | None,
        idempotency_key: str | None,
        read_body: Callable[[], bytes | bytearray | AsyncIterable[bytes]],
    ) -> Reply[AsyncReplyBody]:
        """`Forwarder.handle` for async. `read_body` returns the body's chunks as an async iterable (such as the
        framework's request stream), read only up to MAX_BODY_BYTES, or plain bytes; a coroutine or a sync iterable
        gives no body (400). The iterable returned and its iterator are closed with `aclose()` (each that has one).
        Give it an async generator over `stream.read(65536)` rather than a file-like object iterated by lines, which
        can deliver the whole body as one huge chunk. A streaming body must be closed with `await reply.aclose()` on
        every exit path."""
        # [1] POST and JSON only, then the app's own decision
        refused = _precheck(method, content_type) or await self._decide(request)
        if refused is not None:
            return refused

        # [2] A valid body within the cap, of which only the allowed fields go on
        raw = await _aread(read_body)
        if isinstance(raw, Reply):
            return raw
        ask = _ask(raw, self._allowed, idempotency_key)
        if isinstance(ask, Reply):
            return ask

        # [3] The client is built now, for this request only
        client = _client_for(self._client, AsyncMeridian)
        if client is None:
            return _unavailable()

        # [4] Stream or buffer; every refusal is translated for the browser
        params = cast('ChatParams', ask.params)
        if ask.stream:
            opened = await client.chat.stream(params, idempotency_key=ask.idempotency_key)
            if opened.error is not None:
                return _browser_error(opened.error)
            return _streaming(opened.data.request_id, _AsyncBody(opened.data))
        return _buffered(await client.chat.create(params, idempotency_key=ask.idempotency_key))
