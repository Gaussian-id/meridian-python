"""The framework-neutral forwarding route: the browser calls it, the server adds the key. Frameworks only adapt I/O.

The route is public and spends the app's credit, so it is the SDK's security boundary: the app's `authorize`
decides who may use it, only allowed fields reach the engine, and nothing about the app's own key, account or
credit is ever shown to the browser.
"""

from __future__ import annotations

import inspect
import json
import math
import re
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Iterator, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, Literal, TypeVar, cast

from ._client import AsyncMeridian, Meridian
from ._errors import MeridianError, Result, config_error
from ._stream import AsyncChatStream, ChatStream

if TYPE_CHECKING:
    from .types import ChatParams

R = TypeVar('R')
C = TypeVar('C')

DEFAULT_MAX_TOKENS_CAP = 4096
GENERIC_MESSAGE = 'The service is temporarily unavailable.'

# One streamed event may be this many bytes before the stream is failed closed (the parser's own cap, in bytes)
MAX_EVENT_BYTES = 16 * 1024 * 1024

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

# Fields the route sets itself: `options` can never copy them raw (max_completion_tokens would lift the token cap)
_OWNED_FIELDS = frozenset(
    {'model', 'stream', 'stream_options', 'messages', 'max_tokens', 'max_completion_tokens', 'temperature'}
)
_JSON_TYPE = re.compile(r'application/json\s*(?:;|$)', re.IGNORECASE)
_BROWSER_KEY = re.compile(r'[ -~]{1,255}')
_SSE_HEADERS = {'content-type': 'text/event-stream; charset=utf-8', 'cache-control': 'no-cache'}


@dataclass(frozen=True)
class AllowRule:
    """What the browser may set. `model`: True for any, a list for only those, default none (Meridian decides)."""

    model: bool | Sequence[str] = False
    max_tokens_cap: int = DEFAULT_MAX_TOKENS_CAP
    options: Sequence[str] = ()


@dataclass(frozen=True)
class Reply:
    """What a framework adapter sends back. `passthrough` is set (and nothing else) when `authorize` answered with
    the app's own response."""

    status: int
    headers: dict[str, str] = field(default_factory=lambda: {})
    body: bytes | Iterator[bytes] | AsyncIterator[bytes] = b''
    passthrough: object | None = None


# --- Replies --------------------------------------------------------------------------------------------------------


def _dumps(value: Any) -> bytes:
    return json.dumps(value, separators=(',', ':')).encode()


def _json_reply(status: int, value: Any, request_id: str | None = None, extra: dict[str, str] | None = None) -> Reply:
    headers = {'content-type': 'application/json', **(extra or {})}
    if request_id:
        headers['x-request-id'] = request_id
    return Reply(status, headers, _dumps(value))


def _generic(request_id: str | None) -> dict[str, Any]:
    error: dict[str, Any] = {
        'message': GENERIC_MESSAGE,
        'type': 'api_error',
        'code': 'service_unavailable',
        'retryable': False,
    }
    if request_id:
        error['request_id'] = request_id
    return {'error': error}


def _generic_frame(request_id: str | None) -> bytes:
    return b'data: ' + _dumps(_generic(request_id)) + b'\n\n'


def _browser_error(error: MeridianError) -> Reply:
    # [1] The app's own key, account or credit, and any failure on the SDK's side: the generic 503, request id kept
    if error.code in APP_CREDENTIAL_CODES or error.type == 'client_error':
        return _json_reply(503, _generic(error.request_id), error.request_id)

    # [2] Anything about the end user's own request keeps its real status and code
    body: dict[str, Any] = {'message': error.message, 'type': error.type, 'code': error.code}
    if error.request_id:
        body['request_id'] = error.request_id
    return _json_reply(error.status if error.status > 0 else 502, {'error': body}, error.request_id)


def _refuse(status: int, code: str, message: str, extra: dict[str, str] | None = None) -> Reply:
    kind = 'permission_error' if code == 'permission_denied' else 'invalid_request_error'
    return _json_reply(status, {'error': {'message': message, 'type': kind, 'code': code}}, None, extra)


def _denied() -> Reply:
    return _refuse(403, 'permission_denied', 'Not allowed to use this route.')


def _unavailable() -> Reply:
    return _browser_error(config_error('The Meridian client could not be created.'))


def _unreadable() -> Reply:
    return _refuse(400, 'invalid_request', 'The body could not be read.')


# --- Request checks -------------------------------------------------------------------------------------------------


def _precheck(method: str, content_type: str | None) -> Reply | None:
    """POST and JSON only, checked before authorize: a cross-site form can send neither."""
    if method.upper() != 'POST':
        return _refuse(405, 'method_not_allowed', 'Use POST.', {'allow': 'POST'})
    if not _JSON_TYPE.match(content_type or ''):
        return _refuse(415, 'invalid_request', 'The content-type must be application/json.')
    return None


def _verdict(value: object) -> Reply | None:
    """True proceeds; False or None is a 403; any other value is the app's own response, returned untouched."""
    if value is True:
        return None
    if value is False or value is None:
        return _denied()
    return Reply(0, passthrough=value)


def _browser_key(value: str | None) -> str | None:
    """The browser's idempotency key: 1-255 printable ASCII characters once trimmed of spaces (a header value cannot
    start or end with one). Anything else is None, and the SDK makes a key of its own."""
    key = (value or '').strip(' ')
    return key if _BROWSER_KEY.fullmatch(key) else None


def _client_for(option: object, kind: type[C]) -> C | None:
    """This request's client: the instance given, or a fresh one from the factory. None when there is none to use."""
    try:
        made = option() if callable(option) else option
    except Exception:
        return None
    return made if isinstance(made, kind) else None


# --- The body and what reaches the engine ---------------------------------------------------------------------------


def _not_json(_: str) -> float:
    raise ValueError('not JSON')


def _finite(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise ValueError('a number too large for JSON')
    return value


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


def _parse(raw: bytes) -> dict[str, Any] | Reply:
    # [1] Strict JSON (no NaN or Infinity, which would not re-encode as JSON), never deeper than the parser allows
    try:
        body: Any = json.loads(raw, parse_constant=_not_json, parse_float=_finite)
    except (ValueError, RecursionError):
        return _refuse(400, 'invalid_request', 'The body must be JSON.')

    # [2] An object with a non-empty message list
    if not isinstance(body, dict):
        return _refuse(400, 'invalid_request', 'messages must be a non-empty list.')
    fields = cast('dict[str, Any]', body)
    if not isinstance(fields.get('messages'), list) or not fields['messages']:
        return _refuse(400, 'invalid_request', 'messages must be a non-empty list.')
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


def _ask(raw: bytes, allowed: _Allowed, idempotency_key: str | None) -> _Ask | Reply:
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
_Judgement = Literal['pass', 'hide', 'not_json', 'unjudgeable']


def _skip(_: str) -> None:
    return None


def _judge(event: bytes, first: bool) -> tuple[_Judgement, str | None]:
    """One event, as a browser would read it: hidden (with its request id), passable, not JSON, or unjudgeable."""
    # [1] The data lines in any line ending, a value's one leading space dropped, the stream's BOM ignored
    if len(event) > MAX_EVENT_BYTES:
        return 'unjudgeable', None
    text = event.decode('utf-8', 'replace')
    if first:
        text = text.removeprefix(_BOM)
    lines = [line[5:].removeprefix(' ') for line in _LINE_BREAK.split(text) if line[:5] in ('data', 'data:')]
    if not lines or lines == ['[DONE]']:
        return 'pass', None

    # [2] Numbers are never converted, so no number can make a frame unreadable here yet readable in a browser
    try:
        value: Any = json.loads('\n'.join(lines), parse_int=_skip, parse_float=_skip, parse_constant=_skip)
    except RecursionError:
        return 'unjudgeable', None
    except ValueError:
        return 'not_json', None

    # [3] Hidden only when it is an error whose code is about the app's own key, account or credit
    error: Any = cast('dict[str, Any]', value).get('error') if isinstance(value, dict) else None
    if not isinstance(error, dict):
        return 'pass', None
    fields = cast('dict[str, Any]', error)
    code, request_id = fields.get('code'), fields.get('request_id')
    if not (isinstance(code, str) and code in APP_CREDENTIAL_CODES):
        return 'pass', None
    return 'hide', request_id if isinstance(request_id, str) else None


class _Guard:
    """Judges the engine's events before any byte reaches the browser; the same logic for sync and async.

    Only whole events go out: as they came, or the generic frame for a hidden code. An event that cannot be judged
    (too large, too deep, or cut off) ends the stream closed instead: `closed` is set and nothing more may be read.
    """

    def __init__(self, request_id: str | None) -> None:
        self._request_id = request_id  # the response header's id, for a frame that carries none
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

            # [2] Each one judged whole: an unjudgeable one fails the stream closed
            verdict, request_id = _judge(event, self._first)
            self._first = False
            if verdict == 'unjudgeable':
                return bytes(out) + self.fail()
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
        if not rest:
            return b''
        verdict, request_id = _judge(rest, self._first)
        if verdict == 'pass':
            return rest

        # [2] A hidden one is replaced; one that cannot be read (cut off, or unjudgeable) fails closed
        return _generic_frame(request_id or self._request_id) if verdict == 'hide' else self.fail()

    def fail(self) -> bytes:
        """Drop the unjudged bytes and end the stream with the generic frame, after a blank line closing any event."""
        self._buffer.clear()
        self.closed = True
        return b'\n\n' + _generic_frame(self._request_id)


def _relay(stream: ChatStream) -> Iterator[bytes]:
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


async def _arelay(stream: AsyncChatStream) -> AsyncIterator[bytes]:
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
        # [3] Finished, failed, cancelled, or abandoned by the browser: upstream is closed
        await upstream.aclose()


def _streaming(request_id: str | None, body: Iterator[bytes] | AsyncIterator[bytes]) -> Reply:
    # Headers are the route's own: the engine's content-encoding and content-length never describe these bytes
    headers = dict(_SSE_HEADERS)
    if request_id:
        headers['x-request-id'] = request_id
    return Reply(200, headers, body)


def _buffered(result: Result[Any]) -> Reply:
    return _browser_error(result.error) if result.error is not None else _json_reply(200, result.data)


# --- The two helpers ------------------------------------------------------------------------------------------------


def _require(authorize: object) -> None:
    if not callable(authorize):
        raise TypeError(
            'A forwarding helper needs an authorize(request) function: it decides who may spend your credit.'
        )


class Forwarder(Generic[R]):
    """The forwarding route for a sync framework. `client` is a Meridian, or a function returning one (called per
    request, never at creation). `authorize(request)` returns True to proceed, False or None to refuse (403), or the
    app's own response, which is returned untouched as `Reply.passthrough`."""

    def __init__(
        self,
        *,
        client: Meridian | Callable[[], Meridian],
        authorize: Callable[[R], object],
        allow: AllowRule | None = None,
    ) -> None:
        _require(authorize)
        self._client, self._authorize, self._allowed = client, authorize, _Allowed.of(allow or AllowRule())

    def _decide(self, request: R) -> Reply | None:
        # A raising authorize is a refusal, never a crash and never a way through
        try:
            outcome = self._authorize(request)
        except Exception:
            return _denied()

        # An async authorize given to the sync helper is never awaited here, so it is never a yes
        if inspect.iscoroutine(outcome):
            outcome.close()
        return _denied() if inspect.isawaitable(outcome) else _verdict(outcome)

    def handle(
        self,
        request: R,
        *,
        method: str,
        content_type: str | None,
        idempotency_key: str | None,
        read_body: Callable[[], bytes],
    ) -> Reply:
        # [1] POST and JSON only, then the app's own decision
        refused = _precheck(method, content_type) or self._decide(request)
        if refused is not None:
            return refused

        # [2] A valid body, of which only the allowed fields go on
        try:
            raw = read_body()
        except Exception:
            return _unreadable()
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
            return _streaming(opened.data.request_id, _relay(opened.data))
        return _buffered(client.chat.create(params, idempotency_key=ask.idempotency_key))


class AsyncForwarder(Generic[R]):
    """Forwarder for an async framework: `client` is an AsyncMeridian or a function returning one, and `authorize`
    may be async."""

    def __init__(
        self,
        *,
        client: AsyncMeridian | Callable[[], AsyncMeridian],
        authorize: Callable[[R], object],
        allow: AllowRule | None = None,
    ) -> None:
        _require(authorize)
        self._client, self._authorize, self._allowed = client, authorize, _Allowed.of(allow or AllowRule())

    async def _decide(self, request: R) -> Reply | None:
        # A raising or rejecting authorize is a refusal, never a crash and never a way through
        try:
            outcome = self._authorize(request)
            if inspect.isawaitable(outcome):
                outcome = await outcome
        except Exception:
            return _denied()
        return _verdict(outcome)

    async def ahandle(
        self,
        request: R,
        *,
        method: str,
        content_type: str | None,
        idempotency_key: str | None,
        read_body: Callable[[], Awaitable[bytes]],
    ) -> Reply:
        # [1] POST and JSON only, then the app's own decision
        refused = _precheck(method, content_type) or await self._decide(request)
        if refused is not None:
            return refused

        # [2] A valid body, of which only the allowed fields go on
        try:
            raw = await read_body()
        except Exception:
            return _unreadable()
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
            return _streaming(opened.data.request_id, _arelay(opened.data))
        return _buffered(await client.chat.create(params, idempotency_key=ask.idempotency_key))
