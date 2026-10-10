import asyncio
import gzip
import inspect
import json
import numbers
import re
import threading
from collections.abc import AsyncIterator, Callable, Coroutine, Hashable, Iterator
from typing import Any, cast

import anyio
import httpx
import pytest

from meridian import AsyncMeridian, Meridian
from meridian._client import async_shared_client, shared_client
from meridian.forward import (
    APP_CREDENTIAL_CODES,
    MAX_BODY_BYTES,
    MAX_EVENT_BYTES,
    AllowRule,
    AsyncForwarder,
    AsyncReplyBody,
    Forwarder,
    Reply,
    ReplyBody,
)
from tests.support.fake_engine import Scripted, chunk, completion, engine_error, envelope, frame
from tests.support.modes import resolve
from tests.support.recorder import Recorder

pytestmark = [pytest.mark.anyio, pytest.mark.fake_only]
MSG: list[Any] = [{'role': 'user', 'content': 'hi'}]
Fwd = Forwarder[Any] | AsyncForwarder[Any]


def allow_all(_: object) -> bool:
    return True


def build(
    mode: str,
    engine: Any,
    *,
    authorize: Callable[[Any], object] = allow_all,
    allow: AllowRule | None = None,
    client: Any = None,
    response_type: type | tuple[type, ...] | None = None,
) -> Fwd:
    # One long-lived client per forwarder, built once as an app builds it at start-up (never one per request)
    cls = Meridian if mode == 'sync' else AsyncMeridian
    shared: Any = client or cls(engine.public_url, engine.project_key, transport=engine.transport, max_retries=0)
    options: dict[str, Any] = {} if response_type is None else {'response_type': response_type}
    if mode == 'sync':
        return Forwarder[Any](client=shared, authorize=authorize, allow=allow, **options)
    return AsyncForwarder[Any](client=shared, authorize=authorize, allow=allow, **options)


async def send(
    fwd: Fwd,
    body: Any,
    *,
    method: str = 'POST',
    content_type: str | None = 'application/json',
    key: str | None = None,
    request: Any = None,
    read: Callable[[], Any] | None = None,
) -> Reply[Any]:
    # `read_body` gives the body as bytes (both modes) or as a chunk source (Iterable / AsyncIterable of bytes)
    raw = body if isinstance(body, bytes) else json.dumps(body).encode()
    reader: Any = read or (lambda: raw)
    if isinstance(fwd, Forwarder):
        return fwd.handle(request, method=method, content_type=content_type, idempotency_key=key, read_body=reader)
    return await fwd.ahandle(request, method=method, content_type=content_type, idempotency_key=key, read_body=reader)


def payload(reply: Reply[Any]) -> dict[str, Any]:
    assert isinstance(reply.body, bytes)
    return json.loads(reply.body)


def generic(request_id: str | None) -> dict[str, Any]:
    """The browser's whole view of a hidden refusal (owner, 10 Oct), written out from the spec."""
    error: dict[str, Any] = {
        'message': 'The service is temporarily unavailable.',
        'type': 'api_error',
        'code': 'service_unavailable',
        'retryable': False,
    }
    if request_id:
        error['request_id'] = request_id
    return {'error': error}


async def drain(reply: Reply[Any]) -> bytes:
    body = reply.body
    if isinstance(body, bytes):
        return body
    if isinstance(body, AsyncReplyBody):
        return b''.join([piece async for piece in body])
    return b''.join(body)


def data_events(out: bytes) -> list[str]:
    """The data of every event the browser would read (a test-side reader, any line ending)."""
    text = out.decode().replace('\r\n', '\n').replace('\r', '\n')
    return [block[6:] for block in text.split('\n\n') if block.startswith('data: ')]


def last_event(out: bytes) -> Any:
    return json.loads(data_events(out)[-1])


# --- Request checks -------------------------------------------------------------------------------------------------


def test_a_missing_authorize_raises_at_construction() -> None:
    with pytest.raises(TypeError):
        Forwarder(client=lambda: None, authorize=None)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        AsyncForwarder(client=lambda: None, authorize=None)  # type: ignore[arg-type]


async def test_only_post(mode: str, engine: Any) -> None:
    reply = await send(build(mode, engine), {'messages': MSG}, method='GET')
    assert reply.status == 405 and reply.headers['allow'] == 'POST' and engine.requests == []
    assert payload(reply)['error']['code'] == 'method_not_allowed'


@pytest.mark.parametrize('ctype', ['text/plain', '', None, 'application/jsonx', 'application/x-www-form-urlencoded'])
async def test_only_json_and_checked_before_authorize(mode: str, engine: Any, ctype: str | None) -> None:
    seen: list[object] = []

    def watch(request: object) -> bool:
        seen.append(request)
        return True

    reply = await send(build(mode, engine, authorize=watch), {'messages': MSG}, content_type=ctype)
    assert reply.status == 415 and payload(reply)['error']['code'] == 'invalid_request' and seen == []


async def test_json_with_charset_is_accepted(mode: str, engine: Any) -> None:
    reply = await send(build(mode, engine), {'messages': MSG}, content_type='Application/JSON; charset=utf-8')
    assert reply.status == 200


def boom(_: object) -> bool:
    raise RuntimeError('auth service down')


async def aboom(_: object) -> bool:
    raise RuntimeError('auth service down')


async def async_yes(_: object) -> bool:
    return True


class AppResponse:
    """Stands in for a framework's response class (Starlette Response, Django HttpResponseBase, ...)."""


class MoreSpecificResponse(AppResponse):
    pass


@pytest.mark.parametrize(
    'verdict',
    [False, None, 1, 0, 'yes', '', {}, [], object()],
    ids=['false', 'none', 'one', 'zero', 'yes', 'empty-string', 'dict', 'list', 'object'],
)
async def test_only_the_literal_true_or_a_response_is_a_yes_anything_else_is_403(
    mode: str, engine: Any, verdict: object
) -> None:
    reply = await send(build(mode, engine, authorize=lambda r: verdict), {'messages': MSG})
    assert reply.status == 403 and reply.passthrough is None
    assert payload(reply)['error'] == {
        'message': 'Not allowed to use this route.',
        'type': 'permission_error',
        'code': 'permission_denied',
    }
    assert engine.requests == []


@pytest.mark.parametrize('kind', ['coroutine', 'future'])
async def test_an_awaitable_returned_from_an_async_authorize_is_403_and_a_coroutine_is_closed(
    engine: Any, kind: str
) -> None:
    made: list[Any] = []

    async def inner() -> bool:
        return True

    async def authorize(_: object) -> object:
        made.append(inner() if kind == 'coroutine' else asyncio.get_running_loop().create_future())
        return made[-1]

    reply = await send(build('async', engine, authorize=authorize), {'messages': MSG})
    assert reply.status == 403 and reply.passthrough is None and engine.requests == []
    if kind == 'coroutine':
        assert inspect.getcoroutinestate(cast('Coroutine[Any, Any, Any]', made[0])) == 'CORO_CLOSED'


async def test_a_raising_authorize_is_403(mode: str, engine: Any) -> None:
    reply = await send(build(mode, engine, authorize=boom), {'messages': MSG})
    assert reply.status == 403 and engine.requests == []


async def test_a_raising_async_authorize_is_403(engine: Any) -> None:
    reply = await send(build('async', engine, authorize=aboom), {'messages': MSG})
    assert reply.status == 403 and engine.requests == []


async def test_an_async_authorize_true_proceeds(engine: Any) -> None:
    reply = await send(build('async', engine, authorize=async_yes), {'messages': MSG})
    assert reply.status == 200


@pytest.mark.filterwarnings('error')
async def test_a_sync_forwarder_never_takes_an_async_authorize_as_a_yes(engine: Any) -> None:
    reply = await send(build('sync', engine, authorize=async_yes), {'messages': MSG})
    assert reply.status == 403 and reply.passthrough is None and engine.requests == []


async def test_the_apps_own_reply_is_passed_through_by_default(mode: str, engine: Any) -> None:
    mine = Reply[Any](401, {'location': '/login'}, b'sign in')
    reply = await send(build(mode, engine, authorize=lambda r: mine), {'messages': MSG})
    assert reply.passthrough is mine and engine.requests == []


@pytest.mark.parametrize('response_type', [AppResponse, (Reply, AppResponse)], ids=['type', 'tuple'])
async def test_the_apps_own_response_of_the_configured_type_is_passed_through(
    mode: str, engine: Any, response_type: type | tuple[type, ...]
) -> None:
    for mine in (AppResponse(), MoreSpecificResponse()):
        reply = await send(
            build(mode, engine, authorize=lambda r, m=mine: m, response_type=response_type), {'messages': MSG}
        )
        assert reply.passthrough is mine
    assert engine.requests == []


async def test_a_reply_is_not_a_response_once_another_response_type_is_configured(mode: str, engine: Any) -> None:
    reply = await send(
        build(mode, engine, authorize=lambda r: Reply[Any](200, {}, b'x'), response_type=AppResponse), {'messages': MSG}
    )
    assert reply.status == 403 and reply.passthrough is None and engine.requests == []


async def test_authorize_receives_the_request(mode: str, engine: Any) -> None:
    seen: list[object] = []
    marker = object()
    await send(build(mode, engine, authorize=lambda r: seen.append(r) is None), {'messages': MSG}, request=marker)
    assert seen == [marker]


VALID = json.dumps({'messages': MSG})


def nested(levels: int) -> bytes:
    """A chat body whose containers nest `levels` deep: the body, `messages`, the message, then lists in `content`
    (which the route forwards as it is, so the engine call has to encode all of it again)."""
    inner = levels - 3
    return ('{"messages": [{"role": "user", "content": ' + '[' * inner + ']' * inner + '}]}').encode()


@pytest.mark.parametrize(
    'body',
    [
        pytest.param(b'{not json', id='broken'),
        pytest.param(b'null', id='null'),
        pytest.param(b'[]', id='list'),
        pytest.param(b'"x"', id='string'),
        pytest.param(b'\xff\xfe', id='utf16-bom-only'),
        pytest.param(b'{"messages":"\xc3("}', id='invalid-utf8'),
        pytest.param(b'\xef\xbb\xbf' + VALID.encode(), id='utf8-bom'),
        pytest.param(VALID.encode('utf-16'), id='utf16-with-bom'),
        pytest.param(VALID.encode('utf-16-le'), id='utf16-le'),
        pytest.param(VALID.encode('utf-16-be'), id='utf16-be'),
        pytest.param(VALID.encode('utf-32-le'), id='utf32-le'),
        pytest.param(json.dumps({'messages': []}).encode(), id='no-messages'),
        pytest.param(json.dumps({'messages': 'hi'}).encode(), id='text-messages'),
        pytest.param(b'{"messages": [{"role": "user", "content": "hi"}], "temperature": NaN}', id='nan'),
        pytest.param(b'{"messages": [{"role": "user", "content": "hi"}], "x": 1e400}', id='inf'),
        pytest.param(nested(100_000), id='too-deep-for-the-parser'),
    ],
)
async def test_a_bad_body_is_400_and_never_sent(mode: str, engine: Any, body: bytes) -> None:
    reply = await send(build(mode, engine), body)
    assert reply.status == 400 and payload(reply)['error']['code'] == 'invalid_request' and engine.requests == []


@pytest.mark.parametrize('levels', [65, 66, 200, 900, 1500, 5000])
async def test_a_body_nested_deeper_than_64_levels_is_400_and_never_sent(mode: str, engine: Any, levels: int) -> None:
    reply = await send(build(mode, engine), nested(levels))
    assert reply.status == 400 and payload(reply)['error']['code'] == 'invalid_request' and engine.requests == []


async def test_no_nesting_depth_escapes_the_route_as_an_exception(mode: str, engine: Any) -> None:
    # Where the parser itself gives up depends on the interpreter and the stack under it; on 3.10 a body just inside
    # that limit used to parse and then raise RecursionError when it was encoded for the engine. Every depth is 400.
    fwd = build(mode, engine)
    refused = [levels for levels in range(65, 1100) if (await send(fwd, nested(levels))).status != 400]
    assert refused == [] and engine.requests == []


async def test_a_body_nested_exactly_64_levels_is_not_refused(mode: str, engine: Any) -> None:
    reply = await send(build(mode, engine), nested(64))
    assert reply.status == 200 and len(engine.requests) == 1


async def test_the_depth_is_counted_through_objects_and_lists_anywhere(mode: str, engine: Any) -> None:
    deep = '{"a": [' * 33 + '0' + ']}' * 33  # 66 containers on one path, inside the messages
    body = ('{"messages": [{"role": "user", "content": ' + deep + '}]}').encode()
    reply = await send(build(mode, engine), body)
    assert reply.status == 400 and engine.requests == []


async def test_a_body_that_cannot_be_read_is_400(mode: str, engine: Any) -> None:
    def broken() -> bytes:
        raise OSError('the browser went away')

    reply = await send(build(mode, engine), b'', read=broken)
    assert reply.status == 400 and engine.requests == []


# --- The body cap ---------------------------------------------------------------------------------------------------


def padded(size: int) -> bytes:
    """A valid chat body of exactly `size` bytes."""
    head = '{"messages": [{"role": "user", "content": "hi"}], "pad": "'
    return (head + 'x' * (size - len(head) - 2) + '"}').encode()


async def test_a_body_over_8_mib_is_413_and_never_sent(mode: str, engine: Any) -> None:
    assert MAX_BODY_BYTES == 8 << 20
    reply = await send(build(mode, engine), padded(MAX_BODY_BYTES + 1))
    assert reply.status == 413 and engine.requests == []
    assert payload(reply)['error'] == {
        'message': 'The request body is too large.',
        'type': 'invalid_request_error',
        'code': 'body_too_large',
    }


async def test_a_body_of_exactly_8_mib_is_not_refused_by_the_cap(mode: str, engine: Any) -> None:
    body = padded(MAX_BODY_BYTES)
    assert len(body) == MAX_BODY_BYTES
    reply = await send(build(mode, engine), body)
    assert reply.status == 200 and len(engine.requests) == 1


async def test_the_body_is_read_only_after_authorize_and_capped_before_it_is_parsed(mode: str, engine: Any) -> None:
    reads: list[int] = []

    def read() -> bytes:
        reads.append(1)
        return b'\xff' * (MAX_BODY_BYTES + 1)

    refused = await send(build(mode, engine, authorize=lambda r: False), b'', read=read)
    assert refused.status == 403 and reads == []
    too_big = await send(build(mode, engine), b'', read=read)
    assert too_big.status == 413 and reads == [1] and engine.requests == []


class Source:
    """A body as chunks, as a framework hands it over: a generator (sync) or an async generator (async)."""

    def __init__(self, pieces: Callable[[], Any]) -> None:
        self.pieces, self.pulled, self.closed = pieces, 0, False
        self.made: list[object] = []  # held, so only an explicit close (never garbage collection) closes a source

    def sync(self) -> Iterator[bytes]:
        try:
            for piece in self.pieces():
                self.pulled += 1
                yield piece
        finally:
            self.closed = True

    async def asynchronous(self) -> AsyncIterator[bytes]:
        try:
            for piece in self.pieces():
                self.pulled += 1
                yield piece
        finally:
            self.closed = True

    def reader(self, mode: str) -> Callable[[], Any]:
        def read() -> object:
            self.made.append(self.sync() if mode == 'sync' else self.asynchronous())
            return self.made[-1]

        return read


class UnlookableClose:
    """A chunk source (sync and async) whose `close` and `aclose` cannot even be looked up."""

    def __init__(self, raw: bytes) -> None:
        self.raw = raw

    def __iter__(self) -> Iterator[bytes]:
        return iter([self.raw])

    def __aiter__(self) -> AsyncIterator[bytes]:
        async def pieces() -> AsyncIterator[bytes]:
            yield self.raw

        return pieces()

    @property
    def close(self) -> object:
        raise RuntimeError('no close here')

    @property
    def aclose(self) -> object:
        raise RuntimeError('no aclose here')


async def test_a_close_that_cannot_be_looked_up_does_not_turn_a_good_body_into_a_400(mode: str, engine: Any) -> None:
    raw = json.dumps({'messages': MSG}).encode()
    reply = await send(build(mode, engine), raw, read=lambda: UnlookableClose(raw))
    assert reply.status == 200 and engine.bodies()[0]['messages'] == MSG


@pytest.mark.parametrize('usage', ['x', 1.5, [1], True], ids=repr)
async def test_a_reply_whose_usage_is_no_object_is_forwarded_without_counts(mode: str, engine: Any, usage: Any) -> None:
    engine.next(Scripted(json={**completion('hello'), 'usage': usage}))
    reply = await send(build(mode, engine), {'messages': MSG})
    assert reply.status == 200
    assert payload(reply)['usage'] == {} and payload(reply)['choices'][0]['message']['content'] == 'hello'


def endless() -> Iterator[bytes]:
    while True:
        yield b'x' * 65536


def breaking() -> Iterator[bytes]:
    yield b'{"messages": '
    raise OSError('the browser went away')


async def test_the_body_may_come_as_chunks(mode: str, engine: Any) -> None:
    body = json.dumps({'messages': MSG, 'temperature': 0.5}).encode()
    source = Source(lambda: (body[i : i + 3] for i in range(0, len(body), 3)))
    reply = await send(build(mode, engine), b'', read=source.reader(mode))
    assert reply.status == 200 and engine.bodies()[0]['temperature'] == 0.5 and source.closed


async def test_an_endless_chunk_source_is_413_and_read_no_further_than_the_cap(mode: str, engine: Any) -> None:
    source = Source(endless)
    reply = await send(build(mode, engine), b'', read=source.reader(mode))
    assert reply.status == 413 and payload(reply)['error']['code'] == 'body_too_large' and engine.requests == []
    assert source.pulled == MAX_BODY_BYTES // 65536 + 1 and source.closed


async def test_a_chunk_source_is_read_only_after_authorize(mode: str, engine: Any) -> None:
    source = Source(endless)
    reply = await send(build(mode, engine, authorize=lambda r: False), b'', read=source.reader(mode))
    assert reply.status == 403 and source.pulled == 0


@pytest.mark.parametrize(
    'pieces',
    [
        pytest.param(lambda: iter([b'{"messages": ', '[]}']), id='a-str-chunk'),
        pytest.param(lambda: iter([b'{', 5]), id='an-int-chunk'),
        pytest.param(lambda: breaking(), id='a-raising-source'),
    ],
)
async def test_a_chunk_source_that_breaks_or_gives_no_bytes_is_400(
    mode: str, engine: Any, pieces: Callable[[], Any]
) -> None:
    reply = await send(build(mode, engine), b'', read=Source(pieces).reader(mode))
    assert reply.status == 400 and payload(reply)['error']['code'] == 'invalid_request' and engine.requests == []


async def test_an_async_forwarder_does_not_take_a_sync_iterable_or_a_coroutine(engine: Any) -> None:
    body = json.dumps({'messages': MSG}).encode()

    async def coroutine() -> bytes:
        return body

    made: list[Any] = []
    for read in (lambda: iter([body]), lambda: made.append(coroutine()) or made[-1]):
        reply = await send(build('async', engine), b'', read=read)
        assert reply.status == 400
    assert inspect.getcoroutinestate(made[0]) == 'CORO_CLOSED' and engine.requests == []


# --- What reaches the engine ----------------------------------------------------------------------------------------


async def test_only_allowed_fields_reach_the_engine(mode: str, engine: Any) -> None:
    body: dict[str, Any] = {
        'messages': MSG,
        'model': 'gpt-x',
        'stream_options': {'include_usage': False},
        'tools': [{}],
        'max_tokens': 1_000_000,
        'temperature': 0.3,
        'constructor': 1,
        'x': 2,
    }
    await send(build(mode, engine, allow=AllowRule(options=('max_tokens', 'temperature', 'constructor'))), body)
    sent = engine.bodies()[0]
    assert sent['max_tokens'] == 4096 and sent['temperature'] == 0.3 and sent['constructor'] == 1
    assert 'model' not in sent and 'tools' not in sent and 'x' not in sent and sent['stream'] is False
    assert 'stream_options' not in sent and sent['messages'] == MSG


async def test_options_can_never_name_the_fields_the_route_owns(mode: str, engine: Any) -> None:
    owned = ('model', 'stream', 'stream_options', 'messages', 'max_tokens', 'temperature')
    body = {
        'messages': MSG,
        'model': 'x',
        'stream_options': {'include_usage': False},
        'max_tokens': 10**6,
        'temperature': 'hot',
    }
    await send(build(mode, engine, allow=AllowRule(options=owned)), body)
    assert engine.bodies()[0] == {'messages': MSG, 'max_tokens': 4096, 'stream': False}


async def test_options_can_never_lift_the_cap_through_max_completion_tokens(mode: str, engine: Any) -> None:
    allow = AllowRule(options=['max_completion_tokens'])
    await send(build(mode, engine, allow=allow), {'messages': MSG, 'max_completion_tokens': 1_000_000})
    sent = engine.bodies()[0]
    assert 'max_completion_tokens' not in sent and sent['max_tokens'] == 4096


async def test_options_can_never_copy_n(mode: str, engine: Any) -> None:
    # Many choices per request would multiply what one browser request spends
    await send(build(mode, engine, allow=AllowRule(options=('n', 'seed'))), {'messages': MSG, 'n': 50, 'seed': 1})
    sent = engine.bodies()[0]
    assert 'n' not in sent and sent['seed'] == 1


async def test_options_are_copied_only_when_present(mode: str, engine: Any) -> None:
    allow = AllowRule(options=('top_p', 'seed'))
    await send(build(mode, engine, allow=allow), {'messages': MSG, 'top_p': None})
    sent = engine.bodies()[0]
    assert 'top_p' in sent and sent['top_p'] is None and 'seed' not in sent


async def test_a_bare_string_names_one_option_or_model_never_its_letters(mode: str, engine: Any) -> None:
    allow = AllowRule(model='gpt-4', options='seed')
    await send(build(mode, engine, allow=allow), {'messages': MSG, 'model': 'gpt', 's': 1, 'seed': 7})
    await send(build(mode, engine, allow=allow), {'messages': MSG, 'model': 'gpt-4'})
    first, second = engine.bodies()
    assert 'model' not in first and 's' not in first and first['seed'] == 7
    assert second['model'] == 'gpt-4'


@pytest.mark.parametrize(
    ('given', 'expected'),
    [(-7, 1), (0, 1), (2.9, 2), ('50', 4096), (100, 100), (True, 4096), (None, 4096), (10**30, 4096)],
)
async def test_max_tokens_is_clamped(mode: str, engine: Any, given: object, expected: int) -> None:
    await send(build(mode, engine), {'messages': MSG, 'max_tokens': given})
    assert engine.bodies()[0]['max_tokens'] == expected


async def test_an_absent_max_tokens_sends_the_cap(mode: str, engine: Any) -> None:
    await send(build(mode, engine), {'messages': MSG})
    await send(build(mode, engine, allow=AllowRule(max_tokens_cap=100)), {'messages': MSG})
    assert [body['max_tokens'] for body in engine.bodies()] == [4096, 100]


@pytest.mark.parametrize('cap', [0, -5, 1.5, float('nan'), '10', True])
async def test_an_invalid_cap_falls_back_to_4096(mode: str, engine: Any, cap: Any) -> None:
    await send(build(mode, engine, allow=AllowRule(max_tokens_cap=cap)), {'messages': MSG, 'max_tokens': 10**6})
    assert engine.bodies()[0]['max_tokens'] == 4096


@pytest.mark.parametrize(('given', 'kept'), [(0.3, True), (0, True), (True, False), ('0.3', False), (None, False)])
async def test_temperature_only_as_a_number(mode: str, engine: Any, given: object, kept: bool) -> None:
    await send(build(mode, engine), {'messages': MSG, 'temperature': given})
    assert ('temperature' in engine.bodies()[0]) is kept


async def test_a_model_only_when_allowed(mode: str, engine: Any) -> None:
    await send(build(mode, engine, allow=AllowRule(model=['m1'])), {'messages': MSG, 'model': 'm1'})
    await send(build(mode, engine, allow=AllowRule(model=['m1'])), {'messages': MSG, 'model': 'm2'})
    await send(build(mode, engine, allow=AllowRule(model=True)), {'messages': MSG, 'model': 'any'})
    await send(build(mode, engine, allow=AllowRule(model=True)), {'messages': MSG, 'model': 5})
    first, second, third, fourth = engine.bodies()
    assert first['model'] == 'm1' and 'model' not in second and third['model'] == 'any' and 'model' not in fourth


@pytest.mark.parametrize(
    ('key', 'forwarded'),
    [
        ('k-123', True),
        ('a-b_c.1', True),
        ('x' * 255, True),
        ('!~', True),
        ('', False),
        ('x' * 256, False),
        ('ü', False),
        ('   ', False),
        ('k\n', False),
        ('k\r\nx', False),
        ('k\tx', False),
        ('k\x7f', False),
        ('order 42', False),
        (' k-1', False),
        ('k-1 ', False),
    ],
)
async def test_the_browser_idempotency_key(mode: str, engine: Any, key: str, forwarded: bool) -> None:
    # The engine's own rule (and the client's): 1-255 characters, each visible ASCII (0x21-0x7E). Anything else is
    # replaced by a key the client makes, never trimmed or refused
    await send(build(mode, engine), {'messages': MSG}, key=key)
    sent = engine.requests[0].headers['idempotency-key']
    assert (sent == key) is forwarded and 1 <= len(sent) <= 255
    assert all(0x21 <= ord(c) <= 0x7E for c in sent)


# --- The client -----------------------------------------------------------------------------------------------------


async def test_a_server_misconfiguration_is_a_generic_503(mode: str, engine: Any) -> None:
    cls = Meridian if mode == 'sync' else AsyncMeridian
    for stream in (False, True):
        reply = await send(
            build(mode, engine, client=lambda: cls(engine.public_url, '')), {'messages': MSG, 'stream': stream}
        )
        assert reply.status == 503 and payload(reply) == generic(None)
        assert 'key' not in json.dumps(payload(reply)).lower()


async def test_a_raising_client_factory_is_a_generic_503(mode: str, engine: Any) -> None:
    def broken() -> Meridian:
        raise RuntimeError('MERIDIAN_KEY is not set')

    reply = await send(build(mode, engine, client=broken), {'messages': MSG})
    assert reply.status == 503 and payload(reply) == generic(None) and engine.requests == []


@pytest.mark.parametrize('wrong', ['none', 'factory-of-the-other-kind', 'instance-of-the-other-kind'])
async def test_a_client_of_the_wrong_kind_is_a_generic_503(mode: str, engine: Any, wrong: str) -> None:
    other = AsyncMeridian if mode == 'sync' else Meridian
    made = other(engine.public_url, engine.project_key, transport=engine.transport)
    client = {'none': lambda: None, 'factory-of-the-other-kind': lambda: made}.get(wrong, made)
    reply = await send(build(mode, engine, client=client), {'messages': MSG})
    assert reply.status == 503 and payload(reply) == generic(None) and engine.requests == []


async def test_the_client_factory_is_called_per_request_not_at_creation(mode: str, engine: Any) -> None:
    calls: list[int] = []
    cls = Meridian if mode == 'sync' else AsyncMeridian
    shared = cls(engine.public_url, engine.project_key, transport=engine.transport)

    def factory() -> Any:
        calls.append(1)
        return shared

    fwd = build(mode, engine, client=factory)
    assert calls == []
    await send(fwd, {'messages': MSG})
    await send(fwd, {'messages': MSG})
    assert len(calls) == 2


async def test_the_forwarder_never_closes_the_client_it_is_given(mode: str, engine: Any) -> None:
    # The app owns the long-lived client and closes it at shutdown: every kind of request must leave it usable
    cls = Meridian if mode == 'sync' else AsyncMeridian
    shared = cls(engine.public_url, engine.project_key, transport=engine.transport, max_retries=0)
    fwd = build(mode, engine, client=lambda: shared)
    engine.next(Scripted(chunks=[frame(chunk('a')).encode(), frame('[DONE]').encode()]), engine_error('rate_limited'))
    streamed = await send(fwd, {'messages': MSG, 'stream': True})
    await drain(streamed)
    refused = await send(fwd, {'messages': MSG})
    after = await send(fwd, {'messages': MSG})
    assert streamed.status == 200 and refused.status == 429 and after.status == 200


async def test_a_client_instance_is_used_as_is(mode: str, engine: Any) -> None:
    cls = Meridian if mode == 'sync' else AsyncMeridian
    reply = await send(
        build(mode, engine, client=cls(engine.public_url, engine.project_key, transport=engine.transport)),
        {'messages': MSG},
    )
    assert reply.status == 200


# --- Buffered replies and refusals ----------------------------------------------------------------------------------


async def test_a_buffered_reply_is_json(mode: str, engine: Any) -> None:
    engine.next(Scripted(json=completion('hey')))
    reply = await send(build(mode, engine), {'messages': MSG})
    assert reply.status == 200 and reply.headers['content-type'] == 'application/json'
    assert payload(reply)['choices'][0]['message']['content'] == 'hey'


async def test_a_buffered_reply_relays_a_valid_request_id_only(mode: str, engine: Any) -> None:
    engine.next(
        Scripted(headers={'x-request-id': 'req_ok'}, json=completion('hey')),
        Scripted(headers={'x-request-id': wire_id('a\r\nset-cookie: x=1')}, json=completion('hey')),
    )
    fwd = build(mode, engine)
    good, bad = await send(fwd, {'messages': MSG}), await send(fwd, {'messages': MSG})
    assert good.headers == {'content-type': 'application/json', 'x-request-id': 'req_ok'}
    assert bad.headers == {'content-type': 'application/json'} and bad.status == 200


@pytest.mark.parametrize(
    ('value', 'relayed'),
    [
        ('7', True),
        ('0', True),
        ('Wed, 21 Oct 2037 07:28:00 GMT', True),
        ('  7  ', True),
        ('1e3', False),
        ('7.5', False),
        ('-1', False),
        ('soon', False),
        ('7\r\nset-cookie: x=1', False),
    ],
)
async def test_a_passed_through_rate_limit_relays_a_valid_retry_after(
    mode: str, engine: Any, value: str, relayed: bool
) -> None:
    engine.next(Scripted(status=429, headers={'retry-after': wire_id(value)}, json=envelope('rate_limited', 'req_rl')))
    reply = await send(build(mode, engine), {'messages': MSG})
    assert reply.status == 429 and payload(reply)['error']['code'] == 'rate_limited'
    assert reply.headers.get('retry-after') == (value.strip() if relayed else None)


@pytest.mark.parametrize('status', [400, 429])
@pytest.mark.parametrize(
    'value',
    [
        'Mon, 01 Jan 99999999999999999999 00:00:00 GMT',  # a year no datetime holds
        'Thu, 01 Jan 0100 00:00:00 +1400',  # year 100 as written: far before the epoch
        'Mon, 01 Jan 0000 00:00:00 GMT',  # a year below 100 is read as 19xx or 20xx: here 2000, an ordinary past date
        'Mon, 01 Jan 0001 00:00:00 +1400',  # likewise 2001
    ],
)
async def test_an_odd_retry_after_date_never_makes_the_forwarder_raise(
    mode: str, engine: Any, value: str, status: int
) -> None:
    code = 'invalid_request' if status == 400 else 'rate_limited'
    engine.next(Scripted(status=status, headers={'retry-after': value}, json=envelope(code, 'req_ra')))
    reply = await send(build(mode, engine), {'messages': MSG})
    assert reply.status == status and payload(reply)['error']['code'] == code


@pytest.mark.parametrize(
    ('code', 'status'),
    [('insufficient_credit', 402), ('hard_limit_reached', 429), ('service_unavailable', 503), ('internal', 500)],
)
async def test_retry_after_is_relayed_for_no_other_code(mode: str, engine: Any, code: str, status: int) -> None:
    engine.next(Scripted(status=status, headers={'retry-after': '7'}, json=envelope(code, 'req_x')))
    reply = await send(build(mode, engine), {'messages': MSG})
    assert 'retry-after' not in reply.headers


async def test_a_rate_limit_before_a_stream_starts_relays_retry_after_too(mode: str, engine: Any) -> None:
    engine.next(Scripted(status=429, headers={'retry-after': '9'}, json=envelope('rate_limited', 'req_rl')))
    reply = await send(build(mode, engine), {'messages': MSG, 'stream': True})
    assert reply.status == 429 and reply.headers['retry-after'] == '9'


def test_the_ten_codes_of_the_spec() -> None:
    assert sorted(APP_CREDENTIAL_CODES) == [
        'byok_required',
        'entitlement_missing',
        'hard_limit_reached',
        'insufficient_credit',
        'invalid_api_key',
        'key_expired',
        'key_revoked',
        'organisation_unavailable',
        'project_host_mismatch',
        'project_unavailable',
    ]


@pytest.mark.parametrize('code', sorted(APP_CREDENTIAL_CODES))
async def test_app_credential_refusals_become_a_generic_503(mode: str, engine: Any, code: str) -> None:
    engine.next(engine_error(code))
    reply = await send(build(mode, engine), {'messages': MSG})
    error = payload(reply)['error']
    assert reply.status == 503 and error['code'] == 'service_unavailable' and error['type'] == 'api_error'
    assert error['message'] == 'The service is temporarily unavailable.' and error['request_id'].startswith('req_fake_')
    assert isinstance(reply.body, bytes) and code.encode() not in reply.body


async def test_the_generic_503_says_nothing_else_and_is_not_retryable(mode: str, engine: Any) -> None:
    headers = {'x-request-id': 'req_hidden', 'retry-after': '9'}
    engine.next(Scripted(status=402, headers=headers, json=envelope('insufficient_credit', 'req_hidden')))
    reply = await send(build(mode, engine), {'messages': MSG})
    assert reply.status == 503 and payload(reply) == generic('req_hidden')
    assert reply.headers == {'content-type': 'application/json', 'x-request-id': 'req_hidden'}


async def test_a_refusal_before_a_stream_starts_is_hidden_too(mode: str, engine: Any) -> None:
    engine.next(Scripted(status=401, headers={'x-request-id': 'req_pre'}, json=envelope('invalid_api_key', 'req_pre')))
    reply = await send(build(mode, engine), {'messages': MSG, 'stream': True})
    assert reply.status == 503 and payload(reply) == generic('req_pre')


async def test_a_network_failure_is_a_generic_503(mode: str, engine: Any) -> None:
    engine.next(Scripted(network_error=True))
    reply = await send(build(mode, engine), {'messages': MSG})
    assert reply.status == 503 and payload(reply) == generic(None)


async def test_the_same_refusal_reaches_server_code_unchanged(mode: str, engine: Any) -> None:
    engine.next(engine_error('insufficient_credit'))
    client = (Meridian if mode == 'sync' else AsyncMeridian)(
        engine.public_url, engine.project_key, transport=engine.transport
    )
    result = await resolve(client.chat.create({'messages': MSG}))
    assert result.error.code == 'insufficient_credit'


@pytest.mark.parametrize(
    ('code', 'status', 'kind'),
    [
        ('invalid_request', 400, 'invalid_request_error'),
        ('model_not_allowed', 403, 'permission_error'),
        ('rate_limited', 429, 'rate_limit_error'),
        ('upstream_error', 502, 'api_error'),
    ],
)
async def test_end_user_refusals_keep_their_real_code(
    mode: str, engine: Any, code: str, status: int, kind: str
) -> None:
    engine.next(Scripted(status=status, headers={'x-request-id': 'req_u'}, json=envelope(code, 'req_u')))
    reply = await send(build(mode, engine), {'messages': MSG})
    assert reply.status == status and reply.headers['x-request-id'] == 'req_u'
    assert payload(reply)['error'] == {'message': f'fake {code}', 'type': kind, 'code': code, 'request_id': 'req_u'}


@pytest.mark.parametrize('stream', [False, True], ids=['buffered', 'before-the-stream'])
async def test_an_unrecognised_engine_code_is_a_generic_503(mode: str, engine: Any, stream: bool) -> None:
    # A newer engine's code might be about the app's own account: it fails closed, like the ten known ones
    headers = {'x-request-id': 'req_new', 'retry-after': '9'}
    engine.next(Scripted(status=402, headers=headers, json=envelope('subscription_expired', 'req_new')))
    reply = await send(build(mode, engine), {'messages': MSG, 'stream': stream})
    assert reply.status == 503 and payload(reply) == generic('req_new')
    assert reply.headers == {'content-type': 'application/json', 'x-request-id': 'req_new'}
    assert isinstance(reply.body, bytes) and b'subscription_expired' not in reply.body


async def test_an_engine_internal_error_is_a_known_code_and_passes_through_unchanged(mode: str, engine: Any) -> None:
    engine.next(Scripted(status=500, headers={'x-request-id': 'req_i'}, json=envelope('internal', 'req_i')))
    reply = await send(build(mode, engine), {'messages': MSG})
    assert reply.status == 500 and reply.headers['x-request-id'] == 'req_i'
    assert payload(reply)['error'] == {
        'message': 'fake internal',
        'type': 'api_error',
        'code': 'internal',
        'request_id': 'req_i',
    }


MISSING = object()
BAD_CODES = [
    pytest.param(MISSING, id='missing'),
    pytest.param(None, id='null'),
    pytest.param(402, id='number'),
    pytest.param(True, id='bool'),
    pytest.param(['insufficient_credit'], id='list'),
    pytest.param({'code': 'insufficient_credit'}, id='object'),
]


def bad_error(code: object) -> dict[str, Any]:
    """An error object the engine never sends: its `code` is missing, empty or not a string."""
    error: dict[str, Any] = {'message': 'the app account 42 is out', 'type': 'api_error', 'request_id': 'req_bad'}
    if code is not MISSING:
        error['code'] = code
    return error


@pytest.mark.parametrize('stream', [False, True], ids=['buffered', 'before-the-stream'])
@pytest.mark.parametrize('code', BAD_CODES)
async def test_an_error_whose_code_is_missing_or_not_a_string_is_a_generic_503(
    mode: str, engine: Any, code: object, stream: bool
) -> None:
    engine.next(Scripted(status=402, headers={'x-request-id': 'req_bad'}, json={'error': bad_error(code)}))
    reply = await send(build(mode, engine), {'messages': MSG, 'stream': stream})
    assert reply.status == 503 and payload(reply) == generic('req_bad')
    assert isinstance(reply.body, bytes) and b'account 42' not in reply.body


def carried_id(error: Any) -> str | None:
    """The request id an error value carries: only an object can carry one."""
    return cast('dict[str, str]', error).get('request_id') if isinstance(error, dict) else None


# What an `error` key may hold without being the engine's envelope (an object with a code)
NOT_AN_ENVELOPE = [
    pytest.param('boom', id='string'),
    pytest.param('', id='empty-string'),
    pytest.param(123, id='number'),
    pytest.param(0, id='zero'),
    pytest.param(True, id='true'),
    pytest.param(False, id='false'),
    pytest.param(['boom'], id='list'),
    pytest.param([], id='empty-list'),
    pytest.param({}, id='empty-object'),
]


@pytest.mark.parametrize('stream', [False, True], ids=['buffered', 'before-the-stream'])
@pytest.mark.parametrize('error', [*NOT_AN_ENVELOPE, pytest.param(None, id='null')])
async def test_an_error_status_whose_error_is_not_an_envelope_is_a_generic_503(
    mode: str, engine: Any, error: object, stream: bool
) -> None:
    engine.next(Scripted(status=402, headers={'x-request-id': 'req_h'}, json={'error': error}))
    reply = await send(build(mode, engine), {'messages': MSG, 'stream': stream})
    assert reply.status == 503 and payload(reply) == generic('req_h')
    assert isinstance(reply.body, bytes) and b'boom' not in reply.body


@pytest.mark.parametrize(
    'error',
    [
        *NOT_AN_ENVELOPE,
        pytest.param(envelope('insufficient_credit', 'req_200')['error'], id='hidden-code'),
        pytest.param(envelope('subscription_expired', 'req_200')['error'], id='unrecognised-code'),
        # A known code that is not about the app's account is still no business of a 200: every non-null error is out
        pytest.param(envelope('internal', 'req_200')['error'], id='internal'),
        pytest.param(envelope('rate_limited', 'req_200')['error'], id='rate_limited'),
        pytest.param(envelope('invalid_request', 'req_200')['error'], id='invalid_request'),
        pytest.param(envelope('service_unavailable', 'req_200')['error'], id='service_unavailable'),
        pytest.param(envelope('upstream_error')['error'], id='known-code-fresh-id'),
    ],
)
async def test_a_200_reply_with_an_error_key_is_a_generic_503(mode: str, engine: Any, error: Any) -> None:
    # The engine answers a failure with its error status, never a 200: a 200 that carries an error is not served on
    engine.next(Scripted(headers={'x-request-id': 'req_hdr'}, json={**completion('hey'), 'error': error}))
    reply = await send(build(mode, engine), {'messages': MSG})
    expected = carried_id(error) or 'req_hdr'  # the error object's own id first, else the response's
    assert reply.status == 503 and payload(reply) == generic(expected)
    assert reply.headers == {'content-type': 'application/json', 'x-request-id': expected}
    assert isinstance(reply.body, bytes) and b'hey' not in reply.body and b'insufficient_credit' not in reply.body


@pytest.mark.parametrize('error', ['boom', {'message': 'm'}])
async def test_a_200_reply_with_an_error_and_no_id_of_its_own_keeps_the_responses_id(
    mode: str, engine: Any, error: Any
) -> None:
    engine.next(Scripted(headers={'x-request-id': 'req_hdr'}, json={**completion('hey'), 'error': error}))
    reply = await send(build(mode, engine), {'messages': MSG})
    assert payload(reply) == generic('req_hdr') and reply.headers['x-request-id'] == 'req_hdr'


async def test_a_200_reply_with_a_null_error_key_is_an_ordinary_reply(mode: str, engine: Any) -> None:
    engine.next(Scripted(json={**completion('hey'), 'error': None}))
    reply = await send(build(mode, engine), {'messages': MSG})
    assert reply.status == 200 and payload(reply)['choices'][0]['message']['content'] == 'hey'


# A reply holding a NaN or Infinity is not JSON (the client reads it strictly, as Go and JavaScript do) and is a
# network error with the response's request id: the generic 503, never a raise. A reply the client decoded may also
# not encode for the browser: nested past the encoder's depth on a Python whose decoder allows more (next test)
@pytest.mark.parametrize('extra', [pytest.param('NaN', id='nan'), pytest.param('[Infinity]', id='infinity')])
async def test_a_buffered_reply_holding_a_nan_is_a_generic_503_with_the_replys_id(
    mode: str, engine: Any, extra: str
) -> None:
    text = '{"choices": [], "extra": ' + extra + '}'
    engine.next(Scripted(text=text, headers={'content-type': 'application/json', 'x-request-id': 'req_enc'}))
    reply = await send(build(mode, engine), {'messages': MSG})
    assert reply.status == 503 and payload(reply) == generic('req_enc')
    assert reply.headers == {'content-type': 'application/json', 'x-request-id': 'req_enc'}


@pytest.mark.parametrize(
    'extra',
    [
        pytest.param('{"a": ' * 10_000 + '1' + '}' * 10_000, id='10000-dicts'),
        pytest.param('[' * 16_000 + ']' * 16_000, id='16000-lists'),
    ],
)
async def test_a_buffered_reply_nested_too_deep_to_encode_is_a_generic_503(mode: str, engine: Any, extra: str) -> None:
    # Where the client's decoder refuses it (older Pythons) it is a network error; where it decodes it, the encoder
    # for the browser refuses it. Either way the browser sees the generic 503 and the call never raises
    text = '{"choices": [], "extra": ' + extra + '}'
    engine.next(Scripted(text=text, headers={'content-type': 'application/json', 'x-request-id': 'req_enc'}))
    reply = await send(build(mode, engine), {'messages': MSG})
    error = payload(reply)['error']
    assert (
        reply.status == 503 and error['code'] == 'service_unavailable' and error.get('request_id') in (None, 'req_enc')
    )
    assert isinstance(reply.body, bytes) and b'choices' not in reply.body


# --- Streaming ------------------------------------------------------------------------------------------------------


async def stream_reply(mode: str, engine: Any, parts: list[bytes], **scripted: Any) -> Reply[Any]:
    engine.next(Scripted(chunks=list(parts), **scripted))
    return await send(build(mode, engine), {'messages': MSG, 'stream': True})


def hidden_frame(code: str = 'insufficient_credit', pad: int = 0) -> bytes:
    body = {
        'error': {'message': 'x' * pad or 'fake', 'type': 'rate_limit_error', 'code': code, 'request_id': 'req_mid'}
    }
    return f'data: {json.dumps(body)}\n\n'.encode()


def pieces(wire: bytes, size: int) -> list[bytes]:
    return [wire[i : i + size] for i in range(0, len(wire), size)]


async def test_streaming_passes_engine_bytes_through(mode: str, engine: Any) -> None:
    wire = frame(chunk('a')) + frame('[DONE]')
    engine.next(Scripted(chunks=[wire], headers={'x-request-id': 'req_s'}))
    body = {'messages': MSG, 'stream': True, 'stream_options': {'include_usage': False}}
    reply = await send(build(mode, engine, allow=AllowRule(options=('stream_options',))), body)
    assert reply.status == 200
    assert reply.headers == {
        'content-type': 'text/event-stream; charset=utf-8',
        'cache-control': 'no-cache',
        'x-request-id': 'req_s',
    }
    assert await drain(reply) == wire.encode()
    sent = engine.bodies()[0]
    assert sent['stream'] is True and sent['stream_options'] == {'include_usage': True}
    assert engine.upstream_closed()


async def test_compression_headers_are_never_forwarded(mode: str, engine: Any) -> None:
    wire = (frame(chunk('a')) + frame('[DONE]')).encode()
    packed = gzip.compress(wire)
    headers = {'content-encoding': 'gzip', 'content-length': str(len(packed))}
    reply = await stream_reply(mode, engine, [packed[:10], packed[10:]], headers=headers)
    assert 'content-encoding' not in reply.headers and 'content-length' not in reply.headers
    assert await drain(reply) == wire


async def test_a_hidden_mid_stream_frame_split_byte_by_byte_is_rewritten(mode: str, engine: Any) -> None:
    wire = frame(chunk('a')).encode() + hidden_frame()
    out = await drain(await stream_reply(mode, engine, pieces(wire, 1)))
    assert out.startswith(frame(chunk('a')).encode())
    assert b'insufficient_credit' not in out and b'service_unavailable' in out and b'req_mid' in out
    assert b'The service is temporarily unavailable.' in out
    assert last_event(out) == generic('req_mid') and out.endswith(b'\n\n')


async def test_a_200kb_hidden_frame_in_512_byte_pieces_never_fails_open(mode: str, engine: Any) -> None:
    wire = hidden_frame('key_revoked', pad=200_000)
    out = await drain(await stream_reply(mode, engine, pieces(wire, 512)))
    assert b'key_revoked' not in out and b'x' * 1000 not in out and b'service_unavailable' in out


async def test_a_200kb_ordinary_frame_passes_through_exactly(mode: str, engine: Any) -> None:
    wire = frame(chunk('y' * 200_000)).encode() + frame('[DONE]').encode()
    out = await drain(await stream_reply(mode, engine, pieces(wire, 512)))
    assert out == wire


async def test_a_non_hidden_mid_stream_error_passes_through_unchanged(mode: str, engine: Any) -> None:
    upstream = f'data: {json.dumps(envelope("upstream_error"))}\r\n\r\n'.encode()
    wire = frame(chunk('a')).encode() + upstream
    out = await drain(await stream_reply(mode, engine, [wire[:7], wire[7:]]))
    assert out == wire


@pytest.mark.parametrize('size', [1, 7, 100_000])
async def test_an_unrecognised_engine_code_in_a_mid_stream_frame_is_rewritten(
    mode: str, engine: Any, size: int
) -> None:
    wire = frame(chunk('a')).encode() + hidden_frame('subscription_expired')
    out = await drain(await stream_reply(mode, engine, pieces(wire, size)))
    assert out.startswith(frame(chunk('a')).encode()) and b'subscription_expired' not in out
    assert last_event(out) == generic('req_mid')


async def test_an_engine_internal_error_frame_passes_through_byte_for_byte(mode: str, engine: Any) -> None:
    wire = frame(chunk('a')).encode() + hidden_frame('internal')
    assert await drain(await stream_reply(mode, engine, pieces(wire, 5))) == wire


@pytest.mark.parametrize('size', [1, 100_000])
@pytest.mark.parametrize('code', BAD_CODES)
async def test_a_mid_stream_error_object_with_a_bad_code_is_replaced_by_the_generic_frame_with_its_own_id(
    mode: str, engine: Any, code: object, size: int
) -> None:
    wire = frame(chunk('a')).encode() + f'data: {json.dumps({"error": bad_error(code)})}\n\n'.encode()
    out = await drain(await stream_reply(mode, engine, pieces(wire, size)))
    assert out.startswith(frame(chunk('a')).encode()) and b'account 42' not in out
    assert last_event(out) == generic('req_bad')


async def test_a_mid_stream_error_frame_without_any_code_or_id_takes_the_response_id(mode: str, engine: Any) -> None:
    wire = frame(chunk('a')).encode() + b'data: {"error": {}}\n\n'
    out = await drain(await stream_reply(mode, engine, [wire], headers={'x-request-id': 'req_h'}))
    assert last_event(out) == generic('req_h')


# An error key that is set (not null) but is no object with a non-empty string `code` ends the stream closed
TERMINAL_ERRORS = [
    *NOT_AN_ENVELOPE,
    *[pytest.param(bad_error(code.values[0]), id=f'code-{code.id}') for code in BAD_CODES],
]


def terminal_wire(error: Any) -> tuple[bytes, bytes, bytes, str]:
    """The first chunk, the frame with the bad error key, the rest the engine would have sent, the id to expect."""
    first, rest = frame(chunk('a')).encode(), frame(chunk('after')).encode() + frame('[DONE]').encode()
    bad = f'data: {json.dumps({"error": error})}\n\n'.encode()
    return first, bad, rest, carried_id(error) or 'req_h'


@pytest.mark.parametrize('error', TERMINAL_ERRORS)
async def test_an_error_key_that_is_no_envelope_with_a_code_ends_the_stream_closed(
    mode: str, engine: Any, error: Any
) -> None:
    first, bad, rest, request_id = terminal_wire(error)
    reply = await stream_reply(mode, engine, [first, bad, *pieces(rest, 20)], headers={'x-request-id': 'req_h'})
    out = await drain(reply)
    assert b'after' not in out and b'[DONE]' not in out and b'boom' not in out and b'account 42' not in out
    assert out.startswith(first) and last_event(out) == generic(request_id) and out.endswith(b'}}\n\n')
    assert engine.upstream_closed() and engine.parts_read() == 2  # nothing after the frame is even read


@pytest.mark.parametrize('error', TERMINAL_ERRORS)
async def test_an_error_key_that_is_no_envelope_ends_the_stream_closed_whatever_the_pieces(
    mode: str, engine: Any, error: Any
) -> None:
    first, bad, rest, request_id = terminal_wire(error)
    reply = await stream_reply(mode, engine, pieces(first + bad + rest, 1), headers={'x-request-id': 'req_h'})
    out = await drain(reply)
    assert b'after' not in out and b'[DONE]' not in out and engine.upstream_closed()
    assert out.startswith(first) and last_event(out) == generic(request_id)


async def test_the_last_event_without_its_blank_line_with_a_bad_error_key_ends_closed(mode: str, engine: Any) -> None:
    wire = frame(chunk('a')).encode() + b'data: {"error": "boom"}'
    out = await drain(await stream_reply(mode, engine, [wire], headers={'x-request-id': 'req_h'}))
    assert b'boom' not in out
    assert out.startswith(frame(chunk('a')).encode()) and last_event(out) == generic('req_h')


async def test_a_null_error_key_is_an_ordinary_chunk(mode: str, engine: Any) -> None:
    # The engine builds every chunk from fixed keys and never sends an `error` on one; a null is no failure
    wire = frame({**chunk('a'), 'error': None}).encode() + frame({'error': None}).encode() + frame('[DONE]').encode()
    assert await drain(await stream_reply(mode, engine, pieces(wire, 5))) == wire


@pytest.mark.parametrize('size', [1, 7, 100_000])
async def test_a_line_break_inside_one_event_is_not_an_event_boundary(mode: str, engine: Any, size: int) -> None:
    # The hidden envelope is split over two data lines joined by CRLF: CR and LF are one line ending, not two
    head, tail = json.dumps(envelope('key_revoked', 'req_crlf')).split('"code"')
    wire = f'data: {head}\r\ndata: "code"{tail}\r\n\r\n'.encode()
    out = await drain(await stream_reply(mode, engine, pieces(wire, size)))
    assert b'key_revoked' not in out and last_event(out) == generic('req_crlf')


async def test_cr_only_line_endings_still_end_events(mode: str, engine: Any) -> None:
    first = frame(chunk('a')).replace('\n', '\r').encode()
    wire = first + hidden_frame().replace(b'\n', b'\r')
    out = await drain(await stream_reply(mode, engine, pieces(wire, 5)))
    assert out.startswith(first) and b'insufficient_credit' not in out and last_event(out) == generic('req_mid')


async def test_a_byte_order_mark_cannot_hide_a_first_hidden_frame(mode: str, engine: Any) -> None:
    wire = b'\xef\xbb\xbf' + hidden_frame()
    out = await drain(await stream_reply(mode, engine, [wire[:1], wire[1:2], wire[2:]]))
    assert b'insufficient_credit' not in out and last_event(out) == generic('req_mid')


async def test_a_byte_order_mark_before_ordinary_frames_passes_through(mode: str, engine: Any) -> None:
    wire = b'\xef\xbb\xbf' + (frame(chunk('a')) + frame('[DONE]')).encode()
    assert await drain(await stream_reply(mode, engine, pieces(wire, 2))) == wire


async def test_a_last_event_without_its_blank_line_is_still_judged(mode: str, engine: Any) -> None:
    wire = frame(chunk('a')).encode() + f'data: {json.dumps(envelope("key_expired", "req_t"))}'.encode()
    out = await drain(await stream_reply(mode, engine, [wire]))
    assert out.startswith(frame(chunk('a')).encode()) and b'key_expired' not in out
    assert last_event(out) == generic('req_t')


async def test_a_last_done_without_its_blank_line_passes_through(mode: str, engine: Any) -> None:
    wire = frame(chunk('a')).encode() + b'data: [DONE]'
    assert await drain(await stream_reply(mode, engine, [wire])) == wire


async def test_a_hidden_frame_with_a_number_too_long_for_python_ints_is_still_rewritten(mode: str, engine: Any) -> None:
    # A browser's JSON.parse reads a 5,000-digit number; Python's int() refuses it. The judge must not care.
    hidden = envelope('key_revoked', 'req_n')
    text = json.dumps(hidden)[:-2] + ', "n": ' + '1' * 5000 + '}}'
    out = await drain(await stream_reply(mode, engine, [f'data: {text}\n\n'.encode()]))
    assert b'key_revoked' not in out and last_event(out) == generic('req_n')


async def test_a_cut_off_last_event_fails_closed(mode: str, engine: Any) -> None:
    cut = b'data: {"error": {"code": "key_expired", "message": "tru'
    out = await drain(
        await stream_reply(mode, engine, [frame(chunk('a')).encode(), cut], headers={'x-request-id': 'r'})
    )
    assert out.startswith(frame(chunk('a')).encode()) and b'key_expired' not in out
    assert last_event(out) == generic('r')


@pytest.mark.parametrize('constant', [b'NaN', b'Infinity', b'-Infinity'])
async def test_an_event_holding_a_non_standard_number_fails_closed(mode: str, engine: Any, constant: bytes) -> None:
    # Not JSON for Go or a browser's JSON.parse: it cannot be judged, so nothing after it is read or sent
    bad = b'data: {"choices": [{"delta": {"content": "late"}}], "x": ' + constant + b'}\n\n'
    parts = [frame(chunk('a')).encode(), bad, frame(chunk('b')).encode(), frame('[DONE]').encode()]
    out = await drain(await stream_reply(mode, engine, parts, headers={'x-request-id': 'r'}))
    assert out.startswith(frame(chunk('a')).encode()) and b'late' not in out and b'[DONE]' not in out
    assert last_event(out) == generic('r')
    assert engine.upstream_closed() and engine.parts_read() == 2


async def test_an_event_too_deep_to_judge_fails_closed(mode: str, engine: Any) -> None:
    deep = b'data: {"error": ' + b'[' * 100_000 + b']' * 100_000 + b'}\n\n'
    out = await drain(
        await stream_reply(mode, engine, [frame(chunk('a')).encode(), deep], headers={'x-request-id': 'r'})
    )
    assert out.startswith(frame(chunk('a')).encode()) and b'[[[[' not in out and last_event(out) == generic('r')


@pytest.mark.parametrize(
    'second',
    [b'data: {"error": {"code": "insufficient_credit", "message": "', frame(chunk('b')).encode()],
    ids=['inside-an-event', 'between-events'],
)
async def test_a_broken_upstream_ends_with_a_terminal_generic_frame(mode: str, engine: Any, second: bytes) -> None:
    parts = [frame(chunk('a')).encode(), second, b'never read']
    reply = await stream_reply(mode, engine, parts, drop_after=2, headers={'x-request-id': 'req_b'})
    out = await drain(reply)
    assert out.startswith(frame(chunk('a')).encode()) and b'insufficient_credit' not in out
    assert out.endswith(
        b'\n\n\n\n' + b'data: ' + json.dumps(generic('req_b'), separators=(',', ':')).encode() + b'\n\n'
    )
    assert engine.upstream_closed()


async def test_an_event_over_the_byte_cap_fails_closed_and_closes_upstream(mode: str, engine: Any) -> None:
    # The event never ends: reading stops as soon as it passes the cap, never at the end of the body
    head = frame(chunk('a')).encode()
    endless = b'data: {"choices": [{"delta": {"content": "' + b'y' * (2 * MAX_EVENT_BYTES)
    parts = [head, *pieces(endless, 1024 * 1024), frame('[DONE]').encode()]
    out = await drain(await stream_reply(mode, engine, parts, headers={'x-request-id': 'req_cap'}))
    assert out.startswith(head + b'\n\n') and b'yyyy' not in out and b'[DONE]' not in out
    assert last_event(out) == generic('req_cap') and len(out) < 1000
    assert engine.upstream_closed() and engine.parts_read() == 1 + MAX_EVENT_BYTES // (1024 * 1024) + 1


async def test_a_whole_event_over_the_byte_cap_in_one_piece_fails_closed(mode: str, engine: Any) -> None:
    huge = b'data: {"choices": [{"delta": {"content": "' + b'y' * MAX_EVENT_BYTES + b'"}}]}\n\n'
    out = await drain(await stream_reply(mode, engine, [huge, frame('[DONE]').encode()]))
    assert b'yyyy' not in out and b'[DONE]' not in out and last_event(out) == generic(None)


async def test_an_event_just_under_the_byte_cap_passes_through(mode: str, engine: Any) -> None:
    filler = MAX_EVENT_BYTES - len(b'data: {"choices": [{"delta": {"content": ""}}]}\n\n')
    big = b'data: {"choices": [{"delta": {"content": "' + b'y' * filler + b'"}}]}\n\n'
    assert len(big) == MAX_EVENT_BYTES
    wire = big + frame('[DONE]').encode()
    assert await drain(await stream_reply(mode, engine, pieces(wire, 1024 * 1024))) == wire


async def close_body(reply: Reply[Any]) -> None:
    # What an adapter does on every exit path: no getattr and no cast, the body type says which close it has
    body = reply.body
    if isinstance(body, ReplyBody):
        body.close()
    elif isinstance(body, AsyncReplyBody):
        await body.aclose()


async def test_a_streaming_body_is_the_closable_type_of_its_mode(mode: str, engine: Any) -> None:
    reply = await stream_reply(mode, engine, [frame('[DONE]').encode()])
    assert isinstance(reply.body, ReplyBody) is (mode == 'sync')
    assert isinstance(reply.body, AsyncReplyBody) is (mode == 'async')
    await close_body(reply)
    assert engine.upstream_closed()


async def test_a_buffered_body_is_bytes_and_neither_closable_type(mode: str, engine: Any) -> None:
    reply = await send(build(mode, engine), {'messages': MSG})
    assert isinstance(reply.body, bytes) and not isinstance(reply.body, (ReplyBody, AsyncReplyBody))


async def test_a_browser_disconnect_closes_upstream(mode: str, engine: Any) -> None:
    parts = [frame(chunk('a')).encode(), frame(chunk('b')).encode(), frame('[DONE]').encode()]
    reply = await stream_reply(mode, engine, parts)
    body: Any = reply.body
    first = next(body) if mode == 'sync' else await body.__anext__()
    assert not engine.upstream_closed()
    await close_body(reply)
    assert first == parts[0] and engine.upstream_closed()
    await close_body(reply)  # a second close is a no-op
    assert engine.upstream_closed()


async def test_a_body_closed_before_it_started_closes_upstream(mode: str, engine: Any) -> None:
    # The upstream response is open as soon as the Reply exists: a browser that leaves before the first read
    # (a framework that never iterates a body) must still release it
    parts = [frame(chunk('a')).encode(), frame('[DONE]').encode()]
    reply = await stream_reply(mode, engine, parts)
    assert not engine.upstream_closed()
    await close_body(reply)
    assert engine.upstream_closed() and engine.parts_read() == 0


async def test_a_body_closed_twice_before_it_started_is_safe_and_reads_nothing(mode: str, engine: Any) -> None:
    reply = await stream_reply(mode, engine, [frame(chunk('a')).encode(), frame('[DONE]').encode()])
    await close_body(reply)
    await close_body(reply)
    assert engine.upstream_closed() and await drain(reply) == b''


async def test_a_body_closed_after_it_finished_is_safe(mode: str, engine: Any) -> None:
    wire = frame(chunk('a')).encode() + frame('[DONE]').encode()
    reply = await stream_reply(mode, engine, [wire])
    assert await drain(reply) == wire
    await close_body(reply)
    await close_body(reply)
    assert engine.upstream_closed()


async def test_a_reply_closes_its_own_body(mode: str, engine: Any) -> None:
    reply = await stream_reply(mode, engine, [frame(chunk('a')).encode(), frame('[DONE]').encode()])
    if mode == 'sync':
        reply.close()
        reply.close()
    else:
        await reply.aclose()
        await reply.aclose()
    assert engine.upstream_closed() and engine.parts_read() == 0


async def test_closing_a_reply_with_a_bytes_body_does_nothing(mode: str, engine: Any) -> None:
    reply = await send(build(mode, engine), {'messages': MSG})
    reply.close()
    await reply.aclose()
    assert reply.status == 200 and payload(reply)['choices']


def test_a_sync_reply_body_narrows_without_casts(engine: Any) -> None:
    engine.next(Scripted(chunks=[frame('[DONE]').encode()]))
    client = Meridian(engine.public_url, engine.project_key, transport=engine.transport)
    reply: Reply[ReplyBody] = Forwarder[Any](client=client, authorize=allow_all).handle(
        None,
        method='POST',
        content_type='application/json',
        idempotency_key=None,
        read_body=lambda: json.dumps({'messages': MSG, 'stream': True}).encode(),
    )
    body = reply.body
    assert not isinstance(body, bytes)
    assert b''.join(body) == frame('[DONE]').encode()
    body.close()


async def test_an_async_reply_body_narrows_without_casts(engine: Any) -> None:
    engine.next(Scripted(chunks=[frame('[DONE]').encode()]))
    client = AsyncMeridian(engine.public_url, engine.project_key, transport=engine.transport)
    reply: Reply[AsyncReplyBody] = await AsyncForwarder[Any](client=client, authorize=allow_all).ahandle(
        None,
        method='POST',
        content_type='application/json',
        idempotency_key=None,
        read_body=lambda: json.dumps({'messages': MSG, 'stream': True}).encode(),
    )
    body = reply.body
    assert not isinstance(body, bytes)
    assert b''.join([piece async for piece in body]) == frame('[DONE]').encode()
    await body.aclose()


# --- The default client: one shared client per settings (and per event loop) --------------------------------------


def default_forwarder(mode: str) -> Fwd:
    if mode == 'sync':
        return Forwarder[Any](authorize=allow_all)
    return AsyncForwarder[Any](authorize=allow_all)


async def test_a_closed_shared_client_is_replaced_so_the_route_keeps_answering(
    mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # `with shared_client() as client:` closes the one client every default route uses: the next use builds a new one
    recorder = Recorder(monkeypatch)
    monkeypatch.setenv('MERIDIAN_URL', f'http://closed-{mode}.engine.test')
    monkeypatch.setenv('MERIDIAN_KEY', 'gmr_' + '7' * 64)
    fwd = default_forwarder(mode)
    assert (await send(fwd, {'messages': MSG})).status == 200
    if mode == 'sync':
        with shared_client() as old:
            pass
    else:
        async with async_shared_client() as aold:
            pass
        old = aold
    assert old.is_closed
    assert (await send(fwd, {'messages': MSG})).status == 200
    new = shared_client() if mode == 'sync' else async_shared_client()
    assert new is not old and not new.is_closed
    assert len(recorder.senders) == 2


async def test_without_a_client_the_forwarder_shares_one_built_from_the_environment(
    mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = Recorder(monkeypatch)
    monkeypatch.setenv('MERIDIAN_URL', f'http://shared-{mode}.engine.test')
    monkeypatch.setenv('MERIDIAN_KEY', 'gmr_' + '2' * 64)
    fwd = default_forwarder(mode)
    first = await send(fwd, {'messages': MSG})
    second = await send(fwd, {'messages': MSG})
    third = await send(default_forwarder(mode), {'messages': MSG})
    assert [first.status, second.status, third.status] == [200, 200, 200]
    assert len(recorder.senders) == 3 and len(set(recorder.senders)) == 1


async def test_a_missing_environment_at_start_up_never_poisons_the_shared_client(
    mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    recorder = Recorder(monkeypatch)
    monkeypatch.delenv('MERIDIAN_URL', raising=False)
    monkeypatch.delenv('MERIDIAN_KEY', raising=False)
    fwd = default_forwarder(mode)
    early = await send(fwd, {'messages': MSG})
    assert early.status == 503 and payload(early) == generic(None) and recorder.senders == []
    monkeypatch.setenv('MERIDIAN_URL', f'http://late-{mode}.engine.test')
    monkeypatch.setenv('MERIDIAN_KEY', 'gmr_' + '3' * 64)
    later = await send(fwd, {'messages': MSG})
    assert later.status == 200 and len(recorder.senders) == 1


async def test_other_settings_get_another_shared_client(mode: str, monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = Recorder(monkeypatch)
    monkeypatch.setenv('MERIDIAN_URL', f'http://one-{mode}.engine.test')
    monkeypatch.setenv('MERIDIAN_KEY', 'gmr_' + '4' * 64)
    await send(default_forwarder(mode), {'messages': MSG})
    monkeypatch.setenv('MERIDIAN_KEY', 'gmr_' + '5' * 64)
    await send(default_forwarder(mode), {'messages': MSG})
    assert len(set(recorder.senders)) == 2


UNUSABLE = [
    pytest.param({'url': None, 'key': None}, id='nothing-set'),
    pytest.param({'url': 'ftp://engine.test', 'key': 'gmr_' + '7' * 64}, id='bad-scheme'),
    pytest.param({'url': 'http://engine.test', 'key': ''}, id='empty-key'),
]


@pytest.mark.parametrize('settings', UNUSABLE)
def test_an_unusable_configuration_is_never_cached(settings: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    from meridian._client import shared_client

    monkeypatch.delenv('MERIDIAN_URL', raising=False)
    monkeypatch.delenv('MERIDIAN_KEY', raising=False)
    first, second = shared_client(**settings), shared_client(**settings)
    assert first is not second
    assert first.models.list().error is not None  # refused, nothing sent


@pytest.mark.parametrize('settings', UNUSABLE)
async def test_an_unusable_configuration_is_never_cached_for_async_either(
    settings: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    from meridian._client import async_shared_client

    monkeypatch.delenv('MERIDIAN_URL', raising=False)
    monkeypatch.delenv('MERIDIAN_KEY', raising=False)
    first, second = async_shared_client(**settings), async_shared_client(**settings)
    assert first is not second
    assert (await first.models.list()).error is not None


async def test_a_usable_configuration_is_cached_once_it_is_fixed(mode: str, monkeypatch: pytest.MonkeyPatch) -> None:
    from meridian._client import async_shared_client, shared_client

    make = shared_client if mode == 'sync' else async_shared_client
    monkeypatch.delenv('MERIDIAN_URL', raising=False)
    monkeypatch.delenv('MERIDIAN_KEY', raising=False)
    assert make() is not make()
    monkeypatch.setenv('MERIDIAN_URL', f'http://fixed-{mode}.engine.test')
    monkeypatch.setenv('MERIDIAN_KEY', 'gmr_' + '8' * 64)
    assert make() is make()


def test_closed_event_loops_do_not_pile_up_in_the_shared_async_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    from meridian import _client  # pyright: ignore[reportPrivateUsage]  (the cache itself is what is under test)

    Recorder(monkeypatch)
    monkeypatch.setenv('MERIDIAN_URL', 'http://pileup.engine.test')
    monkeypatch.setenv('MERIDIAN_KEY', 'gmr_' + '9' * 64)
    fwd = default_forwarder('async')
    loops: list[asyncio.AbstractEventLoop] = []  # held, as a framework may: a closed loop that is still referred to

    async def use() -> int:
        loops.append(asyncio.get_running_loop())
        assert (await send(fwd, {'messages': MSG})).status == 200
        return len(_client._async_shared)  # pyright: ignore[reportPrivateUsage]

    sizes = [asyncio.run(use()) for _ in range(4)]
    assert sizes == [1, 1, 1, 1] and all(loop.is_closed() for loop in loops)


def test_each_event_loop_gets_its_own_shared_async_client(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = Recorder(monkeypatch)
    monkeypatch.setenv('MERIDIAN_URL', 'http://loops.engine.test')
    monkeypatch.setenv('MERIDIAN_KEY', 'gmr_' + '6' * 64)
    fwd = default_forwarder('async')
    statuses = [asyncio.run(send(fwd, {'messages': MSG})).status for _ in range(2)]
    assert statuses == [200, 200] and len(set(recorder.senders)) == 2


# --- Closing while a read is waiting, and a read that ends badly ----------------------------------------------------


async def test_aclose_while_a_read_is_waiting_never_raises_and_closes_upstream(engine: Any) -> None:
    reply = await stream_reply('async', engine, [frame(chunk('a')).encode()], hang=True)
    body = cast('AsyncReplyBody', reply.body)
    seen: list[bytes] = []

    async def reader() -> None:
        async for piece in body:
            seen.append(piece)

    async with anyio.create_task_group() as group:
        group.start_soon(reader)
        await engine.hanging.wait()  # the reader now waits for a second piece inside the running relay
        assert engine.parts_read() == 1
        await body.aclose()  # used to raise: "aclose(): asynchronous generator is already running"
        assert engine.upstream_closed()
        await body.aclose()
        group.cancel_scope.cancel()  # the fake never ends a hung read by itself
    assert seen == [frame(chunk('a')).encode()]


class _SlowClose(httpx.AsyncByteStream):
    """An engine stream that sends one frame and then waits; closing it takes `seconds` (at least a checkpoint), as
    closing a connection does."""

    def __init__(self, seconds: float = 0) -> None:
        self.seconds, self.closed = seconds, False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield frame(chunk('a')).encode()
        await anyio.sleep_forever()

    async def aclose(self) -> None:
        await anyio.sleep(self.seconds)
        self.closed = True


async def slow_stream_body(engine: Any, held: _SlowClose) -> AsyncReplyBody:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, headers={'content-type': 'text/event-stream'}, stream=held)
    )
    client = AsyncMeridian(engine.public_url, engine.project_key, transport=transport)
    reply = await send(AsyncForwarder[Any](client=client, authorize=allow_all), {'messages': MSG, 'stream': True})
    return cast('AsyncReplyBody', reply.body)


async def test_a_relay_closed_inside_a_cancelled_scope_still_closes_upstream(engine: Any) -> None:
    # anyio cancellation is level-triggered: every checkpoint inside a cancelled scope raises again, so an unshielded
    # close would stop at its first checkpoint and leave the engine's stream open
    held = _SlowClose()
    body = await slow_stream_body(engine, held)
    with anyio.CancelScope() as scope:
        assert await body.__anext__() == frame(chunk('a')).encode()
        scope.cancel()
        await body.aclose()
    assert held.closed


async def test_the_relays_close_is_bounded(engine: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    # The bound is patched small and the slow close is finite, so an unbounded relay finishes late and fails the
    # assertions instead of hanging the suite
    monkeypatch.setattr('meridian.forward.CLOSE_WITHIN', 0.2)
    held = _SlowClose(seconds=2)
    body = await slow_stream_body(engine, held)
    assert await body.__anext__() == frame(chunk('a')).encode()
    started = anyio.current_time()
    await body.aclose()
    assert anyio.current_time() - started < 1 and not held.closed  # given up after the bound, not after 2 s


@pytest.mark.parametrize('cancels', [1, 2], ids=['once', 'again-during-the-close'])
async def test_a_relay_cancelled_natively_during_aclose_still_closes_upstream(engine: Any, cancels: int) -> None:
    # asyncio's own `task.cancel()` is edge-triggered and lands inside the close at its first checkpoint, where a
    # shield of anyio's does not hold it: the close runs as a task of its own, and the cancellation is raised after
    held = _SlowClose(seconds=0.2)
    body = await slow_stream_body(engine, held)
    assert await body.__anext__() == frame(chunk('a')).encode()
    closing = asyncio.ensure_future(body.aclose())
    for _ in range(cancels):
        await asyncio.sleep(0.05)  # the close is under way
        closing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await closing
    assert held.closed


class _BlockingStream(httpx.SyncByteStream):
    """One piece, then a read that waits until the response is closed (as a real connection read does)."""

    def __init__(self) -> None:
        self.waiting, self.release, self.closed = threading.Event(), threading.Event(), False

    def __iter__(self) -> Any:
        yield frame(chunk('a')).encode()
        self.waiting.set()
        self.release.wait(10)

    def close(self) -> None:
        self.closed = True
        self.release.set()


def test_close_while_a_read_is_waiting_never_raises_and_closes_upstream(engine: Any) -> None:
    stream = _BlockingStream()
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, headers={'content-type': 'text/event-stream'}, stream=stream)
    )
    client = Meridian(engine.public_url, engine.project_key, transport=transport, max_retries=0)
    reply = Forwarder[Any](client=client, authorize=allow_all).handle(
        None,
        method='POST',
        content_type='application/json',
        idempotency_key=None,
        read_body=lambda: json.dumps({'messages': MSG, 'stream': True}).encode(),
    )
    body = cast('ReplyBody', reply.body)
    outcome: list[object] = []

    def read() -> None:
        try:
            outcome.extend(body)
        except BaseException as exc:  # a reader that fails is a failed test, not a hung one
            outcome.append(exc)

    reader = threading.Thread(target=read)
    reader.start()
    assert stream.waiting.wait(10)  # the reader is inside the running relay
    body.close()  # used to raise: "generator already executing"
    body.close()
    reader.join(10)
    assert not reader.is_alive() and stream.closed
    assert all(isinstance(piece, bytes) for piece in outcome)


async def test_an_exception_mid_iteration_propagates_and_closes_upstream(mode: str, engine: Any) -> None:
    parts = [frame(chunk('a')).encode(), frame(chunk('b')).encode()]
    reply = await stream_reply(mode, engine, parts, fail_after=1)
    body: Any = reply.body
    if mode == 'sync':
        assert next(body) == parts[0]
        with pytest.raises(RuntimeError, match='boom'):
            next(body)
    else:
        assert await body.__anext__() == parts[0]
        with pytest.raises(RuntimeError, match='boom'):
            await body.__anext__()
    assert engine.upstream_closed()
    await close_body(reply)  # a later close is a no-op
    await close_body(reply)
    assert engine.upstream_closed()


async def test_cancelling_a_read_mid_iteration_propagates_and_closes_upstream(engine: Any) -> None:
    reply = await stream_reply('async', engine, [frame(chunk('a')).encode()], hang=True)
    body = cast('AsyncReplyBody', reply.body)
    caught: list[type[BaseException]] = []

    async def read() -> None:
        try:
            async for _ in body:
                pass
        except BaseException as exc:
            caught.append(type(exc))
            raise

    with anyio.move_on_after(0.2) as scope:
        await read()
    assert scope.cancelled_caught and caught == [asyncio.CancelledError]
    assert engine.upstream_closed()
    await close_body(reply)  # a later close is a no-op
    await close_body(reply)


# --- The end-of-stream terminal frame keeps the frame's own id ------------------------------------------------------


async def test_the_last_event_without_its_blank_line_with_a_bad_error_keeps_its_own_request_id(
    mode: str, engine: Any
) -> None:
    wire = frame(chunk('a')).encode() + b'data: {"error":{"message":"m","request_id":"req_x"}}'
    out = await drain(await stream_reply(mode, engine, [wire], headers={'x-request-id': 'req_h'}))
    assert b'"m"' not in out and last_event(out) == generic('req_x')


# --- A request id only reaches a header, or the body, when it is printable ASCII ------------------------------------

BAD_IDS = [
    pytest.param('a\r\nset-cookie: x=1', id='crlf'),
    pytest.param('a\nb', id='lf'),
    pytest.param('x' * 256, id='too-long'),
    pytest.param('café', id='non-ascii'),
    # A header value cannot start or end with a space (a server strips it, so the id would change on the way)
    pytest.param('   ', id='spaces-only'),
    pytest.param(' a', id='leading-space'),
    pytest.param('a ', id='trailing-space'),
]


async def test_a_request_id_with_inner_spaces_is_kept(mode: str, engine: Any) -> None:
    engine.next(Scripted(status=429, headers={'x-request-id': 'req a b'}, json=envelope('rate_limited', 'req a b')))
    reply = await send(build(mode, engine), {'messages': MSG})
    assert reply.headers['x-request-id'] == 'req a b' and payload(reply)['error']['request_id'] == 'req a b'


def wire_id(request_id: str) -> str:
    """The header value as bytes, so even a non-ASCII id reaches the client (httpx accepts bytes at run time)."""
    return cast('str', request_id.encode())


def clean(reply: Reply[Any]) -> bool:
    return 'x-request-id' not in reply.headers and not any('\r' in v or '\n' in v for v in reply.headers.values())


@pytest.mark.parametrize('request_id', BAD_IDS)
async def test_an_end_user_refusal_with_a_bad_request_id_drops_it_from_header_and_body(
    mode: str, engine: Any, request_id: str
) -> None:
    body = envelope('rate_limited', request_id)
    engine.next(Scripted(status=429, headers={'x-request-id': wire_id(request_id)}, json=body))
    reply = await send(build(mode, engine), {'messages': MSG})
    assert reply.status == 429 and clean(reply)
    assert 'request_id' not in payload(reply)['error'] and b'set-cookie' not in cast('bytes', reply.body)


@pytest.mark.parametrize('request_id', BAD_IDS)
async def test_a_generic_503_with_a_bad_request_id_drops_it_from_header_and_body(
    mode: str, engine: Any, request_id: str
) -> None:
    engine.next(
        Scripted(
            status=402,
            headers={'x-request-id': wire_id(request_id)},
            json=envelope('insufficient_credit', request_id),
        )
    )
    reply = await send(build(mode, engine), {'messages': MSG})
    assert reply.status == 503 and clean(reply) and payload(reply) == generic(None)


@pytest.mark.parametrize('request_id', BAD_IDS)
async def test_a_stream_reply_header_and_frames_never_carry_a_bad_request_id(
    mode: str, engine: Any, request_id: str
) -> None:
    latin = wire_id(request_id)
    frame_with_bad_id = (
        f'data: {json.dumps({"error": {"message": "m", "code": "key_revoked", "request_id": request_id}})}\n\n'
    )
    wire = frame(chunk('a')).encode() + frame_with_bad_id.encode()
    reply = await stream_reply(mode, engine, [wire], headers={'x-request-id': latin})
    assert clean(reply)
    out = await drain(reply)
    assert last_event(out) == generic(None) and b'set-cookie' not in out


async def test_a_bad_frame_id_falls_back_to_the_responses_valid_id(mode: str, engine: Any) -> None:
    wire = frame(chunk('a')).encode() + hidden_frame('key_revoked').replace(b'req_mid', b'a\\r\\nb')
    out = await drain(await stream_reply(mode, engine, [wire], headers={'x-request-id': 'req_h'}))
    assert last_event(out) == generic('req_h')


async def test_a_200_with_an_error_and_a_bad_id_takes_the_responses_valid_id(mode: str, engine: Any) -> None:
    error = {'message': 'm', 'code': 'internal', 'request_id': 'a\r\nset-cookie: x=1'}
    engine.next(Scripted(headers={'x-request-id': 'req_hdr'}, json={**completion('hey'), 'error': error}))
    reply = await send(build(mode, engine), {'messages': MSG})
    assert payload(reply) == generic('req_hdr') and reply.headers['x-request-id'] == 'req_hdr'


async def test_a_200_with_an_error_and_only_bad_ids_has_no_id_at_all(mode: str, engine: Any) -> None:
    error = {'message': 'm', 'code': 'internal', 'request_id': 'x' * 300}
    engine.next(Scripted(headers={'x-request-id': ''}, json={**completion('hey'), 'error': error}))
    reply = await send(build(mode, engine), {'messages': MSG})
    assert reply.status == 503 and payload(reply) == generic(None) and 'x-request-id' not in reply.headers


# --- response_type is checked when the forwarder is built --------------------------------------------------------


@pytest.mark.parametrize(
    'bad',
    [
        pytest.param(object, id='object'),
        pytest.param(bool, id='bool'),
        pytest.param(int, id='int'),
        pytest.param((), id='empty-tuple'),
        pytest.param((Reply, int), id='tuple-with-int'),
        pytest.param((Reply, object), id='tuple-with-object'),
        pytest.param((AppResponse, bool), id='tuple-with-bool'),
        pytest.param('Reply', id='string'),
        pytest.param(None, id='none'),
        pytest.param(Reply[Any](200), id='instance'),
        pytest.param((Reply, 'AppResponse'), id='tuple-with-string'),
        pytest.param([Reply], id='list'),
        # A class that takes a verdict that is no response (True, False, 0, 1, None) for the app's own response
        pytest.param(type(None), id='none-type'),
        pytest.param(numbers.Number, id='number-abc'),
        pytest.param(Hashable, id='hashable-abc'),
        pytest.param((AppResponse, numbers.Integral), id='tuple-with-integral'),
        pytest.param(float | int, id='union-type'),
        # A generic alias is no class on any Python (3.10 still calls list[int] a type)
        pytest.param(list[int], id='generic-alias'),
        pytest.param((AppResponse, dict[str, Any]), id='tuple-with-generic-alias'),
        pytest.param(Reply[Any], id='parametrised-reply'),
    ],
)
def test_a_response_type_that_is_not_a_class_or_a_tuple_of_proper_classes_raises_at_construction(
    engine: Any, bad: Any
) -> None:
    sync = Meridian(engine.public_url, engine.project_key, transport=engine.transport)
    made = AsyncMeridian(engine.public_url, engine.project_key, transport=engine.transport)
    with pytest.raises(TypeError, match='response_type'):
        Forwarder[Any](client=sync, authorize=allow_all, response_type=bad)
    with pytest.raises(TypeError, match='response_type'):
        AsyncForwarder[Any](client=made, authorize=allow_all, response_type=bad)


def test_proper_response_types_are_accepted(engine: Any) -> None:
    client = Meridian(engine.public_url, engine.project_key, transport=engine.transport)
    for good in (Reply, AppResponse, (Reply, AppResponse), (MoreSpecificResponse,)):
        Forwarder[Any](client=client, authorize=allow_all, response_type=good)


# --- read_body, and the public names ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    'value',
    [
        pytest.param('{"messages": [{"role": "user", "content": "hi"}]}', id='str'),
        pytest.param(None, id='none'),
        pytest.param(12, id='int'),
        pytest.param(['x'], id='list'),
        pytest.param({'messages': MSG}, id='dict'),
    ],
)
async def test_a_read_body_that_does_not_give_bytes_is_400_and_never_raises(
    mode: str, engine: Any, value: object
) -> None:
    reply = await send(build(mode, engine), b'', read=lambda: value)
    assert reply.status == 400 and payload(reply)['error']['code'] == 'invalid_request' and engine.requests == []


async def test_a_read_body_that_gives_a_bytearray_is_accepted(mode: str, engine: Any) -> None:
    reply = await send(build(mode, engine), b'', read=lambda: bytearray(json.dumps({'messages': MSG}).encode()))
    assert reply.status == 200 and len(engine.requests) == 1


class _Source:
    """A body source that is an iterable of its own, not an iterator: it has a `close` that its iterator lacks (as a
    framework's request-body object does). `close_raises` makes closing fail."""

    def __init__(self, pieces: list[bytes], *, close_raises: bool = False) -> None:
        self.pieces, self.close_raises, self.closed = pieces, close_raises, 0

    def __iter__(self) -> Iterator[bytes]:
        return iter(self.pieces)

    async def _gen(self) -> AsyncIterator[bytes]:
        for piece in self.pieces:
            yield piece

    def __aiter__(self) -> AsyncIterator[bytes]:
        return self._gen()

    def close(self) -> None:
        self.closed += 1
        if self.close_raises:
            raise RuntimeError('close failed')

    async def aclose(self) -> None:
        self.close()


@pytest.mark.parametrize('close_raises', [False, True])
@pytest.mark.parametrize('over', [False, True])
async def test_the_iterable_read_body_returned_is_closed_on_success_and_on_413(
    mode: str, engine: Any, over: bool, close_raises: bool
) -> None:
    raw = json.dumps({'messages': MSG}).encode()
    pieces = [b'x' * (MAX_BODY_BYTES + 1)] if over else [raw[:5], raw[5:]]
    source = _Source(pieces, close_raises=close_raises)
    reply = await send(build(mode, engine), b'', read=lambda: source)
    # Closing is best-effort: a close that raises changes neither the answer nor the promise never to raise
    assert source.closed == 1
    assert reply.status == (413 if over else 200) and (len(engine.requests) == 0) == over


async def test_a_read_body_that_is_its_own_iterator_is_closed_once(mode: str, engine: Any) -> None:
    closes: list[str] = []

    class Gen:
        def __init__(self) -> None:
            self.left = [json.dumps({'messages': MSG}).encode()]

        def __iter__(self) -> 'Gen':
            return self

        def __next__(self) -> bytes:
            if not self.left:
                raise StopIteration
            return self.left.pop()

        def __aiter__(self) -> 'Gen':
            return self

        async def __anext__(self) -> bytes:
            if not self.left:
                raise StopAsyncIteration
            return self.left.pop()

        def close(self) -> None:
            closes.append('close')

        async def aclose(self) -> None:
            closes.append('aclose')

    reply = await send(build(mode, engine), b'', read=Gen)
    assert reply.status == 200 and closes == ['close' if mode == 'sync' else 'aclose']


def test_the_module_lists_its_public_names() -> None:
    import meridian.forward as forward

    # Only what an adapter or an app needs is listed; the constants stay importable for those who look
    expected = {
        'DEFAULT_MAX_TOKENS_CAP',
        'AllowRule',
        'AsyncForwarder',
        'AsyncReplyBody',
        'Forwarder',
        'Reply',
        'ReplyBody',
    }
    assert set(forward.__all__) == expected and all(hasattr(forward, name) for name in forward.__all__)
    for name in ('APP_CREDENTIAL_CODES', 'GENERIC_MESSAGE', 'MAX_BODY_BYTES', 'MAX_BODY_DEPTH', 'MAX_EVENT_BYTES'):
        assert hasattr(forward, name)


def test_the_engine_versions_are_exported() -> None:
    import meridian

    assert re.fullmatch(r'[0-9a-f]{12}', meridian.ENGINE_SPEC_COMMIT)
    assert meridian.ENGINE_API_VERSION and isinstance(meridian.ENGINE_API_VERSION, str)
    assert {'ENGINE_API_VERSION', 'ENGINE_SPEC_COMMIT'} <= set(meridian.__all__)


# --- Stream content the guard cannot judge fails closed ---------------------------------------------------------------

FIRST = frame(chunk('a')).encode()


def closed_stream(request_id: str = 'req_h') -> bytes:
    """The whole stream a browser sees when the guard gives up after one good chunk."""
    return FIRST + b'\n\n' + b'data: ' + json.dumps(generic(request_id), separators=(',', ':')).encode() + b'\n\n'


@pytest.mark.parametrize(
    'bad',
    [
        pytest.param(b'event: error\ndata: {"id": "x", "choices": []}\n\n', id='event-error'),
        pytest.param(b'event: ping\r\ndata: {"choices": []}\r\n\r\n', id='event-other-crlf'),
        pytest.param(b'event: message\nevent: error\ndata: {"choices": []}\n\n', id='event-second-field'),
        pytest.param(b'data: hello\n\n', id='data-not-json'),
        pytest.param(b'data: {"choices":\n\n', id='data-truncated-json'),
        pytest.param(b'data: [1, 2]\n\n', id='data-list'),
        pytest.param(b'data: "text"\n\n', id='data-string'),
        pytest.param(b'data: 5\n\n', id='data-number'),
        pytest.param(b'data: NaN\n\n', id='data-nan'),
        pytest.param(b'data: null\n\n', id='data-null'),
        pytest.param(b'data: true\n\n', id='data-true'),
        pytest.param(b'data: [DONE] \n\n', id='data-done-with-a-space'),
        pytest.param(b'data:\n\n', id='data-empty'),
        pytest.param(b'foo: bar\ndata: {"choices": []}\n\n', id='unknown-field'),
        pytest.param(b'{\n\n"choices": []}\n\n', id='a-json-body-cut-by-a-blank-line'),
    ],
)
async def test_content_the_guard_cannot_judge_ends_the_stream_closed(mode: str, engine: Any, bad: bytes) -> None:
    after = frame(chunk('after')).encode() + frame('[DONE]').encode()
    reply = await stream_reply(mode, engine, [FIRST, bad, after], headers={'x-request-id': 'req_h'})
    out = await drain(reply)
    assert b'after' not in out and b'[DONE]' not in out
    assert out == closed_stream()
    assert engine.upstream_closed() and engine.parts_read() == 2


@pytest.mark.parametrize('size', [1, 5])
async def test_content_the_guard_cannot_judge_ends_the_stream_closed_whatever_the_pieces(
    mode: str, engine: Any, size: int
) -> None:
    wire = FIRST + b'event: error\ndata: {"choices": []}\n\n' + frame(chunk('after')).encode()
    out = await drain(await stream_reply(mode, engine, pieces(wire, size), headers={'x-request-id': 'req_h'}))
    assert out == closed_stream() and engine.upstream_closed()


@pytest.mark.parametrize(
    'rest',
    [
        pytest.param(b'{"choices": []}', id='json-body'),
        pytest.param(b'{"error": {"code": "insufficient_credit", "message": "m"}}', id='json-error-body'),
        pytest.param(b'{\n  "choices": []\n}\n', id='pretty-json-body'),
        pytest.param(b'<html>bad gateway</html>', id='html'),
        pytest.param(b': ping', id='unterminated-comment'),
        pytest.param(b'id: 7', id='unterminated-id'),
        pytest.param(b'event: message', id='unterminated-event'),
    ],
)
async def test_an_upstream_body_that_ends_without_any_data_line_ends_the_stream_closed(
    mode: str, engine: Any, rest: bytes
) -> None:
    out = await drain(await stream_reply(mode, engine, [FIRST, rest], headers={'x-request-id': 'req_h'}))
    assert out == closed_stream() and engine.upstream_closed()


async def test_a_non_sse_json_body_as_the_whole_reply_to_a_stream_request_ends_closed(mode: str, engine: Any) -> None:
    body = json.dumps(completion('secret answer')).encode()
    out = await drain(await stream_reply(mode, engine, [body], headers={'x-request-id': 'req_h'}))
    assert b'secret answer' not in out and out.endswith(b'\n\n') and last_event(out) == generic('req_h')


# --- Case variants of the members the guard reads fail closed (a case-insensitive reader would see another body) ----

P1 = {
    'error': {'message': 'm', 'type': 'rate_limit_error', 'code': 'insufficient_credit', 'request_id': 'req_p1'},
    'Error': None,
    'choices': [{'index': 0, 'delta': {'content': 'x'}, 'finish_reason': None}],
}
P1B = {'error': {'message': 'm', 'type': 'rate_limit_error', 'code': 'insufficient_credit', 'Code': 'rate_limited'}}


# A known end-user code whose case variant names the app's credit: a case-insensitive reader could take either
P1B_REVERSED = {'error': {'message': 'm', 'type': 'rate_limit_error', 'code': 'rate_limited', 'Code': 'key_revoked'}}


@pytest.mark.parametrize(
    'body', [pytest.param(P1, id='P1'), pytest.param(P1B, id='P1b'), pytest.param(P1B_REVERSED, id='P1b-reversed')]
)
@pytest.mark.parametrize('status', [200, 402, 429])
async def test_a_reply_with_a_case_variant_is_a_generic_503(mode: str, engine: Any, body: Any, status: int) -> None:
    engine.next(Scripted(status=status, headers={'x-request-id': 'req_h'}, json=body))
    reply = await send(build(mode, engine), {'messages': MSG})
    assert reply.status == 503 and payload(reply)['error']['code'] == 'service_unavailable'
    body = cast('bytes', reply.body)
    assert b'insufficient_credit' not in body and b'rate_limited' not in body and b'key_revoked' not in body


LONG_S_BODY = {
    'error': {
        'message': 'm',
        'me' + chr(0x17F) * 2 + 'age': 'x',
        'type': 'rate_limit_error',
        'code': 'insufficient_credit',
    }
}


@pytest.mark.parametrize('status', [200, 402])
async def test_a_reply_with_a_long_s_case_variant_is_a_generic_503(mode: str, engine: Any, status: int) -> None:
    # Go's strings.EqualFold folds U+017F (long s) to s: `me<long s><long s>age` is `message` to such a reader
    engine.next(Scripted(status=status, headers={'x-request-id': 'req_h'}, json=LONG_S_BODY))
    reply = await send(build(mode, engine), {'messages': MSG})
    assert reply.status == 503 and payload(reply) == generic('req_h')


@pytest.mark.parametrize(
    'bad', [pytest.param(P1, id='P1'), pytest.param(P1B, id='P1b'), pytest.param(LONG_S_BODY, id='long-s')]
)
async def test_a_frame_with_a_case_variant_ends_the_stream_closed(mode: str, engine: Any, bad: Any) -> None:
    after = frame(chunk('after')).encode() + frame('[DONE]').encode()
    reply = await stream_reply(mode, engine, [FIRST, frame(bad).encode(), after], headers={'x-request-id': 'req_h'})
    out = await drain(reply)
    assert b'after' not in out and b'insufficient_credit' not in out
    assert last_event(out)['error']['code'] == 'service_unavailable' and engine.upstream_closed()


async def test_a_chunk_whose_delta_holds_a_key_named_error_passes_byte_for_byte(mode: str, engine: Any) -> None:
    odd = chunk('b')
    odd['choices'][0]['delta']['Error'] = {'Code': 'insufficient_credit'}
    wire = FIRST + frame(odd).encode() + frame('[DONE]').encode()
    assert await drain(await stream_reply(mode, engine, [wire])) == wire


async def test_comments_ids_retry_and_message_events_still_pass_byte_for_byte(mode: str, engine: Any) -> None:
    wire = (
        b': keepalive\n\n'
        + b'id: 7\nretry: 100\n: note\n'
        + FIRST
        + b'event: message\n'
        + FIRST
        + b'event:\r\ndata:'
        + json.dumps(chunk('b')).encode()
        + b'\r\n\r\n'
        + b'data: [DONE]\n\n'
    )
    assert await drain(await stream_reply(mode, engine, pieces(wire, 3))) == wire


async def test_a_trailing_comment_event_with_its_blank_line_passes(mode: str, engine: Any) -> None:
    wire = FIRST + b'data: [DONE]\n\n' + b': done\n\n'
    assert await drain(await stream_reply(mode, engine, [wire])) == wire
