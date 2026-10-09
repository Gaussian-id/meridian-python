import gzip
import json
from collections.abc import AsyncIterator, Callable
from typing import Any, cast

import pytest

from meridian import AsyncMeridian, Meridian
from meridian.forward import APP_CREDENTIAL_CODES, MAX_EVENT_BYTES, AllowRule, AsyncForwarder, Forwarder, Reply
from tests.support.fake_engine import Scripted, chunk, completion, engine_error, envelope, frame
from tests.support.modes import resolve

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
) -> Fwd:
    if mode == 'sync':
        made = (
            client
            if client is not None
            else (lambda: Meridian(engine.public_url, engine.project_key, transport=engine.transport, max_retries=0))
        )
        return Forwarder[Any](client=made, authorize=authorize, allow=allow)
    made = (
        client
        if client is not None
        else (lambda: AsyncMeridian(engine.public_url, engine.project_key, transport=engine.transport, max_retries=0))
    )
    return AsyncForwarder[Any](client=made, authorize=authorize, allow=allow)


async def send(
    fwd: Fwd,
    body: Any,
    *,
    method: str = 'POST',
    content_type: str | None = 'application/json',
    key: str | None = None,
    request: Any = None,
    read: Callable[[], bytes] | None = None,
) -> Reply:
    raw = body if isinstance(body, bytes) else json.dumps(body).encode()
    reader = read or (lambda: raw)
    if isinstance(fwd, Forwarder):
        return fwd.handle(request, method=method, content_type=content_type, idempotency_key=key, read_body=reader)

    async def aread() -> bytes:
        return reader()

    return await fwd.ahandle(request, method=method, content_type=content_type, idempotency_key=key, read_body=aread)


def payload(reply: Reply) -> dict[str, Any]:
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


async def drain(reply: Reply) -> bytes:
    body: Any = reply.body
    if isinstance(body, bytes):
        return body
    if isinstance(body, AsyncIterator):
        stream = cast('AsyncIterator[bytes]', body)
        return b''.join([piece async for piece in stream])
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


@pytest.mark.parametrize('verdict', [False, None, 1, 'yes'])
async def test_a_verdict_that_is_not_true_is_403_unless_it_is_an_object_response(
    mode: str, engine: Any, verdict: object
) -> None:
    reply = await send(build(mode, engine, authorize=lambda r: verdict), {'messages': MSG})
    if isinstance(verdict, (bool, type(None))):
        assert reply.status == 403 and reply.passthrough is None
        assert payload(reply)['error'] == {
            'message': 'Not allowed to use this route.',
            'type': 'permission_error',
            'code': 'permission_denied',
        }
    else:
        assert reply.passthrough == verdict
    assert engine.requests == []


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


async def test_the_apps_own_response_is_passed_through(mode: str, engine: Any) -> None:
    mine = object()
    reply = await send(build(mode, engine, authorize=lambda r: mine), {'messages': MSG})
    assert reply.passthrough is mine and engine.requests == []


async def test_authorize_receives_the_request(mode: str, engine: Any) -> None:
    seen: list[object] = []
    marker = object()
    await send(build(mode, engine, authorize=lambda r: seen.append(r) is None), {'messages': MSG}, request=marker)
    assert seen == [marker]


@pytest.mark.parametrize(
    'body',
    [
        b'{not json',
        b'null',
        b'[]',
        b'"x"',
        b'\xff\xfe',
        json.dumps({'messages': []}).encode(),
        json.dumps({'messages': 'hi'}).encode(),
        b'{"messages": [{"role": "user", "content": "hi"}], "temperature": NaN}',
        b'{"messages": [{"role": "user", "content": "hi"}], "x": 1e400}',
        b'{"messages": [{"role": "user", "content": "hi"}], "x": ' + b'[' * 100_000 + b']' * 100_000 + b'}',
    ],
    ids=['broken', 'null', 'list', 'string', 'not-utf8', 'no-messages', 'text-messages', 'nan', 'inf', 'too-deep'],
)
async def test_a_bad_body_is_400_and_never_sent(mode: str, engine: Any, body: bytes) -> None:
    reply = await send(build(mode, engine), body)
    assert reply.status == 400 and payload(reply)['error']['code'] == 'invalid_request' and engine.requests == []


async def test_a_body_that_cannot_be_read_is_400(mode: str, engine: Any) -> None:
    def broken() -> bytes:
        raise OSError('the browser went away')

    reply = await send(build(mode, engine), b'', read=broken)
    assert reply.status == 400 and engine.requests == []


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
        ('x' * 255, True),
        ('', False),
        ('x' * 300, False),
        ('ü', False),
        ('   ', False),
        ('k\n', False),
        ('k\tx', False),
    ],
)
async def test_the_browser_idempotency_key(mode: str, engine: Any, key: str, forwarded: bool) -> None:
    await send(build(mode, engine), {'messages': MSG}, key=key)
    sent = engine.requests[0].headers['idempotency-key']
    assert (sent == key) is forwarded and 1 <= len(sent) <= 255


async def test_spaces_around_the_browser_key_are_trimmed_so_it_can_be_sent(mode: str, engine: Any) -> None:
    await send(build(mode, engine), {'messages': MSG}, key=' k-1 ')
    assert engine.requests[0].headers['idempotency-key'] == 'k-1'


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


async def test_the_client_is_built_per_request_not_at_creation(mode: str, engine: Any) -> None:
    calls: list[int] = []
    cls = Meridian if mode == 'sync' else AsyncMeridian

    def factory() -> Any:
        calls.append(1)
        return cls(engine.public_url, engine.project_key, transport=engine.transport)

    fwd = build(mode, engine, client=factory)
    assert calls == []
    await send(fwd, {'messages': MSG})
    await send(fwd, {'messages': MSG})
    assert len(calls) == 2


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


# --- Streaming ------------------------------------------------------------------------------------------------------


async def stream_reply(mode: str, engine: Any, parts: list[bytes], **scripted: Any) -> Reply:
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


async def test_a_browser_disconnect_closes_upstream(mode: str, engine: Any) -> None:
    parts = [frame(chunk('a')).encode(), frame(chunk('b')).encode(), frame('[DONE]').encode()]
    reply = await stream_reply(mode, engine, parts)
    body: Any = reply.body
    if mode == 'sync':
        first = next(body)
        body.close()
    else:
        first = await body.__anext__()
        await body.aclose()
    assert first == parts[0] and engine.upstream_closed()
