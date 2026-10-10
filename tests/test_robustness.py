from collections.abc import Callable
from typing import Any, cast

import httpx
import pytest

from meridian import AsyncMeridian, Meridian, MeridianError, Ok
from meridian._errors import error_from_body
from meridian.admin import AsyncMeridianAdmin, MeridianAdmin
from tests.support.fake_engine import Scripted, completion
from tests.support.modes import make_client, resolve

MSG: Any = [{'role': 'user', 'content': 'hi'}]
LONG_S = chr(0x17F)  # the long s: Go's strings.EqualFold folds it to s

pytestmark = pytest.mark.fake_only


@pytest.mark.parametrize('url', ['http://[::1', 'http://host:abc', 'http://host:99999'])
def test_a_malformed_url_is_a_config_error_never_a_raise(url: str, engine: Any) -> None:
    result = Meridian(url, 'k', transport=engine.transport).chat.create({'messages': MSG})
    assert result.error is not None
    assert (result.error.code, result.error.type, result.error.retryable) == ('invalid_request', 'client_error', False)
    assert engine.requests == []


def test_a_non_ascii_idempotency_key_is_an_error_value(engine: Any) -> None:
    client = Meridian(engine.public_url, engine.project_key, transport=engine.transport)
    result = client.chat.create({'messages': MSG}, idempotency_key='clé-é')
    assert result.error is not None
    assert result.error.type == 'client_error'
    assert result.error.retryable is False


@pytest.mark.parametrize('body', ['null', '[1, 2]', '"text"', '3'])
def test_a_2xx_body_that_is_not_an_object_is_a_network_error(body: str, engine: Any) -> None:
    engine.next(Scripted(text=body, headers={'content-type': 'application/json'}))
    result = Meridian(engine.public_url, engine.project_key, transport=engine.transport).chat.create({'messages': MSG})
    assert result.error is not None
    assert (result.error.code, result.error.status) == ('network_error', 200)


@pytest.mark.parametrize('status', [200, 201])
@pytest.mark.parametrize('body', ['null', ' null\n'])
def test_a_literal_null_models_reply_is_a_network_error(status: int, body: str, engine: Any) -> None:
    engine.next(Scripted(status=status, text=body, headers={'content-type': 'application/json'}))
    result = Meridian(engine.public_url, engine.project_key, transport=engine.transport).models.list()
    assert result.error is not None
    assert (result.error.code, result.error.type, result.error.status) == ('network_error', 'client_error', status)


@pytest.mark.parametrize('call', ['chat', 'models'])
def test_a_204_where_a_body_is_expected_is_a_network_error_never_ok_none(call: str, engine: Any) -> None:
    engine.next(Scripted(status=204))
    client = Meridian(engine.public_url, engine.project_key, transport=engine.transport)
    result = client.chat.create({'messages': MSG}) if call == 'chat' else client.models.list()
    assert result.error is not None
    assert (result.error.code, result.error.status) == ('network_error', 204)
    assert len(engine.requests) == 1


@pytest.mark.anyio
@pytest.mark.parametrize('call', ['chat', 'models'])
async def test_a_204_and_a_null_are_network_errors_when_async_too(call: str, engine: Any) -> None:
    client = AsyncMeridian(engine.public_url, engine.project_key, transport=engine.transport)
    for scripted in (Scripted(status=204), Scripted(text='null', headers={'content-type': 'application/json'})):
        engine.next(scripted)
        result = await (client.chat.create({'messages': MSG}) if call == 'chat' else client.models.list())
        assert result.error is not None
        assert result.error.code == 'network_error'


def test_an_unknown_code_with_a_5xx_status_is_internal_and_not_retryable() -> None:
    text = '{"error":{"message":"m","type":"api_error","code":"brand_new","request_id":"r"}}'
    error = error_from_body(503, None, text)
    assert (error.code, error.status, error.retryable) == ('internal', 503, False)


def test_an_unknown_code_keeps_the_engines_own_code_and_a_known_one_does_not() -> None:
    unknown = '{"error":{"message":"m","type":"api_error","code":"brand_new","request_id":"r"}}'
    known = '{"error":{"message":"m","type":"api_error","code":"internal","request_id":"r"}}'
    new = error_from_body(402, 'h', unknown)
    assert (new.code, new.engine_code, new.message) == ('internal', 'brand_new', 'm (engine code: brand_new)')
    assert error_from_body(500, None, known).engine_code is None
    assert error_from_body(502, None, 'not an envelope').engine_code is None


def test_a_results_response_request_id_is_not_part_of_its_identity(engine: Any) -> None:
    engine.next(Scripted(headers={'x-request-id': 'req_ok'}, json=completion()))
    result = Meridian(engine.public_url, engine.project_key, transport=engine.transport).chat.create({'messages': MSG})
    assert isinstance(result, Ok) and result.request_id == 'req_ok'
    assert Ok(1) == Ok(1, request_id='r') and 'req_ok' not in repr(result)


def test_engine_code_changes_neither_equality_hash_nor_repr() -> None:
    plain = MeridianError('internal', 'api_error', 500, 'm', False, 'r')
    marked = MeridianError('internal', 'api_error', 500, 'm', False, 'r', 'brand_new')
    assert plain == marked and hash(plain) == hash(marked)
    assert 'brand_new' not in repr(marked) and 'engine_code' not in repr(marked)
    assert marked.engine_code == 'brand_new' and plain.engine_code is None


def test_a_retryable_false_in_the_envelope_turns_retries_off() -> None:
    text = '{"error":{"message":"m","type":"api_error","code":"rate_limited","request_id":"r"},"retryable":false}'
    nested = '{"error":{"message":"m","type":"api_error","code":"rate_limited","request_id":"r","retryable":false}}'
    assert error_from_body(429, None, nested).retryable is False
    assert error_from_body(429, None, text).retryable is True  # only the error object is read


def test_retryable_true_in_the_envelope_cannot_turn_retries_on() -> None:
    text = '{"error":{"message":"m","type":"api_error","code":"invalid_request","request_id":"r","retryable":true}}'
    assert error_from_body(400, None, text).retryable is False


def test_an_envelope_cannot_claim_the_sdk_only_client_error_type() -> None:
    text = '{"error":{"message":"m","type":"client_error","code":"internal","request_id":"r"}}'
    assert error_from_body(500, None, text).type == 'api_error'


@pytest.mark.anyio
@pytest.mark.parametrize('depth', [600, 5000])
async def test_a_deeply_nested_chat_reply_is_a_result_or_a_network_error_never_a_raise(
    mode: str, engine: Any, depth: int
) -> None:
    text = '{"choices": [], "extra": ' + '[' * depth + ']' * depth + '}'
    engine.next(Scripted(text=text, headers={'content-type': 'application/json'}))
    result = await resolve(make_client(mode, engine, max_retries=0).chat.create({'messages': MSG}))
    if depth == 600 or result.error is None:
        assert result.error is None and result.data['choices'] == []
    else:
        assert (result.error.code, result.error.status) == ('network_error', 200)


@pytest.mark.anyio
async def test_a_reply_too_deep_to_decode_says_so_without_calling_it_invalid_json(mode: str, engine: Any) -> None:
    text = '{"choices": [], "extra": ' + '[' * 100_000 + ']' * 100_000 + '}'
    engine.next(Scripted(text=text, headers={'content-type': 'application/json'}))
    result = await resolve(make_client(mode, engine, max_retries=0).chat.create({'messages': MSG}))
    assert (result.error.code, result.error.status) == ('network_error', 200)
    assert result.error.message == 'The response could not be decoded.'


@pytest.mark.anyio
@pytest.mark.parametrize('call', ['create', 'stream'])
@pytest.mark.parametrize('params', cast('list[Any]', [None, 'messages', 5, [('messages', [])], ['messages'], object()]))
async def test_chat_params_that_are_no_mapping_are_an_error_value_and_nothing_is_sent(
    mode: str, engine: Any, call: str, params: Any
) -> None:
    chat = make_client(mode, engine).chat
    result = await resolve(getattr(chat, call)(params))
    assert result.data is None
    assert (result.error.code, result.error.type, result.error.retryable) == ('invalid_request', 'client_error', False)
    assert engine.requests == []


ERROR_MEMBERS = [
    pytest.param('boom', None, id='string'),
    pytest.param({}, None, id='empty-object'),
    pytest.param(False, None, id='false'),
    pytest.param({'message': 'm', 'code': 'insufficient_credit', 'request_id': 'req_err'}, 'req_err', id='envelope'),
    pytest.param({'message': 'm', 'code': 'internal', 'request_id': 'a\r\nb'}, None, id='envelope-bad-id'),
    pytest.param({'message': 'm', 'code': 'internal', 'request_id': 7}, None, id='envelope-number-id'),
]


@pytest.mark.anyio
@pytest.mark.parametrize(('error', 'own_id'), ERROR_MEMBERS)
@pytest.mark.parametrize('call', ['chat', 'models'])
async def test_a_2xx_reply_with_an_error_member_is_a_network_error(
    mode: str, engine: Any, call: str, error: Any, own_id: str | None
) -> None:
    engine.next(Scripted(status=200, headers={'x-request-id': 'req_hdr'}, json={**completion('hey'), 'error': error}))
    client = make_client(mode, engine)
    result = await resolve(client.chat.create({'messages': MSG}) if call == 'chat' else client.models.list())
    assert result.data is None
    assert (result.error.code, result.error.type, result.error.status) == ('network_error', 'client_error', 200)
    assert result.error.request_id == (own_id or 'req_hdr')
    assert 'hey' not in repr(result)


@pytest.mark.anyio
@pytest.mark.parametrize('call', ['chat', 'models'])
async def test_a_2xx_reply_with_a_null_error_member_is_a_result(mode: str, engine: Any, call: str) -> None:
    engine.next(Scripted(json={**completion('hey'), 'error': None}))
    client = make_client(mode, engine)
    result = await resolve(client.chat.create({'messages': MSG}) if call == 'chat' else client.models.list())
    assert result.error is None


@pytest.mark.anyio
async def test_a_chat_reply_over_64_mib_is_refused_without_reading_it_all(mode: str, engine: Any) -> None:
    piece = b' ' * (1 << 20)
    engine.next(Scripted(status=200, chunks=[b'{"choices": [], "a": "', *[piece] * 80, b'"}']))
    result = await resolve(make_client(mode, engine).chat.create({'messages': MSG}))
    assert (result.error.code, result.error.status) == ('network_error', 200)
    assert engine.parts_read() <= 66 and engine.upstream_closed()


@pytest.mark.anyio
@pytest.mark.parametrize(
    'scripted',
    [
        pytest.param(Scripted(chunks=[b'{"choices": [', b']}'], drop_after=1), id='dropped'),
        pytest.param(Scripted(chunks=[b'{"choices": [], "a": "', *[b' ' * (1 << 20)] * 80, b'"}']), id='too-large'),
        pytest.param(Scripted(text='null'), id='null'),
        pytest.param(Scripted(text='{not json'), id='not-json'),
    ],
)
async def test_a_reply_the_sdk_cannot_use_keeps_the_responses_request_id(
    mode: str, engine: Any, scripted: Scripted
) -> None:
    scripted.headers = {'x-request-id': 'req_hdr'}
    engine.next(scripted)
    result = await resolve(make_client(mode, engine, max_retries=0).chat.create({'messages': MSG}))
    assert (result.error.code, result.error.status, result.error.request_id) == ('network_error', 200, 'req_hdr')


@pytest.mark.anyio
async def test_an_error_body_over_1_mib_is_a_network_error_with_its_status(
    mode: str, engine: Any, clock: list[float]
) -> None:
    big = {'error': {'message': 'm' * (1 << 20), 'type': 'invalid_request_error', 'code': 'invalid_request'}}
    engine.next(Scripted(status=400, headers={'x-request-id': 'req_big'}, json=big))
    result = await resolve(make_client(mode, engine).chat.create({'messages': MSG}))
    assert (result.error.code, result.error.status, result.error.request_id) == ('network_error', 400, 'req_big')
    assert len(engine.requests) == 1


@pytest.mark.anyio
async def test_chat_params_holding_nan_or_infinity_are_refused_with_nothing_sent(mode: str, engine: Any) -> None:
    client = make_client(mode, engine)
    for number in (float('nan'), float('inf'), float('-inf')):
        for opened in (client.chat.create, client.chat.stream):
            result = await resolve(opened({'messages': MSG, 'temperature': number}))
            assert (result.error.code, result.error.type, result.error.retryable) == (
                'invalid_request',
                'client_error',
                False,
            )
    assert engine.requests == []


# Bodies a case-insensitive decoder (Go's encoding/json) reads one way and a browser's JSON.parse another: never judged
P1 = {
    'error': {'message': 'm', 'type': 'rate_limit_error', 'code': 'insufficient_credit', 'request_id': 'req_p1'},
    'Error': None,
    'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'hey'}, 'finish_reason': 'stop'}],
}
P1B = {'error': {'message': 'm', 'type': 'rate_limit_error', 'code': 'insufficient_credit', 'Code': 'rate_limited'}}
CASE_VARIANTS = [
    pytest.param(P1, id='P1'),
    pytest.param(P1B, id='P1b'),
    pytest.param({**completion('hey'), 'Error': 'x'}, id='top-level-Error'),
    pytest.param({**completion('hey'), 'ERROR': None}, id='top-level-ERROR-null'),
    pytest.param({**completion('hey'), 'CHOICES': []}, id='top-level-CHOICES'),
    # Go's strings.EqualFold folds U+017F (long s) to s, which lower() does not
    pytest.param({**completion('hey'), 'choice' + LONG_S: []}, id='top-level-choice-long-s'),
    pytest.param(
        {
            **completion('hey'),
            'error': {'message': 'm', 'me' + LONG_S * 2 + 'age': 'x', 'type': 'api_error', 'code': 'internal'},
        },
        id='message-long-s-inside-error',
    ),
]


@pytest.mark.anyio
@pytest.mark.parametrize('body', CASE_VARIANTS)
async def test_a_2xx_reply_with_a_case_variant_of_a_member_the_sdk_reads_fails_closed(
    mode: str, engine: Any, body: Any
) -> None:
    engine.next(Scripted(status=200, json=body))
    result = await resolve(make_client(mode, engine).chat.create({'messages': MSG}))
    assert (result.error.code, result.error.status) == ('network_error', 200)


@pytest.mark.anyio
async def test_an_error_body_with_a_case_variant_of_its_code_is_no_envelope(
    mode: str, engine: Any, clock: list[float]
) -> None:
    engine.next(Scripted(status=402, headers={'x-request-id': 'req_v'}, json=P1B))
    result = await resolve(make_client(mode, engine).chat.create({'messages': MSG}))
    assert (result.error.code, result.error.status, result.error.request_id) == ('network_error', 402, 'req_v')


@pytest.mark.anyio
async def test_case_variants_deeper_than_the_envelope_are_ordinary_content(mode: str, engine: Any) -> None:
    reply = completion('hey')
    reply['choices'][0]['message']['Error'] = {'Code': 'x'}
    reply['usage']['Error'] = None
    engine.next(Scripted(json=reply))
    result = await resolve(make_client(mode, engine).chat.create({'messages': MSG}))
    assert result.error is None and result.data['choices'][0]['message']['Error'] == {'Code': 'x'}


CLIENTS = {
    'Meridian': (Meridian, 'sync'),
    'AsyncMeridian': (AsyncMeridian, 'async'),
    'MeridianAdmin': (MeridianAdmin, 'sync'),
    'AsyncMeridianAdmin': (AsyncMeridianAdmin, 'async'),
}
WRONG_SETTINGS: dict[str, Callable[[str], dict[str, Any]]] = {
    'url-int': lambda _: {'url': 123},
    'url-bytes': lambda _: {'url': b'http://engine.test'},
    'key-int': lambda _: {'key': 123},
    'key-bytes': lambda _: {'key': b'gmr_x'},
    'http-client-str': lambda _: {'http_client': 'x', 'transport': None},
    'transport-str': lambda _: {'transport': 'x'},
    'http-client-other-mode': lambda kind: {
        'http_client': httpx.AsyncClient() if kind == 'sync' else httpx.Client(),
        'transport': None,
    },
    'transport-other-mode': lambda kind: {
        'transport': httpx.AsyncHTTPTransport() if kind == 'sync' else httpx.HTTPTransport()
    },
}


@pytest.mark.anyio
@pytest.mark.parametrize('setting', WRONG_SETTINGS)
@pytest.mark.parametrize('name', CLIENTS)
async def test_a_wrong_typed_setting_gives_a_refusing_client_never_a_raise(
    engine: Any, name: str, setting: str
) -> None:
    cls, kind = CLIENTS[name]
    admin = 'Admin' in name
    options: dict[str, Any] = {
        'url': engine.admin_url if admin else engine.public_url,
        'key': engine.integrator_key if admin else engine.project_key,
        'transport': engine.transport,
        **WRONG_SETTINGS[setting](kind),
    }
    client: Any = cls(**options)
    result = await resolve(client.orgs.list() if admin else client.models.list())
    assert (result.error.code, result.error.type, result.error.retryable) == ('invalid_request', 'client_error', False)
    assert engine.requests == []
    await resolve(client.close() if kind == 'sync' else client.aclose())


def test_http_client_together_with_transport_is_refused(engine: Any) -> None:
    with httpx.Client() as http:
        client = Meridian(engine.public_url, engine.project_key, http_client=http, transport=engine.transport)
        result = client.chat.create({'messages': MSG})
    assert result.error is not None
    assert (result.error.code, result.error.type, result.error.retryable) == ('invalid_request', 'client_error', False)
    assert engine.requests == []
