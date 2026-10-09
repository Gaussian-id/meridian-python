from typing import Any

import httpx
import pytest

from meridian import AsyncMeridian, Meridian, MeridianError, Ok
from meridian._errors import error_from_body
from tests.support.fake_engine import Scripted, completion

MSG: Any = [{'role': 'user', 'content': 'hi'}]


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


def test_http_client_together_with_transport_is_refused(engine: Any) -> None:
    with httpx.Client() as http:
        client = Meridian(engine.public_url, engine.project_key, http_client=http, transport=engine.transport)
        result = client.chat.create({'messages': MSG})
    assert result.error is not None
    assert (result.error.code, result.error.type, result.error.retryable) == ('invalid_request', 'client_error', False)
    assert engine.requests == []
