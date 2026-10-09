from typing import Any

import httpx
import pytest

from meridian import Meridian
from meridian._errors import error_from_body
from tests.support.fake_engine import Scripted

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


def test_an_unknown_code_with_a_5xx_status_is_internal_and_not_retryable() -> None:
    text = '{"error":{"message":"m","type":"api_error","code":"brand_new","request_id":"r"}}'
    error = error_from_body(503, None, text)
    assert (error.code, error.status, error.retryable) == ('internal', 503, False)


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
