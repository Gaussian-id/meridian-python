import json
from typing import TYPE_CHECKING, Any

import anyio
import httpx
import pytest

from meridian import AsyncMeridian, Meridian, Ok
from tests.support.engine import make_engine
from tests.support.fake_engine import Scripted, completion, engine_error
from tests.support.modes import make_client, resolve

MSG: Any = [{'role': 'user', 'content': 'hi'}]
if TYPE_CHECKING:
    from meridian.types import ChatParams

pytestmark = pytest.mark.anyio


@pytest.mark.fake_only
async def test_create_returns_the_completion_and_three_token_counts(mode: str, engine: Any) -> None:
    engine.next(
        Scripted(
            json=completion(
                'hello',
                prompt_tokens=4,
                completion_tokens=1,
                total_tokens=5,
                prompt_tokens_details={'cached_tokens': 0},
                cost=12,
            )
        )
    )
    result = await resolve(make_client(mode, engine).chat.create({'messages': MSG}))
    assert result.error is None
    assert result.data['choices'][0]['message']['content'] == 'hello'
    assert result.data['usage'] == {'prompt_tokens': 4, 'completion_tokens': 1, 'total_tokens': 5}


@pytest.mark.fake_only
async def test_create_sends_bearer_key_no_model_and_stream_false(mode: str, engine: Any) -> None:
    params: Any = {'messages': MSG, 'stream': True}
    await resolve(make_client(mode, engine).chat.create(params))
    sent = engine.requests[0]
    assert sent.url == httpx.URL('http://engine.test/v1/chat/completions')
    assert sent.headers['authorization'] == f'Bearer {engine.project_key}'
    body = json.loads(sent.content)
    assert body['stream'] is False
    assert 'model' not in body
    assert 1 <= len(sent.headers['idempotency-key']) <= 255


@pytest.mark.fake_only
async def test_engine_error_comes_back_as_an_error_value(mode: str, engine: Any) -> None:
    engine.next(engine_error('invalid_api_key'))
    result = await resolve(make_client(mode, engine).chat.create({'messages': MSG}))
    assert result.data is None
    assert (result.error.code, result.error.type, result.error.status) == (
        'invalid_api_key',
        'authentication_error',
        401,
    )
    assert result.error.retryable is False
    assert result.error.request_id.startswith('req_fake_')


@pytest.mark.fake_only
@pytest.mark.parametrize(
    ('status', 'text', 'retryable'),
    [(502, '<html>bad gateway</html>', True), (503, '', True), (404, '<html>nope</html>', False)],
)
async def test_a_body_that_is_not_meridians_is_a_network_error_with_the_real_status(
    mode: str, engine: Any, status: int, text: str, retryable: bool
) -> None:
    engine.next(*[Scripted(status=status, text=text, headers={'x-request-id': 'req_proxy'})] * 3)
    result = await resolve(make_client(mode, engine, max_retries=0).chat.create({'messages': MSG}))
    assert (result.error.code, result.error.status, result.error.retryable) == ('network_error', status, retryable)
    assert result.error.request_id == 'req_proxy'


@pytest.mark.fake_only
async def test_models_list_returns_the_engine_list(mode: str, engine: Any) -> None:
    engine.next(Scripted(json={'object': 'list', 'data': [{'id': 'm', 'object': 'model', 'owned_by': 'x'}]}))
    result = await resolve(make_client(mode, engine).models.list())
    assert result.data['data'][0]['id'] == 'm'
    assert engine.requests[0].method == 'GET'
    assert engine.requests[0].url.path == '/v1/models'


@pytest.mark.fake_only
@pytest.mark.parametrize('cls', [Meridian, AsyncMeridian])
@pytest.mark.parametrize(('url', 'key'), [('', 'k'), ('ftp://x', 'k'), ('http://engine.test', '')])
async def test_a_bad_url_or_missing_key_is_a_config_error_with_no_request(
    cls: Any, url: str, key: str, engine: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv('MERIDIAN_URL', raising=False)
    monkeypatch.delenv('MERIDIAN_KEY', raising=False)
    result = await resolve(cls(url, key, transport=engine.transport).chat.create({'messages': MSG}))
    assert (result.error.code, result.error.type, result.error.retryable) == ('invalid_request', 'client_error', False)
    assert engine.requests == []


@pytest.mark.fake_only
async def test_url_and_key_default_from_the_environment(
    mode: str, engine: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('MERIDIAN_URL', engine.public_url)
    monkeypatch.setenv('MERIDIAN_KEY', engine.project_key)
    client: Any = Meridian(transport=engine.transport) if mode == 'sync' else AsyncMeridian(transport=engine.transport)
    result = await resolve(client.chat.create({'messages': MSG}))
    assert result.error is None


@pytest.mark.fake_only
def test_a_result_narrows_on_error_is_none(engine: Any) -> None:
    # pyright strict checks this: `.data` is only reachable once `.error is None` has narrowed the union
    params: ChatParams = {'messages': MSG}
    result = Meridian(engine.public_url, engine.project_key, transport=engine.transport).chat.create(params)
    if result.error is None:
        assert result.data['choices'][0]['message']['content'] == 'hello'
        assert isinstance(result, Ok)
    else:
        raise AssertionError(result.error)


@pytest.mark.fake_only
async def test_a_2xx_body_that_is_not_an_object_is_a_network_error_with_the_real_status(mode: str, engine: Any) -> None:
    engine.next(Scripted(status=201, text='[1, 2]'))
    result = await resolve(make_client(mode, engine).chat.create({'messages': MSG}))
    assert (result.error.code, result.error.status) == ('network_error', 201)


@pytest.mark.fake_only
async def test_a_2xx_body_nested_too_deep_to_parse_is_a_network_error_never_a_raise(mode: str, engine: Any) -> None:
    deep = '[' * 100_000 + ']' * 100_000
    engine.next(Scripted(text=deep), Scripted(text=deep))
    client = make_client(mode, engine)
    created = await resolve(client.chat.create({'messages': MSG}))
    listed = await resolve(client.models.list())
    assert (created.error.code, created.error.status) == ('network_error', 200)
    assert (listed.error.code, listed.error.status) == ('network_error', 200)


@pytest.mark.fake_only
async def test_params_nested_too_deep_to_encode_are_an_error_value_never_a_raise(mode: str, engine: Any) -> None:
    deep: Any = []
    for _ in range(100_000):
        deep = [deep]
    client = make_client(mode, engine)
    created = await resolve(client.chat.create({'messages': [{'role': 'user', 'content': deep}]}))
    streamed = await resolve(client.chat.stream({'messages': [{'role': 'user', 'content': deep}]}))
    for result in (created, streamed):
        assert (result.error.code, result.error.type, result.error.retryable) == (
            'invalid_request',
            'client_error',
            False,
        )
    assert engine.requests == []


@pytest.mark.fake_only
async def test_async_client_works_as_a_context_manager_and_closes() -> None:
    engine = make_engine()
    transport: Any = engine.transport
    async with AsyncMeridian(engine.public_url, engine.project_key, transport=transport) as client:
        assert (await client.models.list()).error is None
    await client.aclose()  # closing twice is harmless


@pytest.mark.fake_only
@pytest.mark.parametrize('backend', ['asyncio'])
async def test_cancelling_during_a_body_read_propagates_cancelled_error(engine: Any, backend: str) -> None:
    engine.next(Scripted(chunks=[b'{'], hang=True))
    transport: Any = engine.transport
    client = AsyncMeridian(engine.public_url, engine.project_key, transport=transport)
    with anyio.move_on_after(0.2) as scope:
        await client.chat.create({'messages': MSG})
    assert scope.cancelled_caught  # the call was cancelled, never turned into an Err
