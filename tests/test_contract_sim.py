"""The contract matrix against the real local engine in simulator mode (scripts/contract.py only: these tests are
skipped everywhere else). The simulator's reply is fixed: ten deltas `sim token 1` ... `sim token 10`, ten completion
tokens. The guard has proven the simulator before any test is built; the canary below is the tripwire after it. No test
here prints a key."""

import json
import time
import uuid
from typing import Any

import anyio
import httpx
import pytest
from fastapi import FastAPI

from meridian import AsyncMeridian, Meridian
from meridian.admin import AsyncMeridianAdmin, MeridianAdmin
from meridian.fastapi import forward_route
from meridian.forward import AllowRule, AsyncForwarder, AsyncReplyBody, Forwarder
from tests.support.engine import ContractEngine
from tests.support.modes import make_client, resolve

pytestmark = [pytest.mark.anyio, pytest.mark.contract]

MODEL = 'gemini-3.8-flash'
MSG: Any = [{'role': 'user', 'content': 'hello'}]
PIECES = [f'sim token {n}' for n in range(1, 11)]
FIXED = ''.join(PIECES)
# The engine keeps a revoked key's hash and answers key_revoked; both key codes must stay hidden from a browser
REVOKED = ('key_revoked',)
KEY_CODES = ('invalid_api_key', 'key_revoked')


@pytest.fixture(scope='module', autouse=True)
def canary() -> None:
    """The tripwire: if the reply is not the simulator's, nothing runs (the session stops)."""
    engine = ContractEngine('canary')
    result = Meridian(engine.public_url, engine.project_key, max_retries=0).chat.create(
        {'model': MODEL, 'messages': MSG}
    )
    data = result.data
    if (
        result.error is not None
        or data is None
        or data['choices'][0]['message']['content'] != FIXED
        or data['usage']['completion_tokens'] != 10
    ):
        pytest.exit('The canary failed: the engine did not answer with the simulator reply. Stop and tear down.', 1)


def allow_all(_: object) -> bool:
    return True


async def collect(stream: Any) -> list[str]:
    deltas = [c.delta async for c in stream] if hasattr(stream, '__aiter__') else [c.delta for c in stream]
    return [delta for delta in deltas if delta != '']


def admin(mode: str, engine: Any) -> Any:
    cls = MeridianAdmin if mode == 'sync' else AsyncMeridianAdmin
    return cls(engine.admin_url, engine.integrator_key, max_retries=0)


def client_with(mode: str, engine: Any, key: str) -> Any:
    return make_client(mode, engine, key=str(key), max_retries=0)


class OwnKey:
    """A fresh organisation and project with a key of its own: the seeded keys are never revoked."""

    def __init__(self, mode: str, engine: Any) -> None:
        self.mode, self.engine, self.ops = mode, engine, admin(mode, engine)
        self.org = self.project = self.key_id = self.secret = ''

    async def issue(self) -> 'OwnKey':
        org = await resolve(
            self.ops.orgs.create({'name': 'Contract', 'external_reference': f'contract-{uuid.uuid4()}'})
        )
        assert org.error is None
        self.org = org.data['id']
        project = await resolve(self.ops.projects.create(self.org, {'name': 'Contract project'}))
        assert project.error is None
        self.project = project.data['id']
        issued = await resolve(self.ops.keys.create(self.org, self.project, {}))
        assert issued.error is None
        self.key_id, self.secret = issued.data['key_id'], str(issued.data['api_key'])
        return self

    async def revoke(self) -> None:
        revoked = await resolve(self.ops.keys.revoke(self.org, self.project, self.key_id))
        assert revoked.error is None


async def own_key(mode: str, engine: Any) -> OwnKey:
    return await OwnKey(mode, engine).issue()


async def forwarded(mode: str, client: Any, body: dict[str, Any]) -> tuple[int, dict[str, str], bytes]:
    """The core forwarder, as an app mounts it: one request in, the browser's whole view out."""
    raw = json.dumps(body).encode()
    options: dict[str, Any] = {'client': client, 'authorize': allow_all, 'allow': AllowRule(model=True)}
    if mode == 'sync':
        reply = Forwarder[Any](**options).handle(
            None, method='POST', content_type='application/json', idempotency_key=None, read_body=lambda: raw
        )
        out = reply.body if isinstance(reply.body, bytes) else b''.join(reply.body)
        reply.close()
    else:
        reply = await AsyncForwarder[Any](**options).ahandle(
            None, method='POST', content_type='application/json', idempotency_key=None, read_body=lambda: raw
        )
        if isinstance(reply.body, bytes):
            out = reply.body
        else:
            assert isinstance(reply.body, AsyncReplyBody)
            out = b''.join([piece async for piece in reply.body])
        await reply.aclose()
    return reply.status, dict(reply.headers), out


# --- the matrix ------------------------------------------------------------------------------------------------------


async def test_canary_the_fixed_reply_ten_completion_tokens_and_no_cost(mode: str, engine: Any) -> None:
    result = await resolve(make_client(mode, engine, max_retries=0).chat.create({'model': MODEL, 'messages': MSG}))
    assert result.error is None
    assert result.data['choices'][0]['message']['content'] == FIXED
    assert result.data['usage']['completion_tokens'] == 10
    assert 'cost' not in json.dumps(result.data).lower()


async def test_chat_create_has_usage_and_an_id(mode: str, engine: Any) -> None:
    result = await resolve(make_client(mode, engine, max_retries=0).chat.create({'model': MODEL, 'messages': MSG}))
    assert result.error is None
    usage = result.data['usage']
    assert usage['total_tokens'] == usage['prompt_tokens'] + 10
    assert isinstance(result.data['id'], str) and result.data['id']


async def test_an_invalid_key_is_invalid_api_key_with_the_engine_code_and_request_id(mode: str, engine: Any) -> None:
    client = client_with(mode, engine, 'gmr_' + 'f' * 64)
    result = await resolve(client.chat.create({'model': MODEL, 'messages': MSG}))
    assert result.data is None
    assert (result.error.code, result.error.type, result.error.status) == (
        'invalid_api_key',
        'authentication_error',
        401,
    )
    assert isinstance(result.error.request_id, str) and result.error.request_id


async def test_stream_gives_exactly_ten_text_pieces_then_usage(mode: str, engine: Any) -> None:
    result = await resolve(make_client(mode, engine, max_retries=0).chat.stream({'model': MODEL, 'messages': MSG}))
    assert result.error is None
    stream = result.data
    assert await collect(stream) == PIECES
    assert stream.error is None
    assert stream.usage['completion_tokens'] == 10


async def test_admin_provisions_an_organisation_a_project_and_a_key_lists_and_revokes_it(
    mode: str, engine: Any
) -> None:
    own = await own_key(mode, engine)
    assert own.secret.startswith('gmr_')
    listed = await resolve(own.ops.keys.list(own.org, own.project))
    assert [k['active'] for k in listed.data if k['id'] == own.key_id] == [True]
    assert own.secret not in json.dumps(listed.data)
    await own.revoke()
    after = await resolve(own.ops.keys.list(own.org, own.project))
    assert [k['active'] for k in after.data if k['id'] == own.key_id] == [False]


async def test_a_revoked_key_used_directly_is_refused(mode: str, engine: Any) -> None:
    own = await own_key(mode, engine)
    client = client_with(mode, engine, own.secret)
    assert (await resolve(client.models.list())).error is None
    await own.revoke()
    # [1] The engine may keep accepting a revoked key for up to 5 s (its key cache), so poll for 8 s
    result = await resolve(client.models.list())
    deadline = time.monotonic() + 8
    while result.error is None and time.monotonic() < deadline:
        await anyio.sleep(0.25)
        result = await resolve(client.models.list())
    assert result.data is None
    assert result.error is not None
    assert result.error.status == 401
    assert result.error.code in REVOKED


async def test_forwarding_buffered_and_streamed_through_the_core_forwarder(mode: str, engine: Any) -> None:
    client = make_client(mode, engine, max_retries=0)
    status, _, out = await forwarded(mode, client, {'model': MODEL, 'messages': MSG})
    assert status == 200
    assert json.loads(out)['choices'][0]['message']['content'] == FIXED
    status, headers, out = await forwarded(mode, client, {'model': MODEL, 'messages': MSG, 'stream': True})
    assert status == 200
    assert headers['content-type'].startswith('text/event-stream')
    assert out.decode().rstrip().endswith('data: [DONE]')
    for piece in PIECES:
        assert piece in out.decode()


async def test_forwarding_through_the_fastapi_adapter_against_the_real_engine(engine: Any) -> None:
    client = AsyncMeridian(engine.public_url, engine.project_key, max_retries=0)
    app = FastAPI()
    app.post('/chat')(forward_route(authorize=allow_all, client=client, allow=AllowRule(model=True)))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://app.test') as browser:
        buffered = await browser.post('/chat', json={'model': MODEL, 'messages': MSG})
        streamed = await browser.post('/chat', json={'model': MODEL, 'messages': MSG, 'stream': True})
    assert buffered.status_code == 200
    assert buffered.json()['choices'][0]['message']['content'] == FIXED
    assert streamed.status_code == 200
    assert streamed.headers['content-type'].startswith('text/event-stream')
    assert streamed.text.rstrip().endswith('data: [DONE]')
    await client.aclose()


async def test_forwarding_with_the_revoked_key_is_the_generic_503_and_the_server_still_sees_the_code(
    mode: str, engine: Any
) -> None:
    own = await own_key(mode, engine)
    client = client_with(mode, engine, own.secret)
    await own.revoke()
    status, _, out = await forwarded(mode, client, {'model': MODEL, 'messages': MSG})
    error = json.loads(out)['error']
    assert status == 503
    assert (error['message'], error['type'], error['code'], error['retryable']) == (
        'The service is temporarily unavailable.',
        'api_error',
        'service_unavailable',
        False,
    )
    assert not any(code in out.decode() for code in KEY_CODES)
    direct = await resolve(client.chat.create({'model': MODEL, 'messages': MSG}))
    assert direct.error.code in REVOKED


async def test_sim_fail_the_fallback_still_answers_with_a_completion(mode: str, engine: Any) -> None:
    # Only here because the canary proved the simulator: the model `sim-fail` always fails at its direct provider, so
    # the engine answers through its fallback. Only the reply shape is asserted, nothing about timing.
    result = await resolve(make_client(mode, engine, max_retries=0).chat.create({'model': 'sim-fail', 'messages': MSG}))
    assert result.error is None
    assert isinstance(result.data['choices'][0]['message']['content'], str)
    assert result.data['usage']['completion_tokens'] == 10
