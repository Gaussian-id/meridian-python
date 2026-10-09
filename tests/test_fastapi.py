"""The FastAPI adapter. It is driven through an ASGI call: Starlette's TestClient cannot be type-checked under strict
pyright (it is typed against `httpx2`, which is not a dependency) and cannot make a browser leave mid-stream."""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator, Callable, Iterator, Mapping
from typing import Any, cast

import anyio
import httpx
import pytest
from fastapi import BackgroundTasks, Depends, FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse, Response, StreamingResponse
from starlette.requests import ClientDisconnect
from starlette.types import Message, Scope

from meridian import AsyncMeridian
from meridian._client import async_shared_client
from meridian.fastapi import forward_route, meridian_client
from meridian.forward import AsyncReplyBody
from tests.support.adapter_contract import CHAT, CHECKS, STREAM, Wire, allow
from tests.support.fake_engine import Scripted, chunk, engine_error, frame

pytestmark = [pytest.mark.anyio, pytest.mark.fake_only]
MSG = [{'role': 'user', 'content': 'hi'}]


async def pieces(body: bytes | Iterator[bytes]) -> AsyncIterator[bytes]:
    if isinstance(body, bytes):
        yield body
        return
    for piece in body:
        yield piece


async def drive(
    app: FastAPI,
    *,
    method: str = 'POST',
    path: str = '/chat',
    content_type: str | None = 'application/json',
    headers: Mapping[str, str] | None = None,
    body: bytes | Iterator[bytes] = CHAT,
    leave_after: int | None = None,
    spec_version: str = '2.4',
    on_start: Callable[[], None] = lambda: None,
) -> tuple[int, dict[str, str], bytes]:
    """Call the app as an ASGI server would. `leave_after=n`: the client goes away after n body chunks (spec 2.4: the
    next send raises OSError; older: `receive()` answers http.disconnect). `on_start` runs when the response starts."""
    raw_headers = [(b'host', b'test')]
    if content_type is not None:
        raw_headers.append((b'content-type', content_type.encode()))
    raw_headers.extend((k.lower().encode(), v.encode()) for k, v in (headers or {}).items())
    scope: Scope = {
        'type': 'http',
        'asgi': {'version': '3.0', 'spec_version': spec_version},
        'http_version': '1.1',
        'method': method,
        'scheme': 'http',
        'path': path,
        'raw_path': path.encode(),
        'query_string': b'',
        'headers': raw_headers,
        'client': ('127.0.0.1', 1),
        'server': ('test', 80),
    }
    source = pieces(body)
    gone = anyio.Event()
    if leave_after == 0:
        gone.set()
    delivered = False
    status = 0
    response_headers: dict[str, str] = {}
    received: list[bytes] = []

    async def receive() -> Message:
        nonlocal delivered
        if not delivered:
            try:
                piece = await anext(source)
            except StopAsyncIteration:
                delivered = True
                return {'type': 'http.request', 'body': b'', 'more_body': False}
            return {'type': 'http.request', 'body': piece, 'more_body': True}
        await gone.wait()
        return {'type': 'http.disconnect'}

    async def send(message: Message) -> None:
        nonlocal status
        await anyio.sleep(0)  # a real server's send waits on the socket: that is where a cancellation lands
        if leave_after is not None and len(received) >= leave_after:
            gone.set()
            if spec_version == '2.4':
                raise OSError('the client is gone')
        if message['type'] == 'http.response.start':
            status = message['status']
            response_headers.update((k.decode(), v.decode()) for k, v in message['headers'])
            on_start()
        elif message.get('body'):
            received.append(message['body'])
            if leave_after is not None and len(received) >= leave_after:
                gone.set()

    try:
        await app(scope, receive, send)
    except (OSError, ClientDisconnect):  # how Starlette reports a client that left (spec 2.4); anything else is a bug
        if leave_after is None:
            raise
    return status, response_headers, b''.join(received)


class FastApiHarness:
    spools_the_body = False

    def __init__(self, engine: Any, spec_version: str) -> None:
        self.engine = engine
        self.spec_version = spec_version
        self.client = AsyncMeridian(engine.public_url, engine.project_key, transport=engine.transport, max_retries=0)

    def own_response(self, status: int, data: dict[str, Any]) -> object:
        return JSONResponse(data, status_code=status)

    async def send(
        self,
        *,
        authorize: Callable[[Any], object] = allow,
        method: str = 'POST',
        content_type: str | None = 'application/json',
        body: bytes | Iterator[bytes] = CHAT,
        leave_after: int | None = None,
    ) -> Wire:
        sent: list[Response] = []
        route = forward_route(authorize=authorize, client=self.client)

        async def endpoint(request: Request) -> Response:
            response = await route(request)
            sent.append(response)
            return response

        app = FastAPI()
        app.add_api_route('/chat', endpoint, methods=['GET', 'POST', 'PUT'])
        status, headers, received = await drive(
            app,
            method=method,
            content_type=content_type,
            body=body,
            leave_after=leave_after,
            spec_version=self.spec_version,
        )

        async def close_again() -> int:
            closed = 0
            for response in sent:
                if isinstance(response, StreamingResponse) and isinstance(response.body_iterator, AsyncReplyBody):
                    await response.body_iterator.aclose()
                    closed += 1
            return closed

        return Wire(status, headers, received, close_again)


@pytest.fixture(params=['2.0', '2.4'])
def harness(engine: Any, request: pytest.FixtureRequest) -> FastApiHarness:
    return FastApiHarness(engine, str(request.param))


@pytest.mark.parametrize('name', list(CHECKS))
async def test_the_adapter_contract(name: str, harness: FastApiHarness) -> None:
    await CHECKS[name](harness)


class SlowClose(httpx.AsyncByteStream):
    """An engine stream that sends one frame and then waits. Closing it takes a checkpoint, as a connection does."""

    closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield frame(chunk('a')).encode()
        await anyio.sleep_forever()

    async def aclose(self) -> None:
        await anyio.sleep(0)
        self.closed = True


@pytest.mark.parametrize('spec_version', ['2.0', '2.4'])
async def test_a_cancelled_request_still_closes_upstream(engine: Any, spec_version: str) -> None:
    held = SlowClose()
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, headers={'content-type': 'text/event-stream'}, stream=held)
    )
    client = AsyncMeridian(engine.public_url, engine.project_key, transport=transport)
    app = FastAPI()
    app.post('/chat')(forward_route(authorize=allow, client=client))
    with anyio.CancelScope() as request_scope:
        await drive(app, body=STREAM, spec_version=spec_version, on_start=request_scope.cancel)
    assert request_scope.cancelled_caught and held.closed


class NeverCloses(httpx.AsyncByteStream):
    """An engine stream whose close never returns (a connection that hangs on shutdown)."""

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield frame(chunk('a')).encode()
        await anyio.sleep_forever()

    async def aclose(self) -> None:
        await anyio.sleep_forever()


async def test_a_close_that_never_returns_is_given_up_within_the_bound(engine: Any) -> None:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, headers={'content-type': 'text/event-stream'}, stream=NeverCloses())
    )
    client = AsyncMeridian(engine.public_url, engine.project_key, transport=transport)
    app = FastAPI()
    app.post('/chat')(forward_route(authorize=allow, client=client))
    started = time.monotonic()
    await drive(app, body=STREAM, leave_after=0)
    assert time.monotonic() - started < 8  # the close is shielded but bounded at 5 s


def test_a_coroutine_function_client_is_refused_at_construction() -> None:
    async def make() -> AsyncMeridian:
        return AsyncMeridian()

    for factory in (meridian_client, make):
        with pytest.raises(TypeError, match='long-lived'):
            forward_route(authorize=allow, client=cast('Any', factory))


async def test_a_response_fastapi_never_sends_still_closes_upstream(engine: Any) -> None:
    async def teardown_fails() -> AsyncIterator[None]:
        yield
        raise RuntimeError('teardown')

    engine.next(Scripted(chunks=[frame(chunk('a')), frame('[DONE]')]))
    client = AsyncMeridian(engine.public_url, engine.project_key, transport=engine.transport, max_retries=0)
    app = FastAPI()
    # A function-scoped yield dependency exits before FastAPI sends the response: its raise means it is never sent
    app.post('/chat', dependencies=[Depends(teardown_fails, scope='function')])(
        forward_route(authorize=allow, client=client)
    )
    with pytest.raises(RuntimeError, match='teardown'):
        await drive(app, body=STREAM)
    assert engine.upstream_closed() and engine.parts_read() == 0


async def test_background_tasks_run_after_upstream_is_closed(engine: Any) -> None:
    engine.next(Scripted(chunks=[*(frame(chunk(str(i))) for i in range(20)), frame('[DONE]')]))
    client = AsyncMeridian(engine.public_url, engine.project_key, transport=engine.transport, max_retries=0)
    route = forward_route(authorize=allow, client=client)
    seen: list[bool] = []

    def note() -> None:
        seen.append(engine.upstream_closed())

    async def chat(request: Request, tasks: BackgroundTasks) -> Response:
        tasks.add_task(note)
        return await route(request)

    app = FastAPI()
    app.post('/chat')(chat)
    # Spec 2.0: Starlette runs the background task even after the browser left, while the relay is still suspended
    await drive(app, body=STREAM, leave_after=1, spec_version='2.0')
    assert seen == [True]


def app_for(engine: Any, authorize: Callable[[Request], object] = lambda r: True) -> FastAPI:
    app = FastAPI()
    client = AsyncMeridian(engine.public_url, engine.project_key, transport=engine.transport, max_retries=0)
    app.post('/api/meridian/chat')(forward_route(authorize=authorize, client=client))
    return app


async def post(app: FastAPI, payload: dict[str, Any], **extra: Any) -> tuple[int, dict[str, str], bytes]:
    return await drive(app, path='/api/meridian/chat', body=json.dumps(payload).encode(), **extra)


def test_a_missing_authorize_raises() -> None:
    with pytest.raises(TypeError):
        forward_route(authorize=cast('Any', None))


async def test_the_apps_own_responses_are_returned(engine: Any) -> None:
    redirect = app_for(engine, authorize=lambda r: RedirectResponse('/login', status_code=303))
    status, headers, _ = await post(redirect, {'messages': MSG})
    assert status == 303 and headers['location'] == '/login'

    async def lines() -> AsyncIterator[bytes]:
        yield b'one\n'
        yield b'two\n'

    streaming = app_for(engine, authorize=lambda r: StreamingResponse(lines(), status_code=402))
    status, _, body = await post(streaming, {'messages': MSG})
    assert status == 402 and body == b'one\ntwo\n'
    assert engine.requests == []


async def test_an_async_authorize_reading_a_header(engine: Any) -> None:
    async def authorize(request: Request) -> object:
        return request.headers.get('x-user') == 'u1'

    app = app_for(engine, authorize=authorize)
    assert (await post(app, {'messages': MSG}))[0] == 403
    assert (await post(app, {'messages': MSG}, headers={'x-user': 'u1'}))[0] == 200


async def test_the_browsers_idempotency_key_reaches_the_engine(engine: Any) -> None:
    await post(app_for(engine), {'messages': MSG}, headers={'idempotency-key': 'abc-123'})
    assert engine.requests[0].headers['idempotency-key'] == 'abc-123'


async def test_an_engine_failure_is_a_response_not_an_exception(engine: Any) -> None:
    engine.next(Scripted(network_error=True))
    status, _, body = await post(app_for(engine), {'messages': MSG})
    assert status == 503 and json.loads(body)['error']['code'] == 'service_unavailable'


async def test_an_engine_error_that_is_not_hidden_keeps_its_status(engine: Any) -> None:
    engine.next(engine_error('model_not_allowed'))
    status, _, body = await post(app_for(engine), {'messages': MSG})
    assert status == 403 and json.loads(body)['error']['code'] == 'model_not_allowed'


async def test_the_default_client_without_settings_is_a_generic_503(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('MERIDIAN_URL', raising=False)
    monkeypatch.delenv('MERIDIAN_KEY', raising=False)
    app = FastAPI()
    app.post('/chat')(forward_route(authorize=lambda r: True))
    status, _, body = await drive(app)
    assert status == 503 and json.loads(body)['error']['code'] == 'service_unavailable'


async def test_the_default_client_is_the_shared_one(monkeypatch: pytest.MonkeyPatch, engine: Any) -> None:
    monkeypatch.setenv('MERIDIAN_URL', engine.public_url)
    monkeypatch.setenv('MERIDIAN_KEY', engine.project_key)
    app = FastAPI()
    seen: list[AsyncMeridian] = []

    @app.get('/who')
    async def who(client: AsyncMeridian = Depends(meridian_client)) -> dict[str, bool]:  # noqa: B008
        seen.append(client)
        return {'shared': client is async_shared_client()}

    for _ in range(2):
        status, _, body = await drive(app, method='GET', path='/who', content_type=None, body=b'')
        assert status == 200 and json.loads(body) == {'shared': True}
    assert seen[0] is seen[1]


async def test_get_is_405_with_allow(engine: Any) -> None:
    app = FastAPI()
    app.api_route('/c', methods=['GET', 'POST'])(forward_route(authorize=lambda r: True))
    status, headers, body = await drive(app, method='GET', path='/c', content_type=None, body=b'')
    assert status == 405 and headers['allow'] == 'POST'
    assert json.loads(body)['error']['code'] == 'method_not_allowed'
