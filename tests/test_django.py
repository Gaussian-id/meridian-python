"""The Django adapter, driven the way servers drive it, through Django's own handlers and middleware.

`forward_view` goes through Django's WSGI handler: the server iterates the response and then calls its `close()`,
however the iteration ended (PEP 3333). `async_forward_view` goes through Django's ASGI handler: a browser that leaves
is seen by Django's own disconnect listener, which cancels the response while the server's write is pending."""

from __future__ import annotations

import asyncio
import contextlib
import functools
import gc
import io
import json
import time
from collections.abc import AsyncIterator, Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any, cast

import anyio
import anyio.to_thread
import django
import httpx
import pytest
from asgiref.sync import sync_to_async
from django.conf import settings
from django.core.handlers.asgi import ASGIHandler
from django.core.handlers.wsgi import WSGIHandler
from django.http import HttpRequest, HttpResponseBase, HttpResponseRedirect, JsonResponse, StreamingHttpResponse
from django.test import AsyncClient, AsyncRequestFactory, RequestFactory, override_settings
from django.urls import clear_url_caches, path

from gauss_meridian import AsyncMeridian, Meridian
from gauss_meridian._client import async_shared_client, shared_client
from gauss_meridian.django import async_forward_view, async_meridian_client, forward_view, meridian_client
from gauss_meridian.forward import AsyncReplyBody, ReplyBody
from tests import django_settings
from tests.support.adapter_contract import CHAT, CHECKS, MEBIBYTE, MESSAGES, STREAM, Wire, allow
from tests.support.fake_engine import Scripted, chunk, frame
from tests.support.recorder import Recorder


def untyped(value: object) -> Any:
    """Django has no type hints: what these tests drive of it directly is handled as Any."""
    return value


if not settings.configured:
    untyped(settings).configure(**{k: getattr(django_settings, k) for k in dir(django_settings) if k.isupper()})
    django.setup()

pytestmark = [pytest.mark.anyio, pytest.mark.fake_only]

# Over Django's DATA_UPLOAD_MAX_MEMORY_SIZE (2.5 MiB, which `request.body` enforces) and within the core's 8 MiB cap
BIG = json.dumps({'messages': [{'role': 'user', 'content': 'x' * (3 * MEBIBYTE)}]}).encode()


@dataclass
class Served:
    status: int
    headers: dict[str, str]
    body: bytes


def mount(view: Callable[..., Any]) -> None:
    django_settings.urlpatterns[:] = [path('chat', view)]
    clear_url_caches()


class Upload:
    """`wsgi.input`: hands Django the body's chunks only as it reads them, as a socket would."""

    def __init__(self, body: bytes | Iterator[bytes]) -> None:
        self._chunks = iter([body]) if isinstance(body, bytes) else body
        self._pending = b''

    def read(self, size: int = -1) -> bytes:
        if not self._pending:
            self._pending = next(self._chunks, b'')
        if size < 0:
            size = len(self._pending)
        piece, self._pending = self._pending[:size], self._pending[size:]
        return piece

    def readline(self, size: int = -1) -> bytes:  # Django's LimitedStream wants one; the views never read lines
        raise AssertionError('the body was read by lines')


def wsgi(
    view: Callable[..., Any],
    *,
    method: str = 'POST',
    content_type: str | None = 'application/json',
    headers: Mapping[str, str] | None = None,
    body: bytes | Iterator[bytes] = CHAT,
    leave_after: int | None = None,
) -> Served:
    """One request through Django's WSGI handler, served as a WSGI server would. `leave_after=n`: the server stops
    after n body chunks (0: before the first), because the browser left; it calls `close()` in every case."""
    mount(view)
    environ: dict[str, Any] = {
        'REQUEST_METHOD': method,
        'SCRIPT_NAME': '',
        'PATH_INFO': '/chat',
        'QUERY_STRING': '',
        'SERVER_NAME': 'testserver',
        'SERVER_PORT': '80',
        'SERVER_PROTOCOL': 'HTTP/1.1',
        'HTTP_HOST': 'testserver',
        'wsgi.version': (1, 0),
        'wsgi.url_scheme': 'http',
        'wsgi.input': Upload(body),
        'wsgi.errors': io.StringIO(),
        'wsgi.multithread': True,
        'wsgi.multiprocess': False,
        'wsgi.run_once': False,
        # A streamed body's length is announced up front, as a browser does: here, more than any body that is sent
        'CONTENT_LENGTH': str(len(body) if isinstance(body, bytes) else 1 << 30),
    }
    if content_type is not None:
        environ['CONTENT_TYPE'] = content_type
    for name, value in (headers or {}).items():
        environ['HTTP_' + name.upper().replace('-', '_')] = value
    started: list[tuple[str, list[tuple[str, str]]]] = []

    def start_response(status: str, response_headers: list[tuple[str, str]], *_: object) -> Callable[[bytes], None]:
        started.append((status, response_headers))
        return lambda data: None

    response = untyped(WSGIHandler())(environ, start_response)
    received: list[bytes] = []
    try:
        if leave_after != 0:
            for piece in response:
                if piece:
                    received.append(piece)
                if leave_after is not None and len(received) >= leave_after:
                    break
    finally:
        response.close()
    status, raw_headers = started[0]
    return Served(int(status.split()[0]), {k.lower(): v for k, v in raw_headers}, b''.join(received))


async def asgi(
    view: Callable[..., Any],
    *,
    method: str = 'POST',
    content_type: str | None = 'application/json',
    headers: Mapping[str, str] | None = None,
    body: bytes | Iterator[bytes] = CHAT,
    leave_after: int | None = None,
    raise_on_leave: bool = False,
) -> Served:
    """One request through Django's ASGI handler. `leave_after=n`: the browser leaves once it has n body chunks (0: as
    the response starts); the server's next write then never completes (`receive()` reports the disconnect and Django
    cancels the response) or, with `raise_on_leave`, raises OSError (ASGI spec 2.4)."""
    mount(view)
    raw_headers = [(b'host', b'testserver')]
    if content_type is not None:
        raw_headers.append((b'content-type', content_type.encode()))
    raw_headers.extend((k.lower().encode(), v.encode()) for k, v in (headers or {}).items())
    scope: dict[str, Any] = {
        'type': 'http',
        'asgi': {'version': '3.0', 'spec_version': '2.4'},
        'http_version': '1.1',
        'method': method,
        'scheme': 'http',
        'path': '/chat',
        'raw_path': b'/chat',
        'root_path': '',
        'query_string': b'',
        'headers': raw_headers,
        'client': ('127.0.0.1', 1),
        'server': ('testserver', 80),
    }
    chunks = iter([body]) if isinstance(body, bytes) else body
    delivered = False
    gone = anyio.Event()
    served = Served(0, {}, b'')
    received: list[bytes] = []

    async def receive() -> dict[str, Any]:
        nonlocal delivered
        if not delivered:
            piece = next(chunks, None)
            if piece is not None:
                return {'type': 'http.request', 'body': piece, 'more_body': True}
            delivered = True
            return {'type': 'http.request', 'body': b'', 'more_body': False}
        await gone.wait()
        return {'type': 'http.disconnect'}

    async def send(message: dict[str, Any]) -> None:
        if message['type'] == 'http.response.start' and leave_after == 0:
            gone.set()
        if gone.is_set():
            if raise_on_leave:
                raise OSError('the client is gone')
            await anyio.sleep_forever()
        if message['type'] == 'http.response.start':
            served.status = message['status']
            served.headers.update((k.decode().lower(), v.decode()) for k, v in message['headers'])
        elif message.get('body'):
            received.append(message['body'])
            if leave_after is not None and len(received) >= leave_after:
                gone.set()

    try:
        with anyio.fail_after(10):
            await ASGIHandler()(scope, receive, send)
    finally:
        served.body = b''.join(received)
    return served


def sync_client(engine: Any) -> Meridian:
    return Meridian(engine.public_url, engine.project_key, transport=engine.transport, max_retries=0)


def async_client(engine: Any) -> AsyncMeridian:
    return AsyncMeridian(engine.public_url, engine.project_key, transport=engine.transport, max_retries=0)


def held(response: object) -> object:
    """The streamed body Django holds for a response (the core's ReplyBody / AsyncReplyBody, for a forwarded stream)."""
    return getattr(response, '_iterator', None)


class WsgiHarness:
    spools_the_body = False

    def __init__(self, engine: Any) -> None:
        self.engine = engine
        self.client = sync_client(engine)

    def own_response(self, status: int, data: dict[str, Any]) -> object:
        return JsonResponse(data, status=status)

    async def send(
        self,
        *,
        authorize: Callable[[Any], object] = allow,
        method: str = 'POST',
        content_type: str | None = 'application/json',
        body: bytes | Iterator[bytes] = CHAT,
        leave_after: int | None = None,
    ) -> Wire:
        sent: list[HttpResponseBase] = []
        inner = forward_view(authorize=authorize, client=self.client)

        @functools.wraps(inner)
        def view(request: HttpRequest) -> HttpResponseBase:
            response = inner(request)
            sent.append(response)
            return response

        call = functools.partial(
            wsgi, view, method=method, content_type=content_type, body=body, leave_after=leave_after
        )
        served = await anyio.to_thread.run_sync(call)  # a WSGI server's worker thread: no event loop runs there

        async def close_again() -> int:
            bodies = [body for body in map(held, sent) if isinstance(body, ReplyBody)]
            for body in bodies:
                body.close()
            return len(bodies)

        return Wire(served.status, served.headers, served.body, close_again)


class AsgiHarness:
    spools_the_body = True  # Django's ASGI handler reads the whole body into a temporary file before any view runs

    def __init__(self, engine: Any) -> None:
        self.engine = engine
        self.client = async_client(engine)

    def own_response(self, status: int, data: dict[str, Any]) -> object:
        return JsonResponse(data, status=status)

    async def send(
        self,
        *,
        authorize: Callable[[Any], object] = allow,
        method: str = 'POST',
        content_type: str | None = 'application/json',
        body: bytes | Iterator[bytes] = CHAT,
        leave_after: int | None = None,
    ) -> Wire:
        sent: list[HttpResponseBase] = []
        inner = async_forward_view(authorize=authorize, client=self.client)

        @functools.wraps(inner)
        async def view(request: HttpRequest) -> HttpResponseBase:
            response = await inner(request)
            sent.append(response)
            return response

        served = await asgi(view, method=method, content_type=content_type, body=body, leave_after=leave_after)

        async def close_again() -> int:
            bodies = [body for body in map(held, sent) if isinstance(body, AsyncReplyBody)]
            for body in bodies:
                await body.aclose()
            return len(bodies)

        return Wire(served.status, served.headers, served.body, close_again)


@pytest.fixture(params=['wsgi', 'asgi'])
def harness(engine: Any, request: pytest.FixtureRequest) -> WsgiHarness | AsgiHarness:
    return WsgiHarness(engine) if request.param == 'wsgi' else AsgiHarness(engine)


@pytest.mark.parametrize('name', list(CHECKS))
async def test_the_adapter_contract(name: str, harness: WsgiHarness | AsgiHarness) -> None:
    await CHECKS[name](harness)


# --- The body goes through the core's cap, never through request.body ------------------------------------------------


def test_a_body_over_djangos_upload_limit_is_forwarded_by_the_wsgi_view(engine: Any) -> None:
    assert settings.DATA_UPLOAD_MAX_MEMORY_SIZE < len(BIG) < 8 * MEBIBYTE
    served = wsgi(forward_view(authorize=allow, client=sync_client(engine)), body=BIG)
    assert served.status == 200 and engine.bodies()[0]['messages'][0]['content'] == 'x' * (3 * MEBIBYTE)


async def test_a_body_over_djangos_upload_limit_is_forwarded_by_the_asgi_view(engine: Any) -> None:
    served = await asgi(async_forward_view(authorize=allow, client=async_client(engine)), body=BIG)
    assert served.status == 200 and engine.bodies()[0]['messages'][0]['content'] == 'x' * (3 * MEBIBYTE)


# --- The engine's stream is closed on every exit ----------------------------------------------------------------------


async def test_asgi_a_write_that_raises_mid_stream_still_closes_upstream(engine: Any) -> None:
    engine.next(Scripted(chunks=[*(frame(chunk(str(i))) for i in range(20)), frame('[DONE]')]))
    view = async_forward_view(authorize=allow, client=async_client(engine))
    # The shielded close waits on the event loop, where Django's disconnect listener cancels the response: the
    # cancellation, raised once the close is over, then ends the response in place of the write's OSError
    with contextlib.suppress(OSError):
        await asgi(view, body=STREAM, leave_after=1, raise_on_leave=True)
    assert engine.upstream_closed() and engine.parts_read() < 21


async def test_asgi_a_first_write_that_raises_closes_upstream_once_the_response_is_dropped(engine: Any) -> None:
    # Django then calls neither aiter(response) nor response.close(): only dropping the response is left
    engine.next(Scripted(chunks=[frame(chunk('a')), frame('[DONE]')]))
    view = async_forward_view(authorize=allow, client=async_client(engine))
    with pytest.raises(OSError, match='gone'):
        await asgi(view, body=STREAM, leave_after=0, raise_on_leave=True)
    gc.collect()
    for _ in range(10):
        await anyio.sleep(0)
    assert engine.upstream_closed() and engine.parts_read() == 0


async def unstarted(engine: Any) -> HttpResponseBase:
    """A forwarded stream the async view answered, which nothing has iterated."""
    engine.next(Scripted(chunks=[frame(chunk('a')), frame('[DONE]')]))
    request = untyped(AsyncRequestFactory()).post('/chat', data=STREAM, content_type='application/json')
    response = await async_forward_view(authorize=allow, client=async_client(engine))(request)
    assert response.status_code == 200 and not engine.upstream_closed()
    return response


async def test_asgi_close_from_djangos_worker_thread_closes_an_unstarted_stream(engine: Any) -> None:
    response = await unstarted(engine)
    await sync_to_async(response.close)()  # how Django's ASGI handler closes a response
    assert engine.upstream_closed() and engine.parts_read() == 0


async def test_asgi_close_on_the_event_loops_thread_closes_an_unstarted_stream(engine: Any) -> None:
    response = await unstarted(engine)
    response.close()  # how Django's AsyncClient closes a response
    for _ in range(10):
        await anyio.sleep(0)
    assert engine.upstream_closed() and engine.parts_read() == 0


async def test_asgi_djangos_async_client_streams(engine: Any) -> None:
    mount(async_forward_view(authorize=allow, client=async_client(engine)))
    engine.next(Scripted(chunks=[frame(chunk('b')), frame('[DONE]')]))
    response = await untyped(AsyncClient()).post('/chat', data=STREAM, content_type='application/json')
    body = b''.join([part async for part in response.streaming_content])
    assert response.status_code == 200 and body == (frame(chunk('b')) + frame('[DONE]')).encode()
    assert engine.upstream_closed()


class HungClose(httpx.AsyncByteStream):
    """An engine stream that keeps sending frames and whose close never returns (a connection that hangs on shutdown).
    Its close notes the event loop it ran on."""

    def __init__(self, *, hangs: bool = True) -> None:
        self.hangs = hangs
        self.closed_on: list[asyncio.AbstractEventLoop] = []

    async def __aiter__(self) -> AsyncIterator[bytes]:
        while True:
            yield frame(chunk('a')).encode()
            await anyio.sleep(0)

    async def aclose(self) -> None:
        self.closed_on.append(asyncio.get_running_loop())
        if self.hangs:
            await anyio.sleep_forever()


def held_client(engine: Any, stream: HungClose) -> AsyncMeridian:
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, headers={'content-type': 'text/event-stream'}, stream=stream)
    )
    return AsyncMeridian(engine.public_url, engine.project_key, transport=transport, max_retries=0)


async def held_response(engine: Any, stream: HungClose) -> HttpResponseBase:
    request = untyped(AsyncRequestFactory()).post('/chat', data=STREAM, content_type='application/json')
    return await async_forward_view(authorize=allow, client=held_client(engine, stream))(request)


@pytest.mark.parametrize('leave_after', [0, 1])
async def test_asgi_a_close_that_never_returns_is_given_up_within_the_bound(engine: Any, leave_after: int) -> None:
    # Before the first byte (the response's close) and mid-stream (the stream's own end), through Django's handler
    view = async_forward_view(authorize=allow, client=held_client(engine, HungClose()))
    started = time.monotonic()
    await asgi(view, body=STREAM, leave_after=leave_after)
    assert time.monotonic() - started < 8  # the close is bounded at 5 s, the worker thread waits 6 s at most


@pytest.mark.parametrize('started', [False, True])
async def test_asgi_a_close_that_never_returns_is_given_up_from_djangos_worker_thread(
    engine: Any, started: bool
) -> None:
    response = untyped(await held_response(engine, HungClose()))
    parts = aiter(response)
    if started:
        await anext(parts)
    began = time.monotonic()
    with anyio.fail_after(8):
        await sync_to_async(response.close)()  # how Django's ASGI handler closes a response
        await parts.aclose()
    assert time.monotonic() - began < 8


async def test_asgi_close_from_a_foreign_thread_runs_on_the_requests_loop(engine: Any) -> None:
    held = HungClose(hangs=False)
    response = await held_response(engine, held)
    with anyio.fail_after(8):
        await anyio.to_thread.run_sync(response.close)  # a thread asgiref knows nothing about
    assert held.closed_on == [asyncio.get_running_loop()]


@pytest.mark.parametrize('kind', ['wsgi', 'asgi'])
async def test_the_views_read_the_body_in_chunks(engine: Any, kind: str) -> None:
    # A request.read(-1) would take the whole body at once, past the cap, before the core could refuse it
    data = b'x' * (9 * MEBIBYTE)
    factory = untyped(AsyncRequestFactory() if kind == 'asgi' else RequestFactory())
    request = factory.post('/chat', data=data, content_type='application/json')
    sizes: list[int] = []
    got = 0
    original = request.read

    def read(size: int = -1) -> bytes:
        nonlocal got
        sizes.append(size)
        piece: bytes = original(size)
        got += len(piece)
        return piece

    request.read = read
    if kind == 'asgi':
        response = await async_forward_view(authorize=allow, client=async_client(engine))(request)
    else:
        response = forward_view(authorize=allow, client=sync_client(engine))(request)
    assert response.status_code == 413 and engine.requests == []
    assert sizes and all(0 < size <= 65536 for size in sizes)
    assert got <= 8 * MEBIBYTE + 65536


def test_the_async_view_under_wsgi_is_the_generic_503_never_buffered(engine: Any) -> None:
    # A deployment mistake is the app's own: the browser learns nothing about it, as for any configuration error
    engine.next(Scripted(chunks=[frame(chunk('a')), frame('[DONE]')]))
    served = wsgi(async_forward_view(authorize=allow, client=async_client(engine)), body=STREAM)
    assert served.status == 503
    assert json.loads(served.body) == {
        'error': {
            'message': 'The service is temporarily unavailable.',
            'type': 'api_error',
            'code': 'service_unavailable',
            'retryable': False,
        }
    }
    assert engine.requests == []


# --- CSRF -------------------------------------------------------------------------------------------------------------

TOKEN = 'a' * 32  # a well-formed CSRF secret, sent as the cookie and as the header


@pytest.mark.parametrize('exempt', [True, False])
def test_csrf_wsgi(engine: Any, exempt: bool) -> None:
    view = forward_view(authorize=allow, client=sync_client(engine), csrf_exempt=exempt)
    served = wsgi(view)
    assert served.status == (200 if exempt else 403) and len(engine.requests) == (1 if exempt else 0)
    with_token = wsgi(view, headers={'cookie': f'csrftoken={TOKEN}', 'x-csrftoken': TOKEN})
    assert with_token.status == 200


@pytest.mark.parametrize('exempt', [True, False])
async def test_csrf_asgi(engine: Any, exempt: bool) -> None:
    view = async_forward_view(authorize=allow, client=async_client(engine), csrf_exempt=exempt)
    served = await asgi(view)
    assert served.status == (200 if exempt else 403) and len(engine.requests) == (1 if exempt else 0)
    with_token = await asgi(view, headers={'cookie': f'csrftoken={TOKEN}', 'x-csrftoken': TOKEN})
    assert with_token.status == 200


# --- The app's own responses ------------------------------------------------------------------------------------------


def test_any_django_response_from_authorize_is_passed_through(engine: Any) -> None:
    redirect = wsgi(forward_view(authorize=lambda r: HttpResponseRedirect('/login'), client=sync_client(engine)))
    assert redirect.status == 302 and redirect.headers['location'] == '/login'
    own_stream = forward_view(
        authorize=lambda r: StreamingHttpResponse(iter([b'one\n', b'two\n']), status=402), client=sync_client(engine)
    )
    streamed = wsgi(own_stream)
    assert streamed.status == 402 and streamed.body == b'one\ntwo\n'
    assert engine.requests == []


async def test_an_async_authorize_and_a_redirect_through_the_asgi_view(engine: Any) -> None:
    async def authorize(request: HttpRequest) -> object:
        if untyped(request).headers.get('x-user') == 'u1':
            return True
        return HttpResponseRedirect('/login')

    view = async_forward_view(authorize=authorize, client=async_client(engine))
    refused = await asgi(view)
    assert refused.status == 302 and refused.headers['location'] == '/login' and engine.requests == []
    assert (await asgi(view, headers={'x-user': 'u1'})).status == 200


# --- Clients from settings --------------------------------------------------------------------------------------------


def test_the_settings_client_is_the_shared_one(engine: Any) -> None:
    with override_settings(MERIDIAN={'URL': engine.public_url, 'KEY': engine.project_key}):
        client = meridian_client()
        assert client is meridian_client() and client is shared_client(engine.public_url, engine.project_key)


def test_missing_settings_fall_back_to_the_environment(engine: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('MERIDIAN_URL', engine.public_url)
    monkeypatch.setenv('MERIDIAN_KEY', engine.project_key)
    assert meridian_client() is shared_client()
    with override_settings(MERIDIAN={'URL': engine.public_url}):
        assert meridian_client() is shared_client()
    with override_settings(MERIDIAN=None):
        assert meridian_client() is shared_client()


@pytest.mark.parametrize('conf', [{}, 'not a mapping', {'URL': 5, 'KEY': 'k'}, {'URL': 'http://engine.test', 'KEY': 7}])
def test_unusable_settings_are_never_cached(conf: object, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('MERIDIAN_URL', raising=False)
    monkeypatch.delenv('MERIDIAN_KEY', raising=False)
    with override_settings(MERIDIAN=conf):
        assert meridian_client() is not meridian_client()


@pytest.mark.parametrize('conf', ['not a mapping', {'URL': 5}, {'URL': '', 'KEY': ''}])
def test_a_setting_that_cannot_work_is_not_replaced_by_the_environment(
    conf: object, engine: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv('MERIDIAN_URL', engine.public_url)
    monkeypatch.setenv('MERIDIAN_KEY', engine.project_key)
    with override_settings(MERIDIAN=conf):
        assert meridian_client() is not meridian_client()
        assert meridian_client() is not shared_client()


async def test_the_async_settings_client_is_shared_on_the_loop(engine: Any) -> None:
    with override_settings(MERIDIAN={'URL': engine.public_url, 'KEY': engine.project_key}):
        client = async_meridian_client()
        assert client is async_meridian_client()
        assert client is async_shared_client(engine.public_url, engine.project_key)


CLOSING = {'URL': 'http://django-closed.engine.test', 'KEY': 'gmr_' + '8' * 64}


def test_a_closed_settings_client_is_replaced_so_the_view_keeps_answering(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = Recorder(monkeypatch)
    with override_settings(MERIDIAN=CLOSING):
        with meridian_client() as client:
            assert client.chat.create({'messages': cast('Any', MESSAGES)}).error is None
        served = wsgi(forward_view(authorize=allow))
    assert served.status == 200 and len(recorder.senders) == 2


async def test_a_closed_async_settings_client_is_replaced_so_the_view_keeps_answering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = Recorder(monkeypatch)
    with override_settings(MERIDIAN=CLOSING):
        async with async_meridian_client() as client:
            assert (await client.chat.create({'messages': cast('Any', MESSAGES)})).error is None
        served = await asgi(async_forward_view(authorize=allow))
    assert served.status == 200 and len(recorder.senders) == 2


@pytest.mark.parametrize('conf', [{}, 'not a mapping', {'URL': 5}])
def test_the_views_without_usable_settings_are_a_generic_503(conf: object, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('MERIDIAN_URL', raising=False)
    monkeypatch.delenv('MERIDIAN_KEY', raising=False)
    with override_settings(MERIDIAN=conf):
        served = wsgi(forward_view(authorize=allow))
        assert served.status == 503 and json.loads(served.body)['error']['code'] == 'service_unavailable'


@pytest.mark.parametrize('conf', [{}, 'not a mapping'])
async def test_the_async_view_without_usable_settings_is_a_generic_503(
    conf: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv('MERIDIAN_URL', raising=False)
    monkeypatch.delenv('MERIDIAN_KEY', raising=False)
    with override_settings(MERIDIAN=conf):
        served = await asgi(async_forward_view(authorize=allow))
        assert served.status == 503 and json.loads(served.body)['error']['code'] == 'service_unavailable'


# --- Set-up mistakes raise; nothing else does -------------------------------------------------------------------------


def test_a_missing_authorize_raises() -> None:
    with pytest.raises(TypeError):
        forward_view(authorize=cast('Any', None))
    with pytest.raises(TypeError):
        async_forward_view(authorize=cast('Any', None))


def test_a_coroutine_function_client_is_refused_by_the_async_view() -> None:
    async def make() -> AsyncMeridian:
        return AsyncMeridian()

    with pytest.raises(TypeError, match='long-lived'):
        async_forward_view(authorize=allow, client=cast('Any', make))


def test_an_engine_failure_is_a_response_not_an_exception(engine: Any) -> None:
    engine.next(Scripted(network_error=True))
    served = wsgi(forward_view(authorize=allow, client=sync_client(engine)))
    assert served.status == 503 and json.loads(served.body)['error']['code'] == 'service_unavailable'


async def test_an_engine_failure_is_a_response_through_the_asgi_view(engine: Any) -> None:
    engine.next(Scripted(network_error=True))
    served = await asgi(async_forward_view(authorize=allow, client=async_client(engine)))
    assert served.status == 503 and json.loads(served.body)['error']['code'] == 'service_unavailable'


def test_the_browsers_idempotency_key_reaches_the_engine(engine: Any) -> None:
    wsgi(forward_view(authorize=allow, client=sync_client(engine)), headers={'idempotency-key': 'abc-123'})
    assert engine.requests[0].headers['idempotency-key'] == 'abc-123'


def test_messages_reach_the_engine(engine: Any) -> None:
    wsgi(forward_view(authorize=allow, client=sync_client(engine)))
    assert engine.bodies()[0]['messages'] == MESSAGES
