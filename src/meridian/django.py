"""Django glue: clients from settings and ready forwarding views (WSGI and ASGI). Install with `meridian[django]`.

`settings.MERIDIAN = {'URL': ..., 'KEY': ...}` configures the clients; a missing URL or KEY falls back to the
`MERIDIAN_URL` / `MERIDIAN_KEY` environment variables. Each client is one long-lived shared client per settings (per
event loop for the async one), never one per request.

`forward_view` is for WSGI and `async_forward_view` for ASGI. Under ASGI, Django would buffer the whole stream of the
sync view before sending any of it; under WSGI, it would run the async view in a throwaway event loop and buffer the
whole stream too, so the async view refuses to run there.

Both views are CSRF-exempt by default: the route takes JSON only and answers anything else with 415 before
`authorize` runs, so a cross-site page cannot send it without a CORS preflight, and `authorize` decides who may use
it. Pass `csrf_exempt=False` to keep Django's CSRF check.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Callable, Iterator, Mapping
from typing import Any, Protocol, TypeGuard, TypeVar

from asgiref.sync import async_to_sync
from django.conf import settings
from django.core.handlers.wsgi import WSGIRequest
from django.http import HttpRequest, HttpResponse, HttpResponseBase, JsonResponse, StreamingHttpResponse

from ._client import AsyncMeridian, Meridian, async_shared_client, shared_client
from .forward import AllowRule, AsyncForwarder, AsyncReplyBody, Forwarder, Reply, ReplyBody

__all__ = ['async_forward_view', 'async_meridian_client', 'forward_view', 'meridian_client']

# How much of the browser's body is read at a time (the core stops reading past its 8 MiB cap)
_CHUNK = 65536


# --- Clients from settings ------------------------------------------------------------------------------------------


def _is_mapping(value: object) -> TypeGuard[Mapping[str, object]]:
    return isinstance(value, Mapping)


def _setting(value: object) -> str | None:
    # None falls back to the environment; any other value that is no string can never work, and '' is refused as such
    return value if value is None or isinstance(value, str) else ''


def _settings() -> tuple[str | None, str | None]:
    """`settings.MERIDIAN`'s URL and KEY (None: use the environment). Settings that are no mapping, or cannot be read
    at all, give a configuration that is refused (a generic 503 at the views), never a raise."""
    try:
        conf: object = getattr(settings, 'MERIDIAN', None)
        if conf is None:
            return None, None
        if not _is_mapping(conf):
            return '', ''
        return _setting(conf.get('URL')), _setting(conf.get('KEY'))
    except Exception:
        return '', ''


def meridian_client() -> Meridian:
    """The shared client for `settings.MERIDIAN` (or the environment): one long-lived client per settings, never one
    per request. Settings that cannot work give a client that refuses every call, and are never kept."""
    url, key = _settings()
    return shared_client(url, key)


def async_meridian_client() -> AsyncMeridian:
    """`meridian_client` for async code: one shared client per settings and per running event loop."""
    url, key = _settings()
    return async_shared_client(url, key)


# --- The request ----------------------------------------------------------------------------------------------------


class _Browser(Protocol):
    """What the views read from a Django request (Django has no type hints of its own)."""

    @property
    def method(self) -> str | None: ...

    @property
    def META(self) -> Mapping[str, object]: ...

    def read(self, size: int, /) -> bytes: ...


def _text(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _asked(request: _Browser) -> tuple[str, str | None, str | None]:
    """The method, the content-type and the idempotency key, from META as Django's `request.headers` reads them."""
    return (
        request.method or '',
        _text(request.META.get('CONTENT_TYPE')),
        _text(request.META.get('HTTP_IDEMPOTENCY_KEY')),
    )


def _chunks(request: _Browser) -> Iterator[bytes]:
    # Never `request.body`: it reads the whole body, and refuses one over DATA_UPLOAD_MAX_MEMORY_SIZE (2.5 MiB)
    while piece := request.read(_CHUNK):
        yield piece


async def _achunks(request: _Browser) -> AsyncIterator[bytes]:
    # Under ASGI Django has already spooled the body to a file: reading it is a local file read, as `request.body` is
    while piece := request.read(_CHUNK):
        yield piece


# --- Responses --------------------------------------------------------------------------------------------------------


_R = TypeVar('_R', bound=HttpResponseBase)


def _headed(response: _R, reply: Reply[Any]) -> _R:
    # The reply's own headers, as given: the core never sets content-length or content-encoding
    for name, value in reply.headers.items():
        response[name] = value
    return response


def _response(reply: Reply[ReplyBody]) -> HttpResponseBase:
    # [1] The app's own response from authorize, returned as it is
    if isinstance(reply.passthrough, HttpResponseBase):
        return reply.passthrough

    # [2] Bytes, or a stream: Django registers the body's close() with the response, and the WSGI server calls the
    #     response's close() however the iteration ended (finished, abandoned by the browser, or never started)
    if isinstance(reply.body, bytes):
        return _headed(HttpResponse(reply.body, status=reply.status), reply)
    return _headed(StreamingHttpResponse(reply.body, status=reply.status), reply)


# Closes scheduled on an event loop, held until they finish (a task nothing refers to may be collected mid-way)
_scheduled: set[asyncio.Task[None]] = set()


def _close_soon(body: AsyncReplyBody) -> None:
    """Close the body in a task of the running loop: for code on the loop's own thread, which cannot await it."""
    task = asyncio.get_running_loop().create_task(body.aclose())
    _scheduled.add(task)
    task.add_done_callback(_scheduled.discard)


class _AsyncForwardedStream(StreamingHttpResponse):
    """A streamed reply for ASGI that closes the engine's stream on every exit.

    Django's ASGI handler sends a streaming response through `aclosing(aiter(response))` and then calls the response's
    `close()`, but each of those closes only Django's own wrappers, never the body: the body's `aclose()` is neither
    called nor registered with the response. So `__aiter__` closes the body however sending ends (finished, the
    browser gone, a failed write), and `close()` closes it when sending never started (the browser left before the
    first byte), on the event loop the body belongs to. When the server's very first write raises, Django runs
    neither: the body is then closed on that loop once the response is dropped."""

    _body: AsyncReplyBody
    _loop: asyncio.AbstractEventLoop
    _closed = False

    @classmethod
    def of(cls, reply: Reply[AsyncReplyBody], body: AsyncReplyBody) -> _AsyncForwardedStream:
        response = cls(body, status=reply.status)
        response._body, response._loop = body, asyncio.get_running_loop()
        return _headed(response, reply)

    async def _aclose_body(self) -> None:
        self._closed = True
        await self._body.aclose()

    async def __aiter__(self) -> AsyncGenerator[bytes, None]:
        try:
            # The streaming content as Django would send it (a middleware may have wrapped it, async or not)
            content = self.streaming_content
            if isinstance(content, AsyncIterator):
                async for part in content:
                    yield part
            else:
                for part in content:
                    yield part
        finally:
            await self._aclose_body()

    def close(self) -> None:
        # [1] Django calls this from a worker thread (sync_to_async): the close runs on the request's own event loop.
        #     Called on the loop's own thread (Django's AsyncClient does), it is scheduled there instead
        self._closed = True
        with contextlib.suppress(Exception):
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                async_to_sync(self._body.aclose)()
            else:
                _close_soon(self._body)

        # [2] Then Django's own close (its resources, and the request_finished signal)
        super().close()

    def __del__(self) -> None:
        # Dropped with neither close run (the server's first write raised): closed on the request's loop, if it runs
        if not self._closed:
            with contextlib.suppress(Exception):
                self._loop.call_soon_threadsafe(_close_soon, self._body)


def _async_response(reply: Reply[AsyncReplyBody]) -> HttpResponseBase:
    # [1] The app's own response from authorize, returned as it is
    if isinstance(reply.passthrough, HttpResponseBase):
        return reply.passthrough

    # [2] Bytes, or a stream that is closed whatever happens to it
    if isinstance(reply.body, bytes):
        return _headed(HttpResponse(reply.body, status=reply.status), reply)
    return _AsyncForwardedStream.of(reply, reply.body)


def _needs_asgi() -> HttpResponseBase:
    message = 'async_forward_view needs an ASGI server; under WSGI, use forward_view.'
    return JsonResponse({'error': {'message': message, 'type': 'api_error', 'code': 'internal'}}, status=500)


# --- The views --------------------------------------------------------------------------------------------------------


def forward_view(
    *,
    authorize: Callable[[HttpRequest], object],
    allow: AllowRule | None = None,
    client: Meridian | Callable[[], Meridian] | None = None,
    csrf_exempt: bool = True,
) -> Callable[[HttpRequest], HttpResponseBase]:
    """A sync view for WSGI: `path('api/meridian/chat', forward_view(authorize=...))`.

    Only the literal `True` from `authorize(request)` lets the request through; any Django response (`HttpResponse`,
    `JsonResponse`, a redirect, a `StreamingHttpResponse`) is returned as the app's own answer; anything else, or a
    raise, is a 403. `client` is a Meridian, or a function returning the same long-lived one every time; by default
    the shared client for `settings.MERIDIAN`. The body is read in chunks through the core's 8 MiB cap, never through
    `request.body` (so DATA_UPLOAD_MAX_MEMORY_SIZE does not apply). The view never raises for an API, network or
    configuration problem: each is a response."""
    forwarder: Forwarder[HttpRequest] = Forwarder(
        client=meridian_client if client is None else client,
        authorize=authorize,
        allow=allow,
        response_type=HttpResponseBase,
    )

    def view(request: HttpRequest) -> HttpResponseBase:
        method, content_type, key = _asked(request)
        reply = forwarder.handle(
            request, method=method, content_type=content_type, idempotency_key=key, read_body=lambda: _chunks(request)
        )
        return _response(reply)

    # What Django's csrf_exempt decorator marks, and what its CsrfViewMiddleware reads
    view.__dict__['csrf_exempt'] = csrf_exempt
    return view


def async_forward_view(
    *,
    authorize: Callable[[HttpRequest], object],
    allow: AllowRule | None = None,
    client: AsyncMeridian | Callable[[], AsyncMeridian] | None = None,
    csrf_exempt: bool = True,
) -> Callable[[HttpRequest], Awaitable[HttpResponseBase]]:
    """`forward_view` as an async view, for ASGI only (under WSGI it answers a 500 that says so, and never calls the
    engine). `authorize` may be async; a sync one runs on the event loop, so it must not touch the database (use
    `await request.auser()` in an async one). `client` is an AsyncMeridian, or a plain (not async) function returning
    the same long-lived one every time; by default the shared client for `settings.MERIDIAN` on the running loop.

    Django's ASGI handler reads the whole request body into a temporary file before any view runs; the view then reads
    it in chunks through the core's 8 MiB cap."""
    if inspect.iscoroutinefunction(client):
        raise TypeError(
            'client must be an AsyncMeridian, or a plain function returning one long-lived client: '
            'an async function cannot be called for a client.'
        )
    forwarder: AsyncForwarder[HttpRequest] = AsyncForwarder(
        client=async_meridian_client if client is None else client,
        authorize=authorize,
        allow=allow,
        response_type=HttpResponseBase,
    )

    async def view(request: HttpRequest) -> HttpResponseBase:
        # [1] Under WSGI the stream would be buffered whole, in a throwaway event loop: refused
        if isinstance(request, WSGIRequest):
            return _needs_asgi()

        # [2] The spooled body, in chunks through the core's cap
        method, content_type, key = _asked(request)
        reply = await forwarder.ahandle(
            request, method=method, content_type=content_type, idempotency_key=key, read_body=lambda: _achunks(request)
        )
        return _async_response(reply)

    view.__dict__['csrf_exempt'] = csrf_exempt
    return view
