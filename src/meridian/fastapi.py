"""FastAPI glue: a client dependency and a ready forwarding endpoint. Install with `meridian[fastapi]`.

The shared client lives as long as the app: close it at shutdown in the app's lifespan, on the loop that used it
(`await (await meridian_client()).aclose()`). Under trio there is no asyncio loop to share a client on, so the default
builds a client for every request that nothing closes: pass `forward_route(client=...)` there. A `client=` function is
called for every request and must return the same long-lived client each time, never a new one.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from contextlib import AsyncExitStack

try:
    import anyio  # comes with Starlette, which FastAPI is built on: no dependency of its own
    from fastapi import Request
    from fastapi.responses import Response, StreamingResponse
    from starlette.background import BackgroundTask
    from starlette.types import Receive, Scope, Send
except ImportError as missing:
    raise ImportError('meridian.fastapi needs FastAPI: install it with `pip install "meridian[fastapi]"`.') from missing

from ._client import AsyncMeridian, async_shared_client
from .forward import AllowRule, AsyncForwarder, AsyncReplyBody, Reply

__all__ = ['forward_route', 'meridian_client']

# How long closing the engine's stream may take once the request is over (a hung connection must not hold the request)
_CLOSE_WITHIN = 5


async def meridian_client() -> AsyncMeridian:
    """A dependency (`Depends(meridian_client)`): the shared client for `MERIDIAN_URL` / `MERIDIAN_KEY` on the app's
    event loop. It is one long-lived client, never one per request, and it is not closed by the request: the app
    closes it at shutdown (`await (await meridian_client()).aclose()` in the lifespan). It is `async` on purpose:
    FastAPI runs a plain dependency in a worker thread, which has no event loop, so a plain function could not find
    the loop's client. Being a coroutine function, it is not a `client=` for `forward_route`."""
    return async_shared_client()


async def _close(reply: Reply[AsyncReplyBody]) -> None:
    # Shielded, so a cancelled request still closes the engine's stream; bounded, so a close that hangs is given up
    with anyio.move_on_after(_CLOSE_WITHIN, shield=True):
        await reply.aclose()


class _ForwardedStream(StreamingResponse):
    """A streamed reply that closes the engine's stream on every exit. Starlette never closes a `body_iterator`: not
    when the browser leaves, not when sending fails, and not when the body never started."""

    def __init__(self, reply: Reply[AsyncReplyBody], body: AsyncReplyBody) -> None:
        super().__init__(body, status_code=reply.status, headers=reply.headers)
        self._reply = reply

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        # [1] Starlette runs a background task before returning (even after the browser left): hold it back
        background: BackgroundTask | None = self.background
        self.background = None

        # [2] Send, then close the engine's stream whatever happened
        try:
            await super().__call__(scope, receive, send)
        finally:
            await _close(self._reply)

        # [3] The background task runs once the engine's stream is closed, as Starlette would have run it
        if background is not None:
            await background()


def _response(reply: Reply[AsyncReplyBody]) -> Response:
    # [1] The app's own response from authorize, returned as it is
    if isinstance(reply.passthrough, Response):
        return reply.passthrough

    # [2] Bytes, or a stream that is closed whatever happens to it
    if isinstance(reply.body, bytes):
        return Response(content=reply.body, status_code=reply.status, headers=reply.headers)
    return _ForwardedStream(reply, reply.body)


def forward_route(
    *,
    authorize: Callable[[Request], object],
    allow: AllowRule | None = None,
    client: AsyncMeridian | Callable[[], AsyncMeridian] | None = None,
) -> Callable[[Request], Awaitable[Response]]:
    """An async endpoint for `app.post(path)(forward_route(authorize=...))`.

    `authorize(request)` may be async. Only the literal `True` lets the request through; a Starlette/FastAPI `Response`
    is returned as the app's own answer (a 401, a redirect); anything else, or a raise, is a 403. `client` is an
    AsyncMeridian, or a plain (not async) function returning the same long-lived one every time; by default the shared
    `MERIDIAN_URL` / `MERIDIAN_KEY` client. The endpoint never raises for an API, network or configuration problem:
    each is a response.

    A streamed reply closes the engine's stream when it is sent, however sending ends. When FastAPI never sends it
    (a function-scoped yield dependency raising on exit, say), the stream is closed with the request's dependencies,
    on FastAPI versions that expose them to the request (`fastapi_inner_astack`, as the pinned version does); on
    others it may stay open, since nothing closes it, not even when the reply is collected."""
    if inspect.iscoroutinefunction(client):
        raise TypeError(
            'client must be an AsyncMeridian, or a plain function returning one long-lived client: '
            'an async function (such as meridian_client) cannot be called for a client.'
        )
    forwarder: AsyncForwarder[Request] = AsyncForwarder(
        client=client, authorize=authorize, allow=allow, response_type=Response
    )

    async def endpoint(request: Request) -> Response:
        # [1] The body is read through the core's cap, chunk by chunk: never `await request.body()`, which reads it all
        reply = await forwarder.ahandle(
            request,
            method=request.method,
            content_type=request.headers.get('content-type'),
            idempotency_key=request.headers.get('idempotency-key'),
            read_body=request.stream,
        )

        # [2] A response FastAPI never sends still closes the engine's stream, with the request's dependencies
        stack = request.scope.get('fastapi_inner_astack')
        if isinstance(stack, AsyncExitStack):
            stack.push_async_callback(_close, reply)
        return _response(reply)

    return endpoint
