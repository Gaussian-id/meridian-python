"""FastAPI glue: a client dependency and a ready forwarding endpoint. Install with `meridian[fastapi]`."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

import anyio
from fastapi import Request
from fastapi.responses import Response, StreamingResponse
from starlette.types import Receive, Scope, Send

from ._client import AsyncMeridian, async_shared_client
from .forward import AllowRule, AsyncForwarder, AsyncReplyBody, Reply

__all__ = ['forward_route', 'meridian_client']


async def meridian_client() -> AsyncMeridian:
    """A dependency (`Depends(meridian_client)`): the shared client for `MERIDIAN_URL` / `MERIDIAN_KEY` on the app's
    event loop. It is one long-lived client, never one per request, and it is not closed by the request: the app
    closes it at shutdown. It is `async` on purpose: FastAPI runs a plain dependency in a worker thread, which has no
    event loop, so a plain function could not find the loop's client."""
    return async_shared_client()


class _ForwardedStream(StreamingResponse):
    """A streamed reply that closes the engine's stream on every exit. Starlette never closes a `body_iterator`: not
    when the browser leaves, not when sending fails, and not when the body never started."""

    def __init__(self, reply: Reply[AsyncReplyBody], body: AsyncReplyBody) -> None:
        super().__init__(body, status_code=reply.status, headers=reply.headers)
        self._reply = reply

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            with anyio.CancelScope(shield=True):  # a cancelled request must still close the engine's stream
                await self._reply.aclose()


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
    AsyncMeridian or a function returning one; by default the shared `MERIDIAN_URL` / `MERIDIAN_KEY` client. The
    endpoint never raises for an API, network or configuration problem: each is a response."""
    forwarder: AsyncForwarder[Request] = AsyncForwarder(
        client=client, authorize=authorize, allow=allow, response_type=Response
    )

    async def endpoint(request: Request) -> Response:
        # The body is read through the core's cap, chunk by chunk: never `await request.body()`, which reads it all
        reply = await forwarder.ahandle(
            request,
            method=request.method,
            content_type=request.headers.get('content-type'),
            idempotency_key=request.headers.get('idempotency-key'),
            read_body=request.stream,
        )
        return _response(reply)

    return endpoint
