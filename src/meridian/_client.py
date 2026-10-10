from __future__ import annotations

import asyncio
import threading
import weakref

import httpx

from ._chat import AsyncChatApi, AsyncModelsApi, ChatApi, ModelsApi
from ._config import ASYNC_IO, PUBLIC, SYNC_IO, refusal, resolve_settings
from ._transport import AsyncRefusedTransport, AsyncTransport, RefusedTransport, SyncTransport


class Meridian:
    """The server client: chat and models with a project key (`gmr_`). Never use it in a browser."""

    def __init__(
        self,
        url: str | None = None,
        key: str | None = None,
        *,
        timeout: float | None = None,
        max_retries: int | None = None,
        http_client: httpx.Client | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        """`http_client` and `transport` are mutually exclusive; passing both is a configuration error."""
        base, token = resolve_settings(url, key, PUBLIC)
        refused = refusal(base, token, http_client, transport, PUBLIC, SYNC_IO)
        self._transport: RefusedTransport | SyncTransport = (
            RefusedTransport(refused)
            if refused
            else SyncTransport(
                base, token, timeout=timeout, max_retries=max_retries, http_client=http_client, transport=transport
            )
        )
        self.chat = ChatApi(self._transport)
        self.models = ModelsApi(self._transport)

    @property
    def is_closed(self) -> bool:
        """True once `close()` ran: every call then answers `invalid_request` ("The client is closed.")."""
        return self._transport.is_closed

    def close(self) -> None:
        self._transport.close()

    def __enter__(self) -> Meridian:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class AsyncMeridian:
    """The async server client: the same API as Meridian, awaited."""

    def __init__(
        self,
        url: str | None = None,
        key: str | None = None,
        *,
        timeout: float | None = None,
        max_retries: int | None = None,
        http_client: httpx.AsyncClient | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        """`http_client` and `transport` are mutually exclusive; passing both is a configuration error."""
        base, token = resolve_settings(url, key, PUBLIC)
        refused = refusal(base, token, http_client, transport, PUBLIC, ASYNC_IO)
        self._transport: AsyncRefusedTransport | AsyncTransport = (
            AsyncRefusedTransport(refused)
            if refused
            else AsyncTransport(
                base, token, timeout=timeout, max_retries=max_retries, http_client=http_client, transport=transport
            )
        )
        self.chat = AsyncChatApi(self._transport)
        self.models = AsyncModelsApi(self._transport)

    @property
    def is_closed(self) -> bool:
        """True once `aclose()` ran: every call then answers `invalid_request` ("The client is closed.")."""
        return self._transport.is_closed

    async def aclose(self) -> None:
        await self._transport.aclose()

    async def __aenter__(self) -> AsyncMeridian:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()


# --- The shared default clients -------------------------------------------------------------------------------------
# One long-lived client per (url, key), built on first use, for code that is given no client of its own (the
# forwarding route's default). Only usable settings are kept: a client refused for a missing or bad setting is made
# afresh each time, so an environment fixed after start-up is picked up and a bad one never sticks. A kept client that
# has been closed (`with shared_client() as client:` closes it) is replaced on the next use, never handed out again.

_lock = threading.Lock()
_shared: dict[tuple[str, str], Meridian] = {}
# An httpx.AsyncClient belongs to the event loop it was used on: one per loop. The loop is the weak key, but a client
# that has been used refers back to its loop, so an entry can outlive a loop that nothing else holds: every access
# also drops the entries of loops that are closed
_async_shared: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, dict[tuple[str, str], AsyncMeridian]] = (
    weakref.WeakKeyDictionary()
)


def shared_client(url: str | None = None, key: str | None = None) -> Meridian:
    """The shared client for these settings (defaulting to `MERIDIAN_URL` / `MERIDIAN_KEY`)."""
    # [1] Settings that cannot work give a refusing client, never kept
    base, token = resolve_settings(url, key, PUBLIC)
    if refusal(base, token, None, None, PUBLIC, SYNC_IO) is not None:
        return Meridian(base, token)

    # [2] Otherwise the one client for them, built once (again, if the last one was closed)
    with _lock:
        client = _shared.get((base, token))
        if client is None or client.is_closed:
            client = _shared[base, token] = Meridian(base, token)
        return client


def async_shared_client(url: str | None = None, key: str | None = None) -> AsyncMeridian:
    """`shared_client` for async code: one per settings and per running asyncio event loop. Called with no running
    asyncio loop (trio, for one), it returns a new client each time, which nothing closes: under such a runtime pass
    a `client=` of your own (built once, closed at shutdown) to the forwarder instead of relying on this default.

    The clients of a loop that has since closed are dropped, not closed: their connections belong to that loop, which
    can no longer run the close, and are released when the client is collected."""
    # [1] Settings that cannot work give a refusing client, never kept
    base, token = resolve_settings(url, key, PUBLIC)
    if refusal(base, token, None, None, PUBLIC, ASYNC_IO) is not None:
        return AsyncMeridian(base, token)
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return AsyncMeridian(base, token)

    # [2] Otherwise the one client for them on this loop, built once (again, if the last one was closed); the entries
    #     of closed loops are dropped first
    with _lock:
        for old in [known for known in _async_shared if known.is_closed()]:
            del _async_shared[old]
        clients = _async_shared.setdefault(loop, {})
        client = clients.get((base, token))
        if client is None or client.is_closed:
            client = clients[base, token] = AsyncMeridian(base, token)
        return client
