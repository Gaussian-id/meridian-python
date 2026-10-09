from __future__ import annotations

import os

import httpx

from ._chat import AsyncChatApi, AsyncModelsApi, ChatApi, ModelsApi
from ._config import ASYNC_IO, SYNC_IO, io_refusal, url_refusal
from ._errors import MeridianError, config_error
from ._transport import AsyncRefusedTransport, AsyncTransport, RefusedTransport, SyncTransport


def resolve_settings(url: str | None, key: str | None) -> tuple[str, str]:
    return (
        url if url is not None else os.environ.get('MERIDIAN_URL', ''),
        key if key is not None else os.environ.get('MERIDIAN_KEY', ''),
    )


def refusal(
    url: object, key: object, http_client: object, transport: object, kinds: tuple[type, type]
) -> MeridianError | None:
    """The configuration error both clients answer every call with, or None when the settings are usable."""
    if refused := url_refusal(url, 'The Meridian URL'):
        return refused
    if not isinstance(key, str) or not key:
        return config_error('A Meridian project key is required: pass key= or set MERIDIAN_KEY.')
    return io_refusal(http_client, transport, kinds)


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
        url, key = resolve_settings(url, key)
        refused = refusal(url, key, http_client, transport, SYNC_IO)
        self._transport: RefusedTransport | SyncTransport = (
            RefusedTransport(refused)
            if refused
            else SyncTransport(
                url, key, timeout=timeout, max_retries=max_retries, http_client=http_client, transport=transport
            )
        )
        self.chat = ChatApi(self._transport)
        self.models = ModelsApi(self._transport)

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
        url, key = resolve_settings(url, key)
        refused = refusal(url, key, http_client, transport, ASYNC_IO)
        self._transport: AsyncRefusedTransport | AsyncTransport = (
            AsyncRefusedTransport(refused)
            if refused
            else AsyncTransport(
                url, key, timeout=timeout, max_retries=max_retries, http_client=http_client, transport=transport
            )
        )
        self.chat = AsyncChatApi(self._transport)
        self.models = AsyncModelsApi(self._transport)

    async def aclose(self) -> None:
        await self._transport.aclose()

    async def __aenter__(self) -> AsyncMeridian:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()
