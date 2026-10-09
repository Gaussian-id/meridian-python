from __future__ import annotations

import os
from typing import Any

import httpx

from ._chat import ChatApi, ModelsApi
from ._config import url_refusal
from ._errors import MeridianError, config_error
from ._transport import RefusedTransport, SyncTransport


def resolve_settings(url: str | None, key: str | None) -> tuple[str, str]:
    return (
        url if url is not None else os.environ.get('MERIDIAN_URL', ''),
        key if key is not None else os.environ.get('MERIDIAN_KEY', ''),
    )


def refusal(url: str, key: str) -> MeridianError | None:
    return url_refusal(url, 'The Meridian URL') or (
        None if key else config_error('A Meridian project key is required: pass key= or set MERIDIAN_KEY.')
    )


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
        url, key = resolve_settings(url, key)
        refused = refusal(url, key)
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
    """Placeholder until Task 4: accepts the sync client's settings so the package imports."""

    def __init__(self, url: str | None = None, key: str | None = None, **options: Any) -> None:
        self._settings = (url, key, options)
