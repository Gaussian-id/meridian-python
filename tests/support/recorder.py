"""The network under a shared client, for tests: a shared client is built without a transport, so it cannot reach the
fake engine."""

from __future__ import annotations

import json

import httpx
import pytest

from tests.support.fake_engine import completion


class Recorder:
    """Stands in for the network under the default client: records which httpx transport sent each request."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.transports: list[object] = []  # held, so no id is reused while the test runs
        answer = json.dumps(completion('shared')).encode()

        def handle(transport: object, request: httpx.Request) -> httpx.Response:
            self.transports.append(transport)
            return httpx.Response(200, headers={'content-type': 'application/json'}, content=answer)

        async def ahandle(transport: object, request: httpx.Request) -> httpx.Response:
            return handle(transport, request)

        monkeypatch.setattr(httpx.HTTPTransport, 'handle_request', handle)
        monkeypatch.setattr(httpx.AsyncHTTPTransport, 'handle_async_request', ahandle)

    @property
    def senders(self) -> list[int]:
        return [id(transport) for transport in self.transports]
