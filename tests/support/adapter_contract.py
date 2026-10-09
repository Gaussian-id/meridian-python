"""What every framework adapter (FastAPI, Django, Flask) must do, as checks that know nothing about a framework.

Each adapter's test module implements `Harness` (it mounts the adapter's forwarding route over the fake engine and
drives it as a browser would) and runs every entry of `CHECKS` against it. The checks cover what the core cannot
enforce for an adapter: the request body is read through the core's cap, the engine's stream is closed on every exit
path, an app's own response is passed through, and the hidden codes stay hidden.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass
from typing import Any, Protocol

from .fake_engine import Scripted, chunk, completion, engine_error, envelope, frame

MESSAGES = [{'role': 'user', 'content': 'hi'}]
CHAT = json.dumps({'messages': MESSAGES}).encode()
STREAM = json.dumps({'messages': MESSAGES, 'stream': True}).encode()
MEBIBYTE = 1 << 20


@dataclass
class Wire:
    """What the browser saw. `close_again` closes the reply's streamed body once more, as a second owner of it would,
    and says how many streamed bodies it closed (0 would make a double-close check pass without closing anything)."""

    status: int
    headers: dict[str, str]
    body: bytes
    close_again: Callable[[], Awaitable[int]]

    def json(self) -> Any:
        return json.loads(self.body)


def allow(_: Any) -> object:
    return True


class Harness(Protocol):
    """One framework's route, built over the fake engine and driven the way a browser would drive it."""

    engine: Any

    def own_response(self, status: int, data: dict[str, Any]) -> object:
        """The framework's own JSON response, which an `authorize` may return."""
        ...

    async def send(
        self,
        *,
        authorize: Callable[[Any], object] = allow,
        method: str = 'POST',
        content_type: str | None = 'application/json',
        body: bytes | Iterator[bytes] = CHAT,
        leave_after: int | None = None,
    ) -> Wire:
        """Mount the route with this `authorize`, send one request, and return what the browser received.

        `body` is the request body, or its chunks, pulled one at a time as the server asks for them (an async harness
        wraps the iterator). `leave_after=n`: the browser goes away once it has received n chunks of the response body
        (0: before the first), and the Wire holds what it got until then."""
        ...


async def streams_through(h: Harness) -> None:
    h.engine.next(Scripted(headers={'x-request-id': 'req_1'}, chunks=[frame(chunk('a')), frame('[DONE]')]))
    wire = await h.send(body=STREAM)
    assert wire.status == 200
    assert wire.headers['content-type'].startswith('text/event-stream')
    assert wire.headers['cache-control'] == 'no-cache'
    assert wire.headers['x-request-id'] == 'req_1'
    assert 'content-length' not in wire.headers and 'content-encoding' not in wire.headers
    assert wire.body == (frame(chunk('a')) + frame('[DONE]')).encode()
    assert h.engine.upstream_closed()


async def buffered_reply_keeps_its_headers(h: Harness) -> None:
    h.engine.next(Scripted(headers={'x-request-id': 'req_2'}, json=completion('hello')))
    wire = await h.send()
    assert wire.status == 200 and wire.headers['content-type'].startswith('application/json')
    assert wire.headers['x-request-id'] == 'req_2'
    assert wire.json()['choices'][0]['message']['content'] == 'hello'


async def rate_limit_keeps_retry_after(h: Harness) -> None:
    h.engine.next(engine_error('rate_limited', {'retry-after': '7'}))
    wire = await h.send()
    assert wire.status == 429 and wire.headers['retry-after'] == '7'
    assert wire.json()['error']['code'] == 'rate_limited'


async def closed_before_it_starts(h: Harness) -> None:
    h.engine.next(Scripted(chunks=[frame(chunk('a')), frame(chunk('b')), frame('[DONE]')]))
    await h.send(body=STREAM, leave_after=0)
    assert h.engine.upstream_closed()
    assert h.engine.parts_read() == 0


async def closed_when_the_browser_leaves_mid_stream(h: Harness) -> None:
    frames: list[bytes | str] = [frame(chunk(str(i))) for i in range(20)]
    h.engine.next(Scripted(chunks=[*frames, frame('[DONE]')]))
    wire = await h.send(body=STREAM, leave_after=1)
    assert wire.body.startswith(frame(chunk('0')).encode())
    assert h.engine.upstream_closed()
    assert h.engine.parts_read() < 21


async def a_second_close_is_harmless(h: Harness) -> None:
    h.engine.next(Scripted(chunks=[frame(chunk('a')), frame('[DONE]')]))
    wire = await h.send(body=STREAM)
    assert wire.status == 200 and h.engine.upstream_closed()
    assert await wire.close_again() >= 1
    assert await wire.close_again() >= 1
    assert h.engine.upstream_closed()

    h.engine.reset()
    h.engine.next(Scripted(chunks=[frame(chunk('a')), frame(chunk('b')), frame('[DONE]')]))
    gone = await h.send(body=STREAM, leave_after=1)
    assert await gone.close_again() >= 1
    assert await gone.close_again() >= 1
    assert h.engine.upstream_closed()


async def an_oversized_body_is_413_without_draining_it(h: Harness) -> None:
    pulled = 0

    def endless() -> Iterator[bytes]:
        nonlocal pulled
        for _ in range(64):
            pulled += 1
            yield b'x' * MEBIBYTE

    wire = await h.send(body=endless())
    assert wire.status == 413 and wire.json()['error']['code'] == 'body_too_large'
    assert 9 <= pulled <= 10  # the 9th MiB passes the 8 MiB cap, and no further chunk is asked for
    assert h.engine.requests == []


async def the_apps_own_response_is_passed_through(h: Harness) -> None:
    wire = await h.send(authorize=lambda _: h.own_response(401, {'login': True}))
    assert wire.status == 401 and wire.json() == {'login': True}
    assert h.engine.requests == []


async def only_the_literal_true_proceeds(h: Harness) -> None:
    for verdict in (1, 'yes', object(), False, None, [True], {'ok': True}):
        wire = await h.send(authorize=lambda _, verdict=verdict: verdict)
        assert wire.status == 403 and wire.json()['error']['code'] == 'permission_denied'
    assert h.engine.requests == []


async def a_raising_authorize_is_403(h: Harness) -> None:
    def authorize(_: Any) -> object:
        raise RuntimeError('boom')

    wire = await h.send(authorize=authorize)
    assert wire.status == 403 and h.engine.requests == []


async def a_hidden_code_is_a_generic_503(h: Harness) -> None:
    h.engine.next(engine_error('key_revoked'))
    wire = await h.send()
    error = wire.json()['error']
    assert wire.status == 503 and error['code'] == 'service_unavailable'
    assert error['message'] == 'The service is temporarily unavailable.'
    assert b'key_revoked' not in wire.body
    assert wire.headers['x-request-id'] == error['request_id']


async def a_hidden_frame_mid_stream_is_a_generic_frame(h: Harness) -> None:
    h.engine.next(Scripted(chunks=[frame(chunk('a')), frame(envelope('key_revoked'))]))
    wire = await h.send(body=STREAM)
    assert wire.status == 200 and wire.body.startswith(frame(chunk('a')).encode())
    assert b'service_unavailable' in wire.body
    assert b'key_revoked' not in wire.body and b'[DONE]' not in wire.body
    assert h.engine.upstream_closed()


async def only_post_and_only_json_and_before_authorize(h: Harness) -> None:
    asked: list[object] = []

    def authorize(request: Any) -> object:
        asked.append(request)
        return True

    wrong_method = await h.send(method='GET', authorize=authorize)
    assert wrong_method.status == 405 and wrong_method.headers['allow'] == 'POST'
    for content_type in ('text/plain', 'application/x-www-form-urlencoded', None):
        wrong_type = await h.send(content_type=content_type, authorize=authorize)
        assert wrong_type.status == 415
    assert asked == [] and h.engine.requests == []


CHECKS: dict[str, Callable[[Harness], Awaitable[None]]] = {
    check.__name__: check
    for check in (
        streams_through,
        buffered_reply_keeps_its_headers,
        rate_limit_keeps_retry_after,
        closed_before_it_starts,
        closed_when_the_browser_leaves_mid_stream,
        a_second_close_is_harmless,
        an_oversized_body_is_413_without_draining_it,
        the_apps_own_response_is_passed_through,
        only_the_literal_true_proceeds,
        a_raising_authorize_is_403,
        a_hidden_code_is_a_generic_503,
        a_hidden_frame_mid_stream_is_a_generic_frame,
        only_post_and_only_json_and_before_authorize,
    )
}
