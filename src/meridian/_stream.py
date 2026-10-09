from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator, AsyncIterator, Generator, Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

import httpx

from ._errors import Err, MeridianError, Ok, Result, client_error, error_from_envelope, parse_envelope
from ._sse import EventDecoder

USAGE_KEYS = ('prompt_tokens', 'completion_tokens', 'total_tokens')
_STREAM_FAILURES = (httpx.HTTPError, httpx.StreamError)


def token_counts(usage: Mapping[str, Any] | None) -> dict[str, int]:
    """Only the three token counts leave the SDK, and only those present as ints."""
    counts: dict[str, int] = {}
    for key in USAGE_KEYS:
        value = (usage or {}).get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            counts[key] = value
    return counts


@dataclass(frozen=True)
class ChatChunk:
    delta: str
    raw: dict[str, Any]


class _State:
    """The shared reading logic: one event in, a chunk / skip / done out, with the verdict kept on the side."""

    def __init__(self) -> None:
        self.error: MeridianError | None = None
        self.usage: dict[str, int] | None = None
        self.consumed = False
        self.cancelled = False
        self.finished = False

    def step(self, data: str) -> ChatChunk | Literal['skip', 'done']:
        # [1] The end marker, or the engine's final error frame
        if data == '[DONE]':
            return 'done'
        envelope = parse_envelope(data)
        if envelope is not None:
            self.error = error_from_envelope(envelope)
            return 'done'

        # [2] Anything else must be a chat chunk
        try:
            raw: Any = json.loads(data)
        except ValueError:
            self.error = client_error('network_error', 'The stream sent a frame that is not valid JSON.')
            return 'done'
        if not isinstance(raw, dict) or not isinstance(raw.get('choices'), list):  # pyright: ignore[reportUnknownMemberType]
            self.error = client_error('network_error', 'The stream sent a frame that is not a chat chunk.')
            return 'done'
        chunk: dict[str, Any] = raw  # pyright: ignore[reportUnknownVariableType]

        # [3] The usage chunk has no choices; a text chunk carries its delta
        choices: list[Any] = chunk['choices']
        if not choices:
            if isinstance(chunk.get('usage'), dict):
                self.usage = token_counts(chunk['usage'])
            return 'skip'
        first: Any = choices[0]
        delta: Any = cast('dict[str, Any]', first).get('delta') if isinstance(first, dict) else None
        content: Any = cast('dict[str, Any]', delta).get('content') if isinstance(delta, dict) else None
        return ChatChunk(content if isinstance(content, str) else '', chunk)

    def abort(self, message: str) -> None:
        self.error = client_error('aborted', message)

    def failed(self) -> None:
        """The connection broke while reading: aborted if the caller cancelled, otherwise a network error."""
        if self.cancelled:
            self.abort('The stream was aborted.')
        else:
            self.error = client_error('network_error', 'The stream connection failed.')

    def begin(self) -> bool:
        """True when this read may proceed; a second read of an unfinished stream is recorded as aborted."""
        if self.consumed:
            if not self.finished and self.error is None:
                self.abort('This stream was already read.')
            return False
        self.consumed = True
        return True

    def result(self, text: str) -> Result[str]:
        return Err(self.error) if self.error else Ok(text)


def _already_read_frame() -> bytes:
    body = {'error': {'message': 'This stream was already read.', 'type': 'api_error', 'code': 'internal'}}
    return f'data: {json.dumps(body)}\n\n'.encode()


class ChatStream:
    """One streamed reply, read once with `for`."""

    def __init__(self, response: httpx.Response) -> None:
        self._response = response
        self._state = _State()

    @property
    def error(self) -> MeridianError | None:
        return self._state.error

    @property
    def usage(self) -> dict[str, int] | None:
        return self._state.usage

    @property
    def request_id(self) -> str | None:
        return self._response.headers.get('x-request-id')

    def cancel(self) -> None:
        # [1] The caller's own doing: aborted, and nothing reads the body afterwards
        state = self._state
        state.cancelled = True
        if not state.consumed:
            state.consumed = True
            state.abort('The stream was cancelled.')
        self._response.close()

    def _events(self) -> Generator[str, None, None]:
        decoder = EventDecoder()
        for piece in self._response.iter_bytes():
            yield from decoder.feed(piece)
        yield from decoder.close()

    def __iter__(self) -> Iterator[ChatChunk]:
        # [1] A stream is read once
        state = self._state
        if not state.begin():
            return
        events = self._events()
        ended = False
        try:
            # [2] Every event is [DONE], an error frame, the usage chunk or a text chunk
            for data in events:
                out = state.step(data)
                if out == 'done':
                    ended = True
                    return
                if isinstance(out, ChatChunk):
                    yield out
                    if state.cancelled:  # cancel() during iteration: stop now, never read further
                        ended = True
                        state.abort('The stream was cancelled.')
                        return

            # [3] A body that ended without [DONE] lost its connection
            ended = True
            state.error = client_error('network_error', 'The stream ended before it finished.')
        except _STREAM_FAILURES:
            ended = True
            state.failed()
        finally:
            # [4] Done or stopped early: close, which cancels upstream if the body is unfinished
            state.finished = ended
            events.close()
            self._response.close()

    def text(self) -> Result[str]:
        return self._state.result(''.join(c.delta for c in self))

    def iter_bytes(self) -> Iterator[bytes]:
        """The engine's events byte for byte, once: for forwarding to a browser."""
        state = self._state
        if state.consumed:
            yield _already_read_frame()
            return
        state.consumed = True
        try:
            yield from self._response.iter_raw()
        except _STREAM_FAILURES:
            state.failed()
        finally:
            self._response.close()


class AsyncChatStream:
    """One streamed reply, read once with `async for`."""

    def __init__(self, response: httpx.Response) -> None:
        self._response = response
        self._state = _State()

    @property
    def error(self) -> MeridianError | None:
        return self._state.error

    @property
    def usage(self) -> dict[str, int] | None:
        return self._state.usage

    @property
    def request_id(self) -> str | None:
        return self._response.headers.get('x-request-id')

    async def cancel(self) -> None:
        state = self._state
        state.cancelled = True
        if not state.consumed:
            state.consumed = True
            state.abort('The stream was cancelled.')
        await self._response.aclose()

    async def _events(self) -> AsyncGenerator[str, None]:
        decoder = EventDecoder()
        async for piece in self._response.aiter_bytes():
            for data in decoder.feed(piece):
                yield data
        for data in decoder.close():
            yield data

    async def __aiter__(self) -> AsyncIterator[ChatChunk]:
        # [1] A stream is read once
        state = self._state
        if not state.begin():
            return
        events = self._events()
        ended = False
        try:
            # [2] Every event is [DONE], an error frame, the usage chunk or a text chunk
            async for data in events:
                out = state.step(data)
                if out == 'done':
                    ended = True
                    return
                if isinstance(out, ChatChunk):
                    yield out
                    if state.cancelled:  # cancel() during iteration: stop now, never read further
                        ended = True
                        state.abort('The stream was cancelled.')
                        return

            # [3] A body that ended without [DONE] lost its connection
            ended = True
            state.error = client_error('network_error', 'The stream ended before it finished.')
        except asyncio.CancelledError:
            # [4] The task was cancelled: record aborted, close upstream (in finally), and let it propagate
            ended = True
            state.abort('The stream was aborted.')
            raise
        except _STREAM_FAILURES:
            ended = True
            state.failed()
        finally:
            state.finished = ended
            await events.aclose()
            await self._response.aclose()

    async def text(self) -> Result[str]:
        return self._state.result(''.join([c.delta async for c in self]))

    async def aiter_bytes(self) -> AsyncIterator[bytes]:
        """The engine's events byte for byte, once: for forwarding to a browser."""
        state = self._state
        if state.consumed:
            yield _already_read_frame()
            return
        state.consumed = True
        try:
            async for piece in self._response.aiter_raw():
                yield piece
        except _STREAM_FAILURES:
            state.failed()
        finally:
            await self._response.aclose()
