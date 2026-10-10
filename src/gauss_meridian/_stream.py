from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator, Generator, Iterator, Mapping
from dataclasses import dataclass, replace
from typing import Any, Literal, cast

import httpx

from ._errors import (
    NOT_JSON,
    Err,
    MeridianError,
    Ok,
    Result,
    client_error,
    envelope_of,
    error_from_envelope,
    error_member,
    is_object,
    load_json,
)
from ._sse import EventDecoder, EventTooLarge

USAGE_KEYS = ('prompt_tokens', 'completion_tokens', 'total_tokens')
ALREADY_READ = 'The stream was already read.'
_STREAM_FAILURES = (httpx.HTTPError, httpx.StreamError)


def token_counts(usage: object) -> dict[str, int]:
    """Only the three token counts leave the SDK, and only those present as ints. Anything but an object has none."""
    fields: Mapping[str, Any] = cast('Mapping[str, Any]', usage) if isinstance(usage, Mapping) else {}
    counts: dict[str, int] = {}
    for key in USAGE_KEYS:
        value = fields.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            counts[key] = value
    return counts


@dataclass(frozen=True)
class ChatChunk:
    delta: str
    raw: dict[str, Any]


class _State:
    """The shared reading logic: one event in, a chunk / skip / done out, with the verdict kept on the side."""

    def __init__(self, request_id: str | None) -> None:
        self.request_id = request_id  # the response header's id, for an error frame that carries none
        self.error: MeridianError | None = None
        self.usage: dict[str, int] | None = None
        self.consumed = False
        self.cancelled = False
        self.finished = False

    def step(self, data: str) -> ChatChunk | Literal['skip', 'done']:
        # [1] The end marker, or a frame that is not JSON
        if data == '[DONE]':
            return 'done'
        raw: Any = load_json(data)
        if raw is NOT_JSON:
            self.error = self.failure('network_error', 'The stream sent a frame that is not valid JSON.')
            return 'done'

        # [2] A frame with an error member is the engine's final error frame when it is an envelope with a code and a
        # message; any other error member (or a case variant of a member the SDK reads) cannot be read and ends it
        member = error_member(raw)
        if member is not None:
            envelope = envelope_of(member)
            if envelope is None:
                self.error = self.failure('network_error', 'The stream sent an error frame that cannot be read.')
                return 'done'
            error = error_from_envelope(envelope)
            self.error = (
                error if error.request_id or not self.request_id else replace(error, request_id=self.request_id)
            )
            return 'done'

        # [3] Anything else must be a chat chunk
        if not is_object(raw) or not isinstance(raw.get('choices'), list):
            self.error = self.failure('network_error', 'The stream sent a frame that is not a chat chunk.')
            return 'done'
        chunk = raw

        # [4] The usage chunk has no choices; a text chunk carries its delta
        choices: list[Any] = chunk['choices']
        if not choices:
            if isinstance(chunk.get('usage'), dict):
                self.usage = token_counts(chunk['usage'])
            return 'skip'
        first: Any = choices[0]
        delta: Any = cast('dict[str, Any]', first).get('delta') if isinstance(first, dict) else None
        content: Any = cast('dict[str, Any]', delta).get('content') if isinstance(delta, dict) else None
        return ChatChunk(content if isinstance(content, str) else '', chunk)

    def failure(self, code: Literal['network_error', 'aborted'], message: str) -> MeridianError:
        """A failure on the SDK's side of this stream, carrying the response's request id."""
        return client_error(code, message, request_id=self.request_id)

    def abort(self, message: str) -> None:
        """Record `aborted`, unless the stream already ended with a verdict of its own."""
        if self.error is None:
            self.error = self.failure('aborted', message)

    def too_large(self) -> None:
        self.error = self.failure('network_error', 'The stream sent an event larger than 16 MiB.')

    def ended_early(self) -> None:
        self.error = self.failure('network_error', 'The stream ended before it finished.')

    def failed(self) -> None:
        """The connection broke while reading: aborted if the caller cancelled, otherwise a network error."""
        if self.cancelled:
            self.abort('The stream was aborted.')
        else:
            self.error = self.failure('network_error', 'The stream connection failed.')

    def begin(self) -> bool:
        """True when this read may proceed. A stream is read once: any later read (after a read, finished or not, or a
        cancel) is a mistake, recorded as `aborted` unless the stream already ended with an error of its own."""
        if self.consumed:
            self.abort(ALREADY_READ)
            return False
        self.consumed = True
        return True

    def cancel(self) -> None:
        self.cancelled = True
        self.consumed = True
        if not self.finished:
            self.abort('The stream was cancelled.')

    def settle(self, ended: bool) -> None:
        """A read is over. One that stopped before the end (the caller left the loop) is aborted, never silent."""
        if not ended:
            self.abort('The stream was closed before it finished.')
        self.finished = True

    def result(self, text: str) -> Result[str]:
        return Err(self.error) if self.error else Ok(text)


def _already_read_frame() -> bytes:
    body = {'error': {'message': ALREADY_READ, 'type': 'api_error', 'code': 'internal'}}
    return f'data: {json.dumps(body)}\n\n'.encode()


class ChatStream:
    """One streamed reply, read once: with `for`, `text()` or `iter_bytes()`. Any later read gives nothing (`text()`
    an `Err`) and sets `.error` to `aborted` ("The stream was already read."), unless the stream had already ended with
    an error of its own."""

    def __init__(self, response: httpx.Response) -> None:
        self._response = response
        self._state = _State(response.headers.get('x-request-id'))

    @property
    def error(self) -> MeridianError | None:
        return self._state.error

    @property
    def usage(self) -> dict[str, int] | None:
        return self._state.usage

    @property
    def request_id(self) -> str | None:
        return self._state.request_id

    def cancel(self) -> None:
        # The caller's own doing: aborted (unless already finished), and nothing reads the body afterwards
        self._state.cancel()
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
                        return

            # [3] A body that ended without [DONE] lost its connection
            ended = True
            state.ended_early()
        except EventTooLarge:
            ended = True
            state.too_large()
        except _STREAM_FAILURES:
            ended = True
            state.failed()
        finally:
            # [4] Done or abandoned: settle the verdict, then close, which cancels upstream if the body is unfinished
            state.settle(ended)
            events.close()
            self._response.close()

    def text(self) -> Result[str]:
        return self._state.result(''.join(c.delta for c in self))

    def iter_bytes(self) -> Generator[bytes, None, None]:
        """The engine's events byte for byte (content-decoding undone), once: for forwarding to a browser. A second
        read records `aborted` and gives one error frame saying so."""
        # [1] A stream is read once
        state = self._state
        if not state.begin():
            yield _already_read_frame()
            return
        ended = False
        try:
            # [2] Hand every piece over, stopping at once if the caller cancels meanwhile
            for piece in self._response.iter_bytes():
                yield piece
                if state.cancelled:
                    ended = True
                    return
            ended = True
        except _STREAM_FAILURES:
            ended = True
            state.failed()
        finally:
            # [3] Finished, broken or abandoned: settle the verdict and close upstream
            state.settle(ended)
            self._response.close()


class AsyncChatStream:
    """One streamed reply, read once (`async for`, `text()` or `aiter_bytes()`), as ChatStream is.

    `.error` is final only once the iteration is closed: a loop left with `break` is closed by the event loop later,
    so close it yourself (`await stream.cancel()`, or iterate inside `contextlib.aclosing(stream.__aiter__())`) before
    reading `.error`."""

    def __init__(self, response: httpx.Response) -> None:
        self._response = response
        self._state = _State(response.headers.get('x-request-id'))

    @property
    def error(self) -> MeridianError | None:
        return self._state.error

    @property
    def usage(self) -> dict[str, int] | None:
        return self._state.usage

    @property
    def request_id(self) -> str | None:
        return self._state.request_id

    async def cancel(self) -> None:
        self._state.cancel()
        await self._response.aclose()

    async def _events(self) -> AsyncGenerator[str, None]:
        decoder = EventDecoder()
        async for piece in self._response.aiter_bytes():
            for data in decoder.feed(piece):
                yield data
        for data in decoder.close():
            yield data

    async def __aiter__(self) -> AsyncGenerator[ChatChunk, None]:
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
                        return

            # [3] A body that ended without [DONE] lost its connection
            ended = True
            state.ended_early()
        except EventTooLarge:
            ended = True
            state.too_large()
        except asyncio.CancelledError:
            # [4] The task was cancelled: record aborted, close upstream (in finally), and let it propagate
            state.abort('The stream was aborted.')
            raise
        except _STREAM_FAILURES:
            ended = True
            state.failed()
        finally:
            # [5] Done or abandoned (an abandoned async generator is closed by the event loop's finaliser,
            # which is what ends a task cancelled in the consumer's own loop body): settle, then close upstream
            state.settle(ended)
            await events.aclose()
            await self._response.aclose()

    async def text(self) -> Result[str]:
        return self._state.result(''.join([c.delta async for c in self]))

    async def aiter_bytes(self) -> AsyncGenerator[bytes, None]:
        """The engine's events byte for byte (content-decoding undone), once: for forwarding to a browser. A second
        read records `aborted` and gives one error frame saying so."""
        # [1] A stream is read once
        state = self._state
        if not state.begin():
            yield _already_read_frame()
            return
        ended = False
        try:
            # [2] Hand every piece over, stopping at once if the caller cancels meanwhile
            async for piece in self._response.aiter_bytes():
                yield piece
                if state.cancelled:
                    ended = True
                    return
            ended = True
        except asyncio.CancelledError:
            state.abort('The stream was aborted.')
            raise
        except _STREAM_FAILURES:
            ended = True
            state.failed()
        finally:
            # [3] Finished, broken or abandoned: settle the verdict and close upstream
            state.settle(ended)
            await self._response.aclose()
