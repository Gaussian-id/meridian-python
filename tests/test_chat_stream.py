import asyncio
import json
from typing import Any

import pytest

from meridian import AsyncMeridian
from tests.support.fake_engine import Scripted, chunk, engine_error, envelope, frame, usage_chunk
from tests.support.modes import make_client, resolve

MSG: list[Any] = [{'role': 'user', 'content': 'hi'}]
pytestmark = [pytest.mark.anyio, pytest.mark.fake_only]


def raw_frame(value: Any) -> str:
    """A frame with its non-ASCII characters left as UTF-8, as a real engine sends them."""
    return f'data: {json.dumps(value, ensure_ascii=False)}\n\n'


async def collect(stream: Any) -> list[str]:
    if hasattr(stream, '__aiter__'):
        return [c.delta async for c in stream]
    return [c.delta for c in stream]


async def open_stream(mode: str, engine: Any, *parts: Any, **scripted: Any) -> Any:
    engine.next(Scripted(chunks=list(parts), **scripted))
    result = await resolve(make_client(mode, engine).chat.stream({'messages': MSG}))
    assert result.error is None
    return result.data


async def test_streams_text_and_usage(mode: str, engine: Any) -> None:
    s = await open_stream(
        mode, engine, frame(chunk('Hel')), frame(chunk('lo')), frame(usage_chunk(3, 2)), frame('[DONE]')
    )
    assert await collect(s) == ['Hel', 'lo']
    assert s.error is None and s.usage == {'prompt_tokens': 3, 'completion_tokens': 2, 'total_tokens': 5}


async def test_always_asks_for_usage_and_streaming(mode: str, engine: Any) -> None:
    engine.next(Scripted(chunks=[frame('[DONE]')]))
    await resolve(
        make_client(mode, engine).chat.stream(
            {'messages': MSG, 'stream': False, 'stream_options': {'include_usage': False}}  # type: ignore[typeddict-item]
        )
    )
    body = json.loads(engine.requests[0].content)
    assert body['stream'] is True and body['stream_options'] == {'include_usage': True}


async def test_frames_split_anywhere_arrive_intact(mode: str, engine: Any) -> None:
    emoji = 'Halo 😀 selamat'
    wire = (raw_frame(chunk(emoji)) + frame(chunk('ok')) + frame('[DONE]')).replace('\n', '\r\n').encode()
    cut = wire.index('😀'.encode()) + 1  # split inside the 4-byte emoji
    cr = wire.index(b'\r\n\r\n') + 1  # split between CR and LF
    parts = [wire[:cr], wire[cr:cut], wire[cut:]]
    s = await open_stream(mode, engine, *parts)
    assert await collect(s) == [emoji, 'ok'] and s.error is None


async def test_a_byte_at_a_time_wire_with_crlf(mode: str, engine: Any) -> None:
    wire = (raw_frame(chunk('añ')) + frame('[DONE]')).replace('\n', '\r\n').encode()
    assert b'\r\n\r\n' in wire
    s = await open_stream(mode, engine, *[wire[i : i + 1] for i in range(len(wire))])
    assert await collect(s) == ['añ']


async def test_a_multi_line_data_field_split_after_cr(mode: str, engine: Any) -> None:
    first = b'data: {"id":"x","object":"chat.completion.chunk","created":0,"model":"m",\r'
    rest = (
        b'\ndata: "choices":[{"index":0,"delta":{"content":"x"},"finish_reason":null}]}\r\n\r\n'
        + frame('[DONE]').encode()
    )
    s = await open_stream(mode, engine, first, rest)
    assert await collect(s) == ['x']


async def test_comment_lines_are_ignored(mode: str, engine: Any) -> None:
    s = await open_stream(mode, engine, b': keep-alive\n\n', frame(chunk('a')), frame('[DONE]'))
    assert await collect(s) == ['a'] and s.error is None


async def test_a_mid_stream_error_frame_ends_with_that_error(mode: str, engine: Any) -> None:
    s = await open_stream(mode, engine, frame(chunk('a')), frame(envelope('upstream_error')))
    assert await collect(s) == ['a']
    assert (s.error.code, s.error.status) == ('upstream_error', 502)


async def test_a_dropped_connection_is_a_network_error_never_success(mode: str, engine: Any) -> None:
    s = await open_stream(mode, engine, frame(chunk('a')), frame(chunk('b')), drop_after=1)
    text = await resolve(s.text())
    assert text.error.code == 'network_error'


async def test_a_body_that_ends_without_done_is_a_network_error(mode: str, engine: Any) -> None:
    s = await open_stream(mode, engine, frame(chunk('a')))
    assert (await resolve(s.text())).error.code == 'network_error'


async def test_a_malformed_frame_is_a_network_error(mode: str, engine: Any) -> None:
    s = await open_stream(mode, engine, frame('{not json'))
    await collect(s)
    assert s.error.code == 'network_error' and 'valid JSON' in s.error.message


async def test_a_stream_reads_once(mode: str, engine: Any) -> None:
    s = await open_stream(mode, engine, frame(chunk('a')), frame('[DONE]'))
    assert await collect(s) == ['a']
    assert await collect(s) == [] and s.error is None  # a finished stream keeps its result
    s2 = await open_stream(mode, engine, frame(chunk('a')), frame('[DONE]'))
    await resolve(s2.cancel())
    assert (await collect(s2)) == [] and s2.error.code == 'aborted'


async def test_cancel_mid_stream_ends_as_aborted(mode: str, engine: Any) -> None:
    s = await open_stream(mode, engine, frame(chunk('a')), frame(chunk('b')), frame('[DONE]'))
    seen: list[str] = []
    if mode == 'sync':
        for c in s:
            seen.append(c.delta)
            s.cancel()
    else:
        async for c in s:
            seen.append(c.delta)
            await s.cancel()
    assert seen == ['a'] and (s.error.code, s.error.retryable) == ('aborted', False)


async def test_a_refusal_before_streaming_is_an_error_value(mode: str, engine: Any) -> None:
    engine.next(engine_error('insufficient_credit'))
    result = await resolve(make_client(mode, engine).chat.stream({'messages': MSG}))
    assert (result.error.code, result.error.status) == ('insufficient_credit', 402)


async def test_iter_bytes_passes_the_engine_bytes_unchanged(mode: str, engine: Any) -> None:
    wire = frame(chunk('a')) + frame('[DONE]')
    s = await open_stream(mode, engine, wire)
    data = b''.join(s.iter_bytes()) if mode == 'sync' else b''.join([b async for b in s.aiter_bytes()])
    assert data == wire.encode()


async def test_cancelling_the_task_closes_upstream_and_records_aborted(engine: Any) -> None:
    engine.next(Scripted(chunks=[frame(chunk('a'))], hang=True))
    result = await AsyncMeridian(engine.public_url, engine.project_key, transport=engine.transport).chat.stream(
        {'messages': MSG}
    )
    stream = result.data
    assert stream is not None
    started = asyncio.Event()

    async def consume() -> None:
        async for _ in stream:
            started.set()

    task = asyncio.create_task(consume())
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert stream.error is not None and stream.error.code == 'aborted'
