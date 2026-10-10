import asyncio
import gzip
import json
from collections.abc import Callable
from typing import Any

import pytest

from gauss_meridian import AsyncMeridian
from tests.support.fake_engine import Scripted, chunk, engine_error, envelope, frame, usage_chunk
from tests.support.modes import make_client, resolve

MSG: list[Any] = [{'role': 'user', 'content': 'hi'}]
LONG_S = chr(0x17F)  # the long s: Go's strings.EqualFold folds it to s
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


async def eventually(check: Callable[[], bool]) -> None:
    """Lets the event loop run its pending finalisers (an abandoned async generator is closed in a task)."""
    for _ in range(50):
        if check():
            return
        await asyncio.sleep(0)
    raise AssertionError('the stream was never settled')


async def test_streams_text_and_usage(mode: str, engine: Any) -> None:
    s = await open_stream(
        mode, engine, frame(chunk('Hel')), frame(chunk('lo')), frame(usage_chunk(3, 2)), frame('[DONE]')
    )
    assert await collect(s) == ['Hel', 'lo']
    assert s.error is None and s.usage == {'prompt_tokens': 3, 'completion_tokens': 2, 'total_tokens': 5}
    assert engine.upstream_closed()


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
    # One CRLF event whose data field has two lines, with the emoji in the second. The wire is cut
    # between the CR and LF of the first line, inside the 4-byte emoji (1+3), and between the CR and LF
    # of the closing blank line, so each of those splits really happens.
    emoji = 'Halo 😀 selamat'
    head, tail = json.dumps(chunk(emoji), ensure_ascii=False).split(', "choices"')
    first = f'data: {head},\r\ndata: "choices"{tail}\r\n\r\n'
    wire = (first + frame(chunk('ok')).replace('\n', '\r\n') + frame('[DONE]').replace('\n', '\r\n')).encode()
    c1 = wire.index(b'\r\n') + 1
    e = wire.index('😀'.encode()) + 1
    c2 = wire.index(b'\r\n\r\n') + 3
    assert c1 < e < c2
    parts = [wire[:c1], wire[c1:e], wire[e:c2], wire[c2:]]
    assert b''.join(parts) == wire
    assert parts[0].endswith(b'\r') and parts[1].endswith(b'\xf0') and parts[2].startswith(b'\x9f\x98\x80')
    assert parts[2].endswith(b'\r') and parts[3].startswith(b'\n')
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


async def test_a_leading_byte_order_mark_is_ignored(mode: str, engine: Any) -> None:
    wire = b'\xef\xbb\xbf' + frame(chunk('a')).encode() + frame('[DONE]').encode()
    s = await open_stream(mode, engine, wire[:1], wire[1:])  # the BOM itself split across reads
    assert await collect(s) == ['a'] and s.error is None


async def test_a_bare_data_line_is_an_empty_data_field(mode: str, engine: Any) -> None:
    # SSE: a line with no colon is a field name with an empty value, so `data` alone is data '' (as the forwarding
    # guard reads it): an event of nothing is no chat chunk, and ends the stream closed
    s = await open_stream(mode, engine, frame(chunk('a')), b'data\n\n', frame(chunk('late')), frame('[DONE]'))
    assert await collect(s) == ['a']
    assert s.error.code == 'network_error'


async def test_a_bare_data_line_inside_an_event_is_an_empty_line_of_its_data(mode: str, engine: Any) -> None:
    head, tail = json.dumps(chunk('ok')).split(', "choices"')
    s = await open_stream(mode, engine, f'data: {head},\ndata\ndata: "choices"{tail}\n\n', frame('[DONE]'))
    assert await collect(s) == ['ok'] and s.error is None


async def test_a_large_event_under_the_cap_arrives_whole(mode: str, engine: Any) -> None:
    big = 'y' * (1 << 20)
    wire = (frame(chunk(big)) + frame('[DONE]')).encode()
    s = await open_stream(mode, engine, *[wire[i : i + 4096] for i in range(0, len(wire), 4096)])
    assert await collect(s) == [big] and s.error is None


async def test_an_oversized_event_fails_closed(mode: str, engine: Any) -> None:
    huge = b'data: ' + b'x' * (17 * 1024 * 1024)  # never terminated: the cap trips before any later event is read
    s = await open_stream(mode, engine, huge, frame(chunk('late')), frame('[DONE]'))
    assert await collect(s) == []
    assert s.error.code == 'network_error' and '16 MiB' in s.error.message
    assert engine.upstream_closed()


async def test_a_mid_stream_error_frame_ends_with_that_error(mode: str, engine: Any) -> None:
    s = await open_stream(mode, engine, frame(chunk('a')), frame(envelope('upstream_error')))
    assert await collect(s) == ['a']
    assert (s.error.code, s.error.status) == ('upstream_error', 502)
    assert engine.upstream_closed()


async def test_an_error_frame_without_a_request_id_takes_the_header_one(mode: str, engine: Any) -> None:
    bare = {'error': {'message': 'boom', 'type': 'api_error', 'code': 'upstream_error'}}
    s = await open_stream(mode, engine, frame(bare), headers={'x-request-id': 'req_header'})
    await collect(s)
    assert s.error.code == 'upstream_error' and s.error.request_id == 'req_header'


async def test_an_error_frame_keeps_its_own_request_id(mode: str, engine: Any) -> None:
    s = await open_stream(mode, engine, frame(envelope('upstream_error', 'req_frame')), headers={'x-request-id': 'h'})
    await collect(s)
    assert s.error.request_id == 'req_frame'


async def test_a_dropped_connection_is_a_network_error_never_success(mode: str, engine: Any) -> None:
    s = await open_stream(mode, engine, frame(chunk('a')), frame(chunk('b')), drop_after=1)
    text = await resolve(s.text())
    assert text.error.code == 'network_error'
    assert engine.upstream_closed()


async def test_a_body_that_ends_without_done_is_a_network_error(mode: str, engine: Any) -> None:
    s = await open_stream(mode, engine, frame(chunk('a')))
    assert (await resolve(s.text())).error.code == 'network_error'


async def test_a_malformed_frame_is_a_network_error(mode: str, engine: Any) -> None:
    s = await open_stream(mode, engine, frame('{not json'), frame(chunk('late')), frame('[DONE]'))
    assert await collect(s) == []
    assert s.error.code == 'network_error' and 'valid JSON' in s.error.message
    assert engine.upstream_closed()


@pytest.mark.parametrize(
    ('parts', 'scripted'),
    [
        pytest.param([frame(chunk('a')), frame(chunk('b'))], {'drop_after': 1}, id='dropped'),
        pytest.param([frame(chunk('a'))], {}, id='ended-without-done'),
        pytest.param([frame('{not json')], {}, id='not-json'),
        pytest.param([frame([1, 2])], {}, id='not-a-chunk'),
        pytest.param([b'data: ' + b'x' * (17 * 1024 * 1024)], {}, id='too-large'),
    ],
)
async def test_a_failure_on_the_sdks_side_of_a_stream_keeps_the_responses_request_id(
    mode: str, engine: Any, parts: list[Any], scripted: dict[str, Any]
) -> None:
    s = await open_stream(mode, engine, *parts, headers={'x-request-id': 'req_stream'}, **scripted)
    await collect(s)
    assert (s.error.code, s.error.request_id) == ('network_error', 'req_stream')


P1_FRAME = {
    'error': {'message': 'm', 'type': 'rate_limit_error', 'code': 'insufficient_credit'},
    'Error': None,
    'choices': [{'index': 0, 'delta': {'content': 'x'}, 'finish_reason': None}],
}
P1B_FRAME = {'error': {'message': 'm', 'type': 'rate_limit_error', 'code': 'insufficient_credit', 'Code': 'x'}}


@pytest.mark.parametrize(
    'bad',
    [
        pytest.param(P1_FRAME, id='P1'),
        pytest.param(P1B_FRAME, id='P1b'),
        pytest.param({**chunk('x'), 'Error': 'boom'}, id='top-level-Error'),
        pytest.param({**chunk('x'), 'choice' + LONG_S: []}, id='top-level-choice-long-s'),
        pytest.param(
            {'error': {'message': 'm', 'me' + LONG_S * 2 + 'age': 'x', 'type': 'api_error', 'code': 'internal'}},
            id='message-long-s-inside-error',
        ),
        pytest.param({'error': {'message': 'm'}, **chunk('x')}, id='error-without-a-code-beside-choices'),
        pytest.param({'error': 'boom', **chunk('x')}, id='error-string-beside-choices'),
    ],
)
async def test_a_frame_with_an_error_the_sdk_cannot_read_ends_the_stream_closed(
    mode: str, engine: Any, bad: Any
) -> None:
    s = await open_stream(mode, engine, frame(chunk('a')), frame(bad), frame(chunk('late')), frame('[DONE]'))
    assert await collect(s) == ['a']
    assert s.error.code == 'network_error'
    assert engine.upstream_closed()


async def test_a_chunk_whose_delta_holds_a_key_named_error_is_ordinary_content(mode: str, engine: Any) -> None:
    odd = chunk('b')
    odd['choices'][0]['delta']['Error'] = {'Code': 'insufficient_credit'}
    odd['choices'][0]['delta']['error'] = 'not ours'
    s = await open_stream(mode, engine, frame(chunk('a')), frame(odd), frame('[DONE]'))
    assert await collect(s) == ['a', 'b'] and s.error is None


@pytest.mark.parametrize('constant', ['NaN', 'Infinity', '-Infinity'])
async def test_a_frame_holding_a_non_standard_number_is_a_network_error(mode: str, engine: Any, constant: str) -> None:
    bad = '{"choices": [{"delta": {"content": "late"}}], "x": ' + constant + '}'
    s = await open_stream(mode, engine, frame(chunk('a')), frame(bad), frame('[DONE]'))
    assert await collect(s) == ['a']
    assert s.error.code == 'network_error' and 'valid JSON' in s.error.message
    assert engine.upstream_closed()


async def test_a_frame_nested_too_deep_to_parse_is_a_network_error(mode: str, engine: Any) -> None:
    s = await open_stream(mode, engine, frame('[' * 100_000), frame('[DONE]'))
    assert await collect(s) == []
    assert s.error.code == 'network_error' and 'valid JSON' in s.error.message
    assert engine.upstream_closed()


ALREADY_READ = ('aborted', 'client_error', 'The stream was already read.')


def verdict(error: Any) -> tuple[str, str, str]:
    return (error.code, error.type, error.message)


async def bytes_of(mode: str, stream: Any) -> bytes:
    return b''.join(stream.iter_bytes()) if mode == 'sync' else b''.join([b async for b in stream.aiter_bytes()])


async def test_a_stream_reads_once(mode: str, engine: Any) -> None:
    # A second read is an error, never a silent empty success (story 20; meridian-js and meridian-go do the same)
    s = await open_stream(mode, engine, frame(chunk('a')), frame('[DONE]'))
    assert await collect(s) == ['a'] and s.error is None
    assert await collect(s) == []
    assert verdict(s.error) == ALREADY_READ
    s2 = await open_stream(mode, engine, frame(chunk('a')), frame('[DONE]'))
    await resolve(s2.cancel())
    assert (await collect(s2)) == [] and s2.error.code == 'aborted'


@pytest.mark.parametrize('first', ['iterate', 'text', 'bytes'])
@pytest.mark.parametrize('second', ['iterate', 'text', 'bytes'])
async def test_any_second_read_of_a_finished_stream_records_already_read(
    mode: str, engine: Any, first: str, second: str
) -> None:
    s = await open_stream(mode, engine, frame(chunk('a')), frame('[DONE]'))

    async def read(how: str) -> Any:
        if how == 'iterate':
            return await collect(s)
        if how == 'text':
            return await resolve(s.text())
        return await bytes_of(mode, s)

    await read(first)
    assert s.error is None
    again = await read(second)
    assert verdict(s.error) == ALREADY_READ
    if second == 'text':
        assert again.data is None and verdict(again.error) == ALREADY_READ  # never Ok('')
    if second == 'bytes':
        assert b'The stream was already read.' in again


async def test_a_second_read_keeps_the_verdict_of_a_stream_that_failed(mode: str, engine: Any) -> None:
    s = await open_stream(mode, engine, frame(chunk('a')))  # ends without [DONE]
    first = await resolve(s.text())
    second = await resolve(s.text())
    assert first.error.code == second.error.code == 'network_error'


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
    assert engine.upstream_closed()


async def test_break_then_cancel_ends_as_aborted(mode: str, engine: Any) -> None:
    s = await open_stream(mode, engine, frame(chunk('a')), frame(chunk('b')), frame('[DONE]'))
    if mode == 'sync':
        for _ in s:
            break
        s.cancel()
    else:
        async for _ in s:
            break
        await s.cancel()
    assert s.error.code == 'aborted'
    assert engine.upstream_closed()


async def test_cancel_while_the_iterator_is_still_open_ends_as_aborted(mode: str, engine: Any) -> None:
    s = await open_stream(mode, engine, frame(chunk('a')), frame(chunk('b')), frame('[DONE]'))
    if mode == 'sync':
        it = iter(s)
        next(it)
        s.cancel()
        it.close()
    else:
        ait = s.__aiter__()
        await ait.__anext__()
        await s.cancel()
        await ait.aclose()
    assert s.error.code == 'aborted'
    assert engine.upstream_closed()


async def test_abandoning_the_loop_closes_upstream_and_records_aborted(mode: str, engine: Any) -> None:
    s = await open_stream(mode, engine, frame(chunk('a')), frame(chunk('b')), frame('[DONE]'))
    if mode == 'sync':
        for _ in s:
            break
    else:
        async for _ in s:
            break
    await eventually(lambda: s.error is not None and engine.upstream_closed())
    assert s.error.code == 'aborted'
    assert await collect(s) == []


async def test_a_refusal_before_streaming_is_an_error_value(mode: str, engine: Any) -> None:
    engine.next(engine_error('insufficient_credit'))
    result = await resolve(make_client(mode, engine).chat.stream({'messages': MSG}))
    assert (result.error.code, result.error.status) == ('insufficient_credit', 402)


async def test_iter_bytes_passes_the_engine_bytes_unchanged(mode: str, engine: Any) -> None:
    wire = frame(chunk('a')) + frame('[DONE]')
    s = await open_stream(mode, engine, wire)
    data = b''.join(s.iter_bytes()) if mode == 'sync' else b''.join([b async for b in s.aiter_bytes()])
    assert data == wire.encode()
    assert engine.upstream_closed()


async def test_iter_bytes_hands_over_decoded_bytes_of_a_compressed_response(mode: str, engine: Any) -> None:
    wire = frame(chunk('a')) + frame('[DONE]')
    packed = gzip.compress(wire.encode())
    s = await open_stream(mode, engine, packed[:10], packed[10:], headers={'content-encoding': 'gzip'})
    data = b''.join(s.iter_bytes()) if mode == 'sync' else b''.join([b async for b in s.aiter_bytes()])
    assert data == wire.encode()


async def test_a_compressed_response_still_parses(mode: str, engine: Any) -> None:
    packed = gzip.compress((frame(chunk('a')) + frame('[DONE]')).encode())
    s = await open_stream(mode, engine, packed[:10], packed[10:], headers={'content-encoding': 'gzip'})
    assert await collect(s) == ['a'] and s.error is None


async def test_a_fully_forwarded_stream_is_finished(mode: str, engine: Any) -> None:
    s = await open_stream(mode, engine, frame(chunk('a')), frame('[DONE]'))
    _ = await bytes_of(mode, s)
    assert s.error is None  # finished: nothing went wrong
    assert await collect(s) == [] and verdict(s.error) == ALREADY_READ  # reading it again is the mistake


async def test_cancel_during_iter_bytes_stops_and_records_aborted(mode: str, engine: Any) -> None:
    s = await open_stream(mode, engine, b'data: a\n\n', b'data: b\n\n', b'data: c\n\n')
    seen: list[bytes] = []
    if mode == 'sync':
        for piece in s.iter_bytes():
            seen.append(piece)
            s.cancel()
    else:
        async for piece in s.aiter_bytes():
            seen.append(piece)
            await s.cancel()
    assert seen == [b'data: a\n\n'] and s.error.code == 'aborted'
    assert engine.upstream_closed()


async def test_a_broken_body_during_iter_bytes_is_recorded(mode: str, engine: Any) -> None:
    s = await open_stream(mode, engine, b'data: a\n\n', b'data: b\n\n', drop_after=1)
    data = b''.join(s.iter_bytes()) if mode == 'sync' else b''.join([b async for b in s.aiter_bytes()])
    assert data == b'data: a\n\n' and s.error.code == 'network_error'
    assert engine.upstream_closed()


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
    assert engine.upstream_closed()


async def test_cancelling_the_task_inside_the_loop_body_records_aborted(engine: Any) -> None:
    engine.next(Scripted(chunks=[frame(chunk('a')), frame(chunk('b')), frame('[DONE]')]))
    result = await AsyncMeridian(engine.public_url, engine.project_key, transport=engine.transport).chat.stream(
        {'messages': MSG}
    )
    stream = result.data
    assert stream is not None
    started = asyncio.Event()

    async def consume() -> None:
        async for _ in stream:
            started.set()
            await asyncio.sleep(3600)  # the cancellation lands here, outside the stream's own awaits

    task = asyncio.create_task(consume())
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await eventually(lambda: stream.error is not None and engine.upstream_closed())
    assert stream.error is not None and stream.error.code == 'aborted'
