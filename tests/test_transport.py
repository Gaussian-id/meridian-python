from typing import Any

import pytest

from tests.support.fake_engine import Scripted, completion, engine_error
from tests.support.modes import make_client, resolve

MSG: Any = [{'role': 'user', 'content': 'hi'}]
pytestmark = [pytest.mark.anyio, pytest.mark.fake_only]


async def test_retries_twice_then_returns_the_last_error(mode: str, engine: Any, clock: list[float]) -> None:
    engine.next(*[engine_error('upstream_unavailable')] * 3)
    result = await resolve(make_client(mode, engine).chat.create({'messages': MSG}))
    assert result.error.code == 'upstream_unavailable' and len(engine.requests) == 3 and len(clock) == 2


async def test_a_retry_reuses_the_same_idempotency_key(mode: str, engine: Any, clock: list[float]) -> None:
    engine.next(engine_error('service_unavailable'), Scripted(json=completion()))
    result = await resolve(make_client(mode, engine).chat.create({'messages': MSG}))
    assert result.error is None
    keys = {r.headers['idempotency-key'] for r in engine.requests}
    assert len(engine.requests) == 2 and len(keys) == 1


async def test_a_caller_key_is_sent_on_every_attempt(mode: str, engine: Any, clock: list[float]) -> None:
    engine.next(engine_error('rate_limited'), Scripted(json=completion()))
    await resolve(make_client(mode, engine).chat.create({'messages': MSG}, idempotency_key='mine-1'))
    assert [r.headers['idempotency-key'] for r in engine.requests] == ['mine-1', 'mine-1']


async def test_a_non_retryable_error_is_sent_once(mode: str, engine: Any, clock: list[float]) -> None:
    engine.next(engine_error('invalid_request'))
    await resolve(make_client(mode, engine).chat.create({'messages': MSG}))
    assert len(engine.requests) == 1 and clock == []


async def test_retry_after_is_honoured(mode: str, engine: Any, clock: list[float]) -> None:
    engine.next(engine_error('rate_limited', {'retry-after': '2'}), Scripted(json=completion()))
    await resolve(make_client(mode, engine).chat.create({'messages': MSG}))
    assert clock == [2.0]


@pytest.mark.parametrize('value', ['3600', '31', 'Fri, 31 Dec 9999 23:59:59 GMT'])
async def test_a_wait_longer_than_30_s_returns_the_error_at_once(
    mode: str, engine: Any, clock: list[float], value: str
) -> None:
    engine.next(engine_error('rate_limited', {'retry-after': value}))
    result = await resolve(make_client(mode, engine).chat.create({'messages': MSG}))
    assert result.error.code == 'rate_limited' and result.error.request_id and clock == []


# The parser reads a year below 100 as 19xx or 20xx, so the 0000 and 0001 dates are ordinary dates in the past; only the
# first and the 0100 one (kept as written) are extreme. None of them may raise.
ODD_HTTP_DATES = [
    'Mon, 01 Jan 99999999999999999999 00:00:00 GMT',  # a year no datetime holds: used to raise OverflowError
    'Thu, 01 Jan 0100 00:00:00 +1400',  # year 100 as written: far before the epoch, near the smallest date
    'Mon, 01 Jan 0000 00:00:00 GMT',  # read as the year 2000: a past date
    'Mon, 01 Jan 0001 00:00:00 +1400',  # read as the year 2001: a past date
]


@pytest.mark.parametrize('value', ['0', ' ', '1e3', '0x10', 'Wed, 21 Oct 2015 07:28:00 GMT', '-5', *ODD_HTTP_DATES])
async def test_odd_retry_after_values_fall_back_to_backoff_with_jitter(
    mode: str, engine: Any, clock: list[float], value: str
) -> None:
    engine.next(engine_error('rate_limited', {'retry-after': value}), Scripted(json=completion()))
    await resolve(make_client(mode, engine).chat.create({'messages': MSG}))
    assert len(clock) == 1 and 0.5 <= clock[0] < 0.75


async def test_exactly_30_s_is_waited(mode: str, engine: Any, clock: list[float]) -> None:
    engine.next(engine_error('rate_limited', {'retry-after': '30'}), Scripted(json=completion()))
    await resolve(make_client(mode, engine).chat.create({'messages': MSG}))
    assert clock == [30.0]


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), True, -1, 'two'])
async def test_nonsense_max_retries_falls_back_to_two(mode: str, engine: Any, clock: list[float], bad: Any) -> None:
    engine.next(*[engine_error('upstream_unavailable')] * 5)
    await resolve(make_client(mode, engine, max_retries=bad).chat.create({'messages': MSG}))
    assert len(engine.requests) == 3


async def test_max_retries_zero_sends_once(mode: str, engine: Any, clock: list[float]) -> None:
    engine.next(engine_error('upstream_unavailable'))
    await resolve(make_client(mode, engine, max_retries=0).chat.create({'messages': MSG}))
    assert len(engine.requests) == 1


async def test_a_timeout_is_a_retryable_network_error(mode: str, engine: Any, clock: list[float]) -> None:
    engine.next(Scripted(timeout=True), Scripted(timeout=True), Scripted(timeout=True))
    result = await resolve(make_client(mode, engine, timeout=5).chat.create({'messages': MSG}))
    assert (result.error.code, result.error.retryable) == ('network_error', True)
    assert result.error.message == 'No response within 5 s.' and len(engine.requests) == 3


async def test_a_network_failure_then_success(mode: str, engine: Any, clock: list[float]) -> None:
    engine.next(Scripted(network_error=True), Scripted(json=completion('ok')))
    result = await resolve(make_client(mode, engine).chat.create({'messages': MSG}))
    assert result.data['choices'][0]['message']['content'] == 'ok'


def test_a_future_http_date_is_a_wait() -> None:
    from datetime import datetime, timedelta, timezone
    from email.utils import format_datetime

    from gauss_meridian._transport import retry_after_seconds

    date = format_datetime(datetime.now(timezone.utc) + timedelta(seconds=10), usegmt=True)
    assert 8 < (retry_after_seconds(date) or 0) <= 10
    assert retry_after_seconds(None) is None


@pytest.mark.parametrize('value', ODD_HTTP_DATES)
def test_an_odd_http_date_is_no_wait_and_never_raises(value: str) -> None:
    from gauss_meridian._transport import retry_after_seconds, valid_retry_after

    assert retry_after_seconds(value) is None
    assert valid_retry_after(value) in (None, value)  # a date the engine could send or not: only a raise is wrong


@pytest.mark.parametrize('value', ODD_HTTP_DATES)
@pytest.mark.parametrize('status', [400, 429])
async def test_an_odd_retry_after_date_is_an_error_value_from_every_client(
    mode: str, engine: Any, clock: list[float], value: str, status: int
) -> None:
    from gauss_meridian.admin import AsyncMeridianAdmin, MeridianAdmin

    code = 'invalid_request' if status == 400 else 'rate_limited'
    scripted = [Scripted(status=status, headers={'retry-after': value}, json=engine_error(code).json)] * 3
    engine.next(*scripted)
    result = await resolve(make_client(mode, engine).models.list())
    assert result.data is None and result.error.code == code

    engine.next(*scripted)
    admin_cls = MeridianAdmin if mode == 'sync' else AsyncMeridianAdmin
    admin = admin_cls(engine.admin_url, engine.integrator_key, transport=engine.transport)
    listed = await resolve(admin.orgs.list())
    assert listed.data is None and listed.error.code == code


# --- A retry answered with idempotency_conflict -------------------------------------------------------------------


def conflict() -> Scripted:
    body = {'error': {'message': 'in use', 'type': 'invalid_request_error', 'code': 'idempotency_conflict'}}
    return Scripted(status=409, headers={'x-request-id': 'req_409'}, json=body)


def failed_before() -> dict[str, Scripted]:
    return {
        '503': Scripted(
            status=503,
            headers={'x-request-id': 'req_first'},
            json={'error': {'message': 'down', 'type': 'api_error', 'code': 'service_unavailable'}},
        ),
        'network': Scripted(network_error=True),
        'timeout': Scripted(timeout=True),
    }


@pytest.mark.parametrize('first', ['503', 'network', 'timeout'])
@pytest.mark.parametrize('call', ['create', 'stream'])
async def test_a_retry_that_finds_its_key_in_use_returns_the_earlier_failure(
    mode: str, engine: Any, clock: list[float], call: str, first: str
) -> None:
    # The earlier attempt reached the engine (which now holds its key) and failed: that failure is the answer
    engine.next(failed_before()[first], conflict())
    alone = make_client(mode, engine, timeout=5)
    result = await resolve(getattr(alone.chat, call)({'messages': MSG}))
    engine.next(failed_before()[first])
    expected = await resolve(getattr(make_client(mode, engine, timeout=5, max_retries=0).chat, call)({'messages': MSG}))
    assert result.error == expected.error and result.error.code != 'idempotency_conflict'
    assert (result.error.request_id, result.error.retryable) == (expected.error.request_id, expected.error.retryable)
    assert len(engine.requests) == 3


async def test_a_conflict_on_the_first_attempt_is_returned_as_it_is(mode: str, engine: Any, clock: list[float]) -> None:
    engine.next(conflict())
    result = await resolve(make_client(mode, engine).chat.create({'messages': MSG}))
    assert (result.error.code, result.error.status, result.error.request_id) == ('idempotency_conflict', 409, 'req_409')


async def test_a_conflict_after_a_rate_limit_is_a_real_conflict(mode: str, engine: Any, clock: list[float]) -> None:
    # The engine releases the key when it refuses before running the attempt (rate_limited), so the 409 is genuine
    engine.next(engine_error('rate_limited'), conflict())
    result = await resolve(make_client(mode, engine).chat.create({'messages': MSG}))
    assert (result.error.code, result.error.request_id) == ('idempotency_conflict', 'req_409')


async def test_a_call_without_a_key_never_swaps_a_conflict(mode: str, engine: Any, clock: list[float]) -> None:
    engine.next(engine_error('service_unavailable'), conflict())
    result = await resolve(make_client(mode, engine).models.list())
    assert (result.error.code, result.error.request_id) == ('idempotency_conflict', 'req_409')
    assert len(engine.requests) == 2


# --- How long an attempt waits ------------------------------------------------------------------------------------


def timeouts(request: Any) -> dict[str, float]:
    return dict(request.extensions['timeout'])


async def test_a_buffered_chat_waits_300_s_to_read_by_default(mode: str, engine: Any) -> None:
    client = make_client(mode, engine)
    await resolve(client.chat.create({'messages': MSG}))
    assert timeouts(engine.requests[0]) == {'connect': 60.0, 'read': 300.0, 'write': 60.0, 'pool': 60.0}


async def test_streams_and_other_calls_wait_60_s_by_default(mode: str, engine: Any) -> None:
    client = make_client(mode, engine)
    engine.next(Scripted(chunks=['data: [DONE]\n\n']))
    await resolve(client.chat.stream({'messages': MSG}))
    await resolve(client.models.list())
    assert [timeouts(r) for r in engine.requests] == [dict.fromkeys(('connect', 'read', 'write', 'pool'), 60.0)] * 2


@pytest.mark.parametrize('given', [10, 0.5, 400])
async def test_a_timeout_the_caller_sets_applies_to_every_call(mode: str, engine: Any, given: float) -> None:
    client = make_client(mode, engine, timeout=given)
    engine.next(Scripted(json=completion()), Scripted(chunks=['data: [DONE]\n\n']))
    await resolve(client.chat.create({'messages': MSG}))
    await resolve(client.chat.stream({'messages': MSG}))
    await resolve(client.models.list())
    expected = dict.fromkeys(('connect', 'read', 'write', 'pool'), float(given))
    assert [timeouts(r) for r in engine.requests] == [expected] * 3


@pytest.mark.parametrize('bad', [float('nan'), -1, 0, True, 'x'])
async def test_a_timeout_that_is_no_usable_number_counts_as_not_set(mode: str, engine: Any, bad: Any) -> None:
    await resolve(make_client(mode, engine, timeout=bad).chat.create({'messages': MSG}))
    assert timeouts(engine.requests[0])['read'] == 300.0


async def test_a_buffered_chat_that_times_out_reading_says_how_long_it_waited(
    mode: str, engine: Any, clock: list[float]
) -> None:
    engine.next(*[Scripted(timeout=True)] * 3)
    result = await resolve(make_client(mode, engine).chat.create({'messages': MSG}))
    assert (result.error.code, result.error.message) == ('network_error', 'No response within 300 s.')


# Which calls repeat after a failure that may have reached the engine is pinned through the public clients: a chat
# call (it carries an idempotency key) here, GET / DELETE / every admin write / orgs.create in test_admin.py.


@pytest.mark.parametrize('key', [None, '', 'caller-key-1'])
async def test_a_chat_call_repeats_after_a_failure_that_may_have_reached_the_engine(
    mode: str, engine: Any, clock: list[float], key: str | None
) -> None:
    engine.next(engine_error('upstream_unavailable'), Scripted(network_error=True), Scripted(json=completion()))
    result = await resolve(make_client(mode, engine).chat.create({'messages': MSG}, idempotency_key=key))
    assert result.error is None and len(engine.requests) == 3
    assert len({r.headers['idempotency-key'] for r in engine.requests}) == 1


BAD_KEYS = [
    pytest.param('order 42', id='inner-space'),
    pytest.param(' k', id='leading-space'),
    pytest.param('k ', id='trailing-space'),
    pytest.param('\tk', id='tab'),
    pytest.param('k\x7f', id='delete-char'),
    pytest.param('kü', id='non-ascii'),
    pytest.param('k\r\nx-evil: 1', id='crlf'),
    pytest.param('k\n', id='lf'),
    pytest.param('x' * 256, id='256-chars'),
    pytest.param('   ', id='only-spaces'),
]


@pytest.mark.parametrize('key', BAD_KEYS)
@pytest.mark.parametrize('call', ['create', 'stream'])
async def test_an_idempotency_key_the_engine_would_refuse_is_a_config_error_with_nothing_sent(
    mode: str, engine: Any, call: str, key: str
) -> None:
    client = make_client(mode, engine)
    result = await resolve(getattr(client.chat, call)({'messages': MSG}, idempotency_key=key))
    assert (result.error.code, result.error.type, result.error.retryable) == ('invalid_request', 'client_error', False)
    assert engine.requests == []


@pytest.mark.parametrize('key', ['a-b_c.1', 'x' * 255, '!', '~{}[]"\'', 'k-123'])
async def test_a_valid_idempotency_key_is_sent_as_given(mode: str, engine: Any, key: str) -> None:
    await resolve(make_client(mode, engine).chat.create({'messages': MSG}, idempotency_key=key))
    assert engine.requests[0].headers['idempotency-key'] == key


@pytest.mark.parametrize('key', [None, ''])
async def test_no_idempotency_key_means_one_is_made(mode: str, engine: Any, key: str | None) -> None:
    await resolve(make_client(mode, engine).chat.create({'messages': MSG}, idempotency_key=key))
    made = engine.requests[0].headers['idempotency-key']
    assert 1 <= len(made) <= 255 and all(0x21 <= ord(c) <= 0x7E for c in made)


async def test_cancellation_during_the_backoff_propagates(engine: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio

    import gauss_meridian._clock as clock_module

    async def cancelled(seconds: float) -> None:
        raise asyncio.CancelledError

    monkeypatch.setattr(clock_module, 'asleep', cancelled)
    engine.next(*[engine_error('upstream_unavailable')] * 3)
    with pytest.raises(asyncio.CancelledError):
        await make_client('async', engine).chat.create({'messages': MSG})
    assert len(engine.requests) == 1
