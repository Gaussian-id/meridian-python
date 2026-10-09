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


@pytest.mark.parametrize('value', ['3600', '31'])
async def test_a_wait_longer_than_30_s_returns_the_error_at_once(
    mode: str, engine: Any, clock: list[float], value: str
) -> None:
    engine.next(engine_error('rate_limited', {'retry-after': value}))
    result = await resolve(make_client(mode, engine).chat.create({'messages': MSG}))
    assert result.error.code == 'rate_limited' and result.error.request_id and clock == []


@pytest.mark.parametrize('value', ['0', ' ', '1e3', '0x10', 'Wed, 21 Oct 2015 07:28:00 GMT', '-5'])
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

    from meridian._transport import retry_after_seconds

    date = format_datetime(datetime.now(timezone.utc) + timedelta(seconds=10), usegmt=True)
    assert 8 < (retry_after_seconds(date) or 0) <= 10
    assert retry_after_seconds(None) is None


def test_a_call_that_is_not_safe_to_repeat_retries_only_rate_limited() -> None:
    from meridian._errors import error_from_body
    from meridian._transport import Call, next_wait

    def err(code: str, status: int) -> Any:
        body = '{"error":{"message":"m","type":"api_error","code":"%s","request_id":"r"}}' % code
        return error_from_body(status, 'r', body)

    unsafe = Call('POST', '/x', retry=False)
    assert next_wait(0, err('upstream_unavailable', 502), None, unsafe, 2) is None
    assert next_wait(0, err('rate_limited', 429), '1', unsafe, 2) == 1.0


async def test_cancellation_during_the_backoff_propagates(engine: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio

    import meridian._clock as clock_module

    async def cancelled(seconds: float) -> None:
        raise asyncio.CancelledError

    monkeypatch.setattr(clock_module, 'asleep', cancelled)
    engine.next(*[engine_error('upstream_unavailable')] * 3)
    with pytest.raises(asyncio.CancelledError):
        await make_client('async', engine).chat.create({'messages': MSG})
    assert len(engine.requests) == 1
