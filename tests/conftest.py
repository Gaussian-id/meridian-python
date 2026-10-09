from typing import Any

import pytest

from tests.support.contract_guard import unmarked_engine_tests
from tests.support.engine import CONTRACT, make_engine
from tests.support.modes import MODES


@pytest.fixture
def anyio_backend() -> str:
    return 'asyncio'


@pytest.fixture(params=MODES)
def mode(request: pytest.FixtureRequest) -> str:
    return str(request.param)


@pytest.fixture
def engine() -> Any:
    return make_engine()


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    # [1] In either mode, a test that uses the engine fixture must say which engine it is for
    unmarked = unmarked_engine_tests(items)
    if unmarked:
        raise pytest.UsageError(
            'these tests use the engine fixture without @pytest.mark.fake_only or @pytest.mark.contract: '
            + ', '.join(unmarked)
        )

    # [2] fake_only tests are skipped against the real engine, contract tests when there is none
    skip_fake = pytest.mark.skip(reason='needs the in-memory fake engine')
    skip_real = pytest.mark.skip(reason='needs the real local engine: run scripts/contract.py')
    for item in items:
        if CONTRACT and 'fake_only' in item.keywords:
            item.add_marker(skip_fake)
        elif not CONTRACT and 'contract' in item.keywords:
            item.add_marker(skip_real)


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Fake timers: records every wait the SDK asks for and returns at once."""
    waits: list[float] = []

    def sleep(seconds: float) -> None:
        waits.append(seconds)

    async def asleep(seconds: float) -> None:
        waits.append(seconds)

    import meridian._clock as clock_module

    monkeypatch.setattr(clock_module, 'sleep', sleep)
    monkeypatch.setattr(clock_module, 'asleep', asleep)
    return waits
