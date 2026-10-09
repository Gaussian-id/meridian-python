from typing import Any

import pytest

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
    if CONTRACT:
        skip = pytest.mark.skip(reason='needs the in-memory fake engine')
        for item in items:
            if 'fake_only' in item.keywords:
                item.add_marker(skip)


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
