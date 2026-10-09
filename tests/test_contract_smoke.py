"""The one test that needs the real local engine. It is skipped unless scripts/contract.py runs the suite."""

from typing import Any

import pytest

from tests.support.modes import make_client, resolve

pytestmark = [pytest.mark.anyio, pytest.mark.contract]


async def test_the_local_engine_lists_its_models(mode: str, engine: Any) -> None:
    result = await resolve(make_client(mode, engine).models.list())
    assert result.error is None
    assert result.data['object'] == 'list'
