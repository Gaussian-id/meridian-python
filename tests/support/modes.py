from __future__ import annotations

import inspect
from typing import Any

from meridian import AsyncMeridian, Meridian

MODES = ['sync', 'async']


def make_client(mode: str, engine: Any, **options: Any) -> Any:
    kwargs: dict[str, Any] = {'url': engine.public_url, 'key': engine.project_key, 'transport': engine.transport}
    kwargs.update(options)
    return Meridian(**kwargs) if mode == 'sync' else AsyncMeridian(**kwargs)


async def resolve(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value
