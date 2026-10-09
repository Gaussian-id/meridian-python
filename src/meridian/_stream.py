from __future__ import annotations

from collections.abc import Mapping
from typing import Any

USAGE_KEYS = ('prompt_tokens', 'completion_tokens', 'total_tokens')


def token_counts(usage: Mapping[str, Any] | None) -> dict[str, int]:
    """Only the three token counts leave the SDK, and only those present as ints."""
    counts: dict[str, int] = {}
    for key in USAGE_KEYS:
        value = (usage or {}).get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            counts[key] = value
    return counts
