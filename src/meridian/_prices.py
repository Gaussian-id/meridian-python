"""D-27: the SDK never exposes a price or cost. Over-stripping a user key named like one fails safe."""

from __future__ import annotations

import re
from typing import Any, cast

# Prices and costs, plus the two rates the operator earns on (`recovery_rate`, `write_off_rate`)
PRICED = re.compile('cost|price|per_million|recovery_rate|write_off_rate', re.IGNORECASE)


def without_prices(value: Any) -> Any:
    """A copy of a decoded JSON value without any key that names a price, a cost or an operator rate, at any depth."""
    if isinstance(value, list):
        return [without_prices(item) for item in cast('list[Any]', value)]
    if isinstance(value, dict):
        fields = cast('dict[Any, Any]', value)
        return {k: without_prices(v) for k, v in fields.items() if not (isinstance(k, str) and PRICED.search(k))}
    return value
