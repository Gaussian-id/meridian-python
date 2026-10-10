"""D-27: the SDK never exposes a price or cost. Over-stripping a user key named like one fails safe."""

from __future__ import annotations

import re
from typing import Any, cast

# Prices and costs, plus every rate and sum the operator earns or loses on (the same words the generated admin types
# are filtered by)
PRICED = re.compile(
    'cost|price|per_million|recovery_rate|write_off_rate|bleed|revenue|uncollected|written_off', re.IGNORECASE
)


def _empty(value: Any) -> Any:
    """A new empty container of the same kind, or the value itself when it holds nothing to clean."""
    if isinstance(value, list):
        return []
    if isinstance(value, dict):
        return {}
    return value


def without_prices(value: Any) -> Any:
    """A copy of a decoded JSON value without any key that names a price, a cost or an operator rate, at any depth.

    Built without recursion (an explicit stack), so no nesting depth can make it raise."""
    # [1] The copy's root; each container is made empty, then filled from the stack in the original order
    root = _empty(value)
    pending: list[tuple[Any, Any]] = [(value, root)] if root is not value else []
    while pending:
        source, target = pending.pop()

        # [2] Copy one container's children, leaving out priced keys; containers among them are filled later
        if isinstance(source, list):
            items: list[tuple[Any, Any]] = list(enumerate(cast('list[Any]', source)))
        else:
            fields = cast('dict[Any, Any]', source)
            items = [(k, v) for k, v in fields.items() if not (isinstance(k, str) and PRICED.search(k))]
        for key, child in items:
            copy = _empty(child)
            if isinstance(target, list):
                cast('list[Any]', target).append(copy)
            else:
                cast('dict[Any, Any]', target)[key] = copy
            if copy is not child:
                pending.append((child, copy))
    return root
