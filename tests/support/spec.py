"""The pinned admin document, read the way the admin client is meant to wrap it."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

ROOT = Path(__file__).resolve().parents[2]
COMPONENT_REF = '#/components/'
# D-27 (owner, 10 Oct): these two return only supplier prices, so no SDK wraps them. Named, never counted.
UNWRAPPED_OPERATIONS = frozenset({'supplier_catalogue_active', 'supplier_catalogue_version'})


def admin_document() -> dict[str, Any]:
    return cast('dict[str, Any]', json.loads((ROOT / 'spec' / 'meridian-admin.json').read_text(encoding='utf-8')))


def integrator_operations() -> dict[tuple[str, str], dict[str, Any]]:
    """Every operation whose `security` lists IntegratorKey, keyed by (METHOD, path template)."""
    found: dict[tuple[str, str], dict[str, Any]] = {}
    for path, methods in admin_document()['paths'].items():
        for method, op in cast('dict[str, dict[str, Any]]', methods).items():
            if any('IntegratorKey' in scheme for scheme in op.get('security', [])):
                found[(method.upper(), path)] = op
    return found


def wrapped_operations() -> dict[tuple[str, str], dict[str, Any]]:
    return {key: op for key, op in integrator_operations().items() if op.get('operationId') not in UNWRAPPED_OPERATIONS}


def _refs(node: Any) -> list[str]:
    """Every `#/components/<kind>/<name>` reference below a node, as `<kind>/<name>`."""
    if isinstance(node, list):
        return [r for item in cast('list[Any]', node) for r in _refs(item)]
    if not isinstance(node, dict):
        return []
    found: list[str] = []
    for key, value in cast('dict[str, Any]', node).items():
        if key == '$ref' and isinstance(value, str) and value.startswith(COMPONENT_REF):
            found.append(value.removeprefix(COMPONENT_REF))
        else:
            found.extend(_refs(value))
    return found


def reachable_schemas(operations: list[dict[str, Any]]) -> set[str]:
    """Names of the component schemas the operations reach, directly or through responses and other schemas."""
    components = cast('dict[str, dict[str, Any]]', admin_document()['components'])
    todo = [ref for op in operations for ref in _refs(op)]
    seen: set[str] = set()
    while todo:
        ref = todo.pop()
        if ref not in seen:
            seen.add(ref)
            kind, name = ref.split('/', 1)
            todo.extend(_refs(components[kind][name]))
    return {ref.removeprefix('schemas/') for ref in seen if ref.startswith('schemas/')}
