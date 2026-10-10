import json
from pathlib import Path
from typing import get_args, get_type_hints

import meridian
from meridian import ErrorCode, ErrorType, MeridianError
from meridian._errors import ENGINE_ERROR_CODES, ENGINE_ERROR_TYPES

ROOT = Path(__file__).resolve().parents[1]
SDK_CODES = ('network_error', 'aborted')


def error_body() -> dict[str, list[str]]:
    spec = json.loads((ROOT / 'spec' / 'meridian-public.json').read_text(encoding='utf-8'))
    properties = spec['components']['schemas']['ErrorBody']['properties']
    return {'code': properties['code']['enum'], 'type': properties['type']['enum']}


def members(alias: object) -> list[str]:
    """A Literal's values, through nested Literals."""
    found: list[str] = []
    for arg in get_args(alias):
        found.extend(members(arg) if get_args(arg) else [arg])
    return found


def test_the_sdk_codes_match_the_pinned_error_body_enum() -> None:
    assert list(ENGINE_ERROR_CODES) == error_body()['code']


def test_error_code_is_the_engines_28_codes_and_the_sdks_two() -> None:
    assert members(ErrorCode) == [*ENGINE_ERROR_CODES, *SDK_CODES] == [*error_body()['code'], *SDK_CODES]
    assert len(members(ErrorCode)) == 30


def test_error_type_is_the_engines_five_types_and_client_error() -> None:
    assert sorted(members(ErrorType)) == sorted([*ENGINE_ERROR_TYPES, 'client_error'])
    assert sorted(ENGINE_ERROR_TYPES) == sorted(error_body()['type'])


def test_meridian_error_is_typed_with_them_and_exports_them() -> None:
    hints = get_type_hints(MeridianError)
    assert hints['code'] == ErrorCode and hints['type'] == ErrorType
    assert {'ErrorCode', 'ErrorType'} <= set(meridian.__all__)
