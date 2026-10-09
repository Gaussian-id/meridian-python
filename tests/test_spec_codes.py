import json
from pathlib import Path

from meridian._errors import ENGINE_ERROR_CODES

ROOT = Path(__file__).resolve().parents[1]


def test_the_sdk_codes_match_the_pinned_error_body_enum() -> None:
    spec = json.loads((ROOT / 'spec' / 'meridian-public.json').read_text(encoding='utf-8'))
    enum = spec['components']['schemas']['ErrorBody']['properties']['code']['enum']
    assert list(ENGINE_ERROR_CODES) == enum
