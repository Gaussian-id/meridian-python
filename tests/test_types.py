import json
import re
import subprocess
import sys
from pathlib import Path
from typing import get_args

from typing_extensions import get_type_hints, is_typeddict

ROOT = Path(__file__).resolve().parents[1]


def test_spec_source_records_the_engine_commit() -> None:
    from meridian._version import ENGINE_API_VERSION, ENGINE_SPEC_COMMIT

    source = json.loads((ROOT / 'spec' / 'SOURCE.json').read_text(encoding='utf-8'))
    assert re.fullmatch(r'[0-9a-f]{12}', source['engineCommit'])
    assert source['engineCommit'] == ENGINE_SPEC_COMMIT
    assert source['publicVersion'] == ENGINE_API_VERSION


def _walk(annotation: object, seen: set[int], keys: list[str], classes: list[str]) -> None:
    # [1] Collect every TypedDict key and class name reachable from an annotation, through unions and containers
    if is_typeddict(annotation):
        if id(annotation) in seen:
            return
        seen.add(id(annotation))
        classes.append(getattr(annotation, '__name__', ''))
        for key, value in get_type_hints(annotation).items():
            keys.append(key)
            _walk(value, seen, keys, classes)
    for arg in get_args(annotation):
        _walk(arg, seen, keys, classes)


def test_generated_admin_types_carry_no_price_or_cost_field() -> None:
    from meridian._generated import admin

    seen: set[int] = set()
    keys: list[str] = []
    classes: list[str] = []
    for value in vars(admin).values():
        _walk(value, seen, keys, classes)
    assert keys and classes, 'the generated admin module has no TypedDicts'
    assert [k for k in keys if re.search('cost|price|per_million', k, re.IGNORECASE)] == []
    assert [c for c in classes if re.search('price|cost', c, re.IGNORECASE)] == []
    text = (ROOT / 'src' / 'meridian' / '_generated' / 'admin.py').read_text(encoding='utf-8')
    assert not re.search('per_million|SupplierPrice', text, re.IGNORECASE)


def test_generated_public_types_cover_the_chat_shapes() -> None:
    from meridian._generated import public

    for name in ('ChatCompletionRequest', 'ChatCompletion', 'ChatCompletionChunk', 'ChatMessage', 'Usage', 'ModelList'):
        assert hasattr(public, name), name


def test_meridian_imports_without_typing_extensions() -> None:
    code = (
        'import importlib.abc, sys\n'
        'class Block(importlib.abc.MetaPathFinder):\n'
        '    def find_spec(self, name, path=None, target=None):\n'
        "        if name == 'typing_extensions':\n"
        "            raise ImportError('typing_extensions is blocked')\n"
        'sys.meta_path.insert(0, Block())\n'
        "sys.modules.pop('typing_extensions', None)\n"
        'import meridian\n'
        "assert 'meridian._generated.public' not in sys.modules\n"
    )
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, cwd=ROOT, check=False)
    assert result.returncode == 0, result.stderr
