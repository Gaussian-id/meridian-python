import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_spec_source_records_the_engine_commit() -> None:
    from meridian._version import ENGINE_API_VERSION, ENGINE_SPEC_COMMIT

    source = json.loads((ROOT / 'spec' / 'SOURCE.json').read_text(encoding='utf-8'))
    assert re.fullmatch(r'[0-9a-f]{12}', source['engineCommit'])
    assert source['engineCommit'] == ENGINE_SPEC_COMMIT
    assert source['publicVersion'] == ENGINE_API_VERSION


def test_generated_admin_types_carry_no_price_or_cost_field() -> None:
    text = (ROOT / 'src' / 'meridian' / '_generated' / 'admin.py').read_text(encoding='utf-8')
    fields = re.findall(r'^\s+([A-Za-z_][A-Za-z0-9_]*)\s*:', text, flags=re.MULTILINE)
    assert fields, 'the generated admin module has no fields'
    assert [name for name in fields if re.search('cost|price', name, re.IGNORECASE)] == []


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
