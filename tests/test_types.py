import json
import re
import subprocess
import sys
from pathlib import Path
from typing import get_args

from typing_extensions import get_type_hints, is_typeddict

from tests.support.spec import admin_document, reachable_schemas, wrapped_operations

ROOT = Path(__file__).resolve().parents[1]
# A price or a rate the operator earns on: none of it may appear in a type an integrator sees (D-27)
PRICED_KEY = re.compile(
    'cost|price|per_million|recovery_rate|write_off_rate|bleed|revenue|uncollected|written_off', re.IGNORECASE
)


def test_spec_source_records_the_engine_commit() -> None:
    from gauss_meridian._version import ENGINE_API_VERSION, ENGINE_SPEC_COMMIT

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
    from gauss_meridian._generated import admin

    seen: set[int] = set()
    keys: list[str] = []
    classes: list[str] = []
    for value in vars(admin).values():
        _walk(value, seen, keys, classes)
    assert keys and classes, 'the generated admin module has no TypedDicts'
    assert [k for k in keys if PRICED_KEY.search(k)] == []
    assert [c for c in classes if re.search('price|cost', c, re.IGNORECASE)] == []
    text = (ROOT / 'src' / 'gauss_meridian' / '_generated' / 'admin.py').read_text(encoding='utf-8')
    assert not re.search('per_million|SupplierPrice', text, re.IGNORECASE)


def test_generated_admin_types_come_only_from_the_wrapped_operations() -> None:
    from gauss_meridian._generated import admin

    wrapped = list(wrapped_operations().values())
    assert len(wrapped) == 21
    schemas = admin_document()['components']['schemas']
    expected = {
        name
        for name in reachable_schemas(wrapped)
        if schemas[name].get('type') == 'object' and not re.search('cost|price', name, re.IGNORECASE)
    }
    seen: set[int] = set()
    classes: list[str] = []
    for value in vars(admin).values():
        _walk(value, seen, [], classes)
    assert sorted(classes) == sorted(expected)
    # The operator-only (Platform*) shapes and the supplier catalogue are gone, with their rates and revenue
    assert [c for c in classes if c.startswith(('Platform', 'Supplier'))] == []


ADMIN_SNIPPET = """\
# pyright: strict
from __future__ import annotations

from typing import TYPE_CHECKING

from typing_extensions import assert_type

from gauss_meridian import Result
from gauss_meridian.admin import AsyncMeridianAdmin, IssuedKey, MeridianAdmin, Secret

if TYPE_CHECKING:
    from gauss_meridian._generated.admin import (
        CatalogueModelBody,
        CredentialMetadata,
        KeyBody,
        LogPage,
        ModelAllowlist,
        OrganisationBody,
        ProjectBody,
        UsageReport,
    )


def sync(admin: MeridianAdmin) -> None:
    assert_type(admin.orgs.create({'name': 'A'}), Result[OrganisationBody])
    assert_type(admin.orgs.get('o'), Result[OrganisationBody])
    assert_type(admin.orgs.get_by_reference('r'), Result[OrganisationBody])
    assert_type(admin.orgs.list(), Result[list[OrganisationBody]])
    assert_type(admin.projects.create('o', {'name': 'P'}), Result[ProjectBody])
    assert_type(admin.projects.get('o', 'p'), Result[ProjectBody])
    assert_type(admin.projects.list('o'), Result[list[ProjectBody]])
    assert_type(admin.projects.update('o', 'p', {'name': 'Q'}), Result[ProjectBody])
    assert_type(admin.projects.delete('o', 'p'), Result[None])
    assert_type(admin.keys.create('o', 'p'), Result[IssuedKey])
    assert_type(admin.keys.list('o', 'p'), Result[list[KeyBody]])
    assert_type(admin.keys.revoke('o', 'p', 'k'), Result[None])
    assert_type(admin.provider_credentials.list('o'), Result[list[CredentialMetadata]])
    assert_type(admin.provider_credentials.put('o', 'gemini', {'credential': 'c'}), Result[CredentialMetadata])
    assert_type(admin.provider_credentials.delete('o', 'gemini'), Result[None])
    assert_type(admin.project_models.get('o', 'p'), Result[ModelAllowlist])
    assert_type(admin.project_models.put('o', 'p', {'models': ['m']}), Result[ModelAllowlist])
    assert_type(admin.models.list(), Result[list[CatalogueModelBody]])
    assert_type(admin.models.get('m'), Result[CatalogueModelBody])
    assert_type(admin.analytics.usage('p'), Result[UsageReport])
    assert_type(admin.logs.list('p'), Result[LogPage])

    org = admin.orgs.get('o')
    if org.error is None:
        assert_type(org.data['name'], str)
        _ = org.data['idd']  # expect-error
    key = admin.keys.create('o', 'p')
    if key.error is None:
        assert_type(key.data['api_key'], Secret)


async def asynchronous(admin: AsyncMeridianAdmin) -> None:
    assert_type(await admin.orgs.get('o'), Result[OrganisationBody])
    assert_type(await admin.keys.create('o', 'p'), Result[IssuedKey])
    assert_type(await admin.keys.revoke('o', 'p', 'k'), Result[None])
    assert_type(await admin.logs.list('p', page_size=10), Result[LogPage])
    org = await admin.orgs.get('o')
    if org.error is None:
        _ = org.data['idd']  # expect-error
"""


def test_admin_results_are_typed_with_the_generated_reply_shapes(tmp_path: Path) -> None:
    # pyright strict over a README-style snippet: every operation's result type, and a misspelt key is an error
    snippet = tmp_path / 'admin_snippet.py'
    snippet.write_text(ADMIN_SNIPPET, encoding='utf-8')
    run = subprocess.run(
        [sys.executable, '-m', 'pyright', '--outputjson', '--pythonpath', sys.executable, str(snippet)],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    )
    report = json.loads(run.stdout)
    errors = [d for d in report['generalDiagnostics'] if d['severity'] == 'error']
    expected = [i for i, line in enumerate(ADMIN_SNIPPET.splitlines()) if line.endswith('# expect-error')]
    assert sorted({d['range']['start']['line'] for d in errors}) == expected, errors
    # Each marked line is the misspelt key (its unknown value may be reported too)
    misspelt = [d for d in errors if '"idd" is not a defined key in "OrganisationBody"' in d['message']]
    assert sorted(d['range']['start']['line'] for d in misspelt) == expected, [d['message'] for d in errors]


STREAM_SNIPPET = """\
# pyright: strict
from contextlib import aclosing

from typing_extensions import assert_type

from gauss_meridian import AsyncMeridian


async def read(client: AsyncMeridian) -> None:
    opened = await client.chat.stream({'messages': [{'role': 'user', 'content': 'Hi'}]})
    if opened.error is None:
        stream = opened.data
        async with aclosing(stream.__aiter__()) as chunks:
            async for chunk in chunks:
                assert_type(chunk.delta, str)
                print(chunk.delta, end='')
                if chunk.delta.endswith('.'):
                    break
        print(stream.error)
        text = await stream.text()
        await stream.cancel()
        print(text)
"""


def test_the_readmes_async_stream_example_type_checks_under_pyright_strict(tmp_path: Path) -> None:
    # `aclosing(stream.__aiter__())` needs an async generator (it has aclose()), not just an async iterator
    snippet = tmp_path / 'stream_snippet.py'
    snippet.write_text(STREAM_SNIPPET, encoding='utf-8')
    run = subprocess.run(
        [sys.executable, '-m', 'pyright', '--outputjson', '--pythonpath', sys.executable, str(snippet)],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=False,
    )
    report = json.loads(run.stdout)
    errors = [d for d in report['generalDiagnostics'] if d['severity'] == 'error']
    assert errors == [], errors


def test_the_issued_key_type_has_the_generated_keys_with_a_secret_api_key() -> None:
    from gauss_meridian._generated import admin as generated
    from gauss_meridian.admin import IssuedKey, Secret

    assert sorted(get_type_hints(IssuedKey)) == sorted(get_type_hints(generated.IssuedKey))
    assert get_type_hints(IssuedKey)['api_key'] is Secret


def test_the_generated_types_match_the_pinned_documents() -> None:
    # Drift between spec/ and _generated/ (a hand edit, or a document re-pinned without regenerating) fails the suite
    run = subprocess.run(
        [sys.executable, 'scripts/generate_types.py', '--check'], capture_output=True, text=True, cwd=ROOT, check=False
    )
    assert run.returncode == 0, run.stdout + run.stderr


def test_generated_public_types_cover_the_chat_shapes() -> None:
    from gauss_meridian._generated import public

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
        'import gauss_meridian, gauss_meridian.admin, gauss_meridian.forward\n'
        "assert 'gauss_meridian._generated.public' not in sys.modules\n"
    )
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, cwd=ROOT, check=False)
    assert result.returncode == 0, result.stderr


def test_meridian_types_imports_at_run_time_without_typing_extensions() -> None:
    # Outside TYPE_CHECKING the names exist as plain aliases of dict[str, Any], so a runtime import never raises
    code = (
        'import importlib.abc, sys\n'
        'class Block(importlib.abc.MetaPathFinder):\n'
        '    def find_spec(self, name, path=None, target=None):\n'
        "        if name == 'typing_extensions':\n"
        "            raise ImportError('typing_extensions is blocked')\n"
        'sys.meta_path.insert(0, Block())\n'
        "sys.modules.pop('typing_extensions', None)\n"
        'from typing import Any\n'
        'from gauss_meridian.types import ('
        'ChatCompletion, ChatCompletionChunk, ChatMessage, ChatParams, ModelList, Usage)\n'
        'for alias in (ChatCompletion, ChatCompletionChunk, ChatMessage, ChatParams, ModelList, Usage):\n'
        '    assert alias == dict[str, Any], alias\n'
        "assert 'gauss_meridian._generated.public' not in sys.modules\n"
    )
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, cwd=ROOT, check=False)
    assert result.returncode == 0, result.stderr
