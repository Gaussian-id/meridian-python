"""The built package: what the wheel and the sdist hold, what they declare, and what importing them pulls in.

Both are built once with `uv build` into a temporary directory (never into the repository). The import checks run the
wheel itself (a pure-Python wheel imports straight from its zip) in a fresh interpreter, with the frameworks blocked
so they behave as if no extra were installed.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.fake_only
EXTRAS = {'fastapi': 'fastapi', 'django': 'django', 'flask': 'flask'}
FRAMEWORKS = ('fastapi', 'starlette', 'django', 'asgiref', 'flask', 'werkzeug')
# Nothing of the repository's own machinery, tests or secrets may ship
UNSHIPPED = re.compile(r'(^|/)(tests|\.superpowers|\.omc|\.venv|scripts|spec)/|(^|/)\.env')
PRICE = re.compile(r'cost|price', re.IGNORECASE)


@pytest.fixture(scope='module')
def dist(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp('dist')
    uv = os.environ.get('UV') or shutil.which('uv')
    assert uv, 'uv builds the package: run the suite with `uv run`'
    command = [uv, 'build', '--out-dir', str(out)]
    if os.environ.get('UV_OFFLINE'):
        command.append('--offline')
    built = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False, timeout=600)
    assert built.returncode == 0, built.stderr
    return out


def wheel_of(dist: Path) -> Path:
    return next(dist.glob('meridian-*.whl'))


def wheel_files(dist: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(wheel_of(dist)) as wheel:
        return {name: wheel.read(name) for name in wheel.namelist()}


def sdist_files(dist: Path) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    with tarfile.open(next(dist.glob('meridian-*.tar.gz'))) as sdist:
        for member in sdist.getmembers():
            extracted = sdist.extractfile(member) if member.isfile() else None
            if extracted is not None:
                files[member.name.partition('/')[2]] = extracted.read()  # without the meridian-0.0.0/ prefix
    return files


def test_the_wheel_ships_the_typed_package(dist: Path) -> None:
    names = set(wheel_files(dist))
    for required in (
        'meridian/py.typed',
        'meridian/__init__.py',
        'meridian/forward.py',
        'meridian/admin.py',
        'meridian/types.py',
        'meridian/fastapi.py',
        'meridian/django.py',
        'meridian/flask.py',
        'meridian/_generated/public.py',
        'meridian/_generated/admin.py',
    ):
        assert required in names, required


def test_the_sdist_ships_the_package_source(dist: Path) -> None:
    names = set(sdist_files(dist))
    for required in ('src/meridian/py.typed', 'src/meridian/flask.py', 'pyproject.toml', 'README.md', 'LICENSE'):
        assert required in names, required


@pytest.mark.parametrize('kind', ['wheel', 'sdist'])
def test_nothing_but_the_package_ships(dist: Path, kind: str) -> None:
    files = wheel_files(dist) if kind == 'wheel' else sdist_files(dist)
    assert [name for name in files if UNSHIPPED.search(name)] == []

    # The generated types never hold a price field (D-27): they are filtered before generation
    generated = {name: data for name, data in files.items() if '/_generated/' in name}
    assert generated and [name for name, data in generated.items() if PRICE.search(data.decode())] == []


def test_httpx_is_the_only_runtime_dependency_and_each_extra_adds_only_its_framework(dist: Path) -> None:
    metadata = next(data for name, data in wheel_files(dist).items() if name.endswith('.dist-info/METADATA')).decode()
    declared: dict[str, list[str]] = {}
    for line in metadata.splitlines():
        if line.startswith('Requires-Dist:'):
            requirement, _, marker = line.removeprefix('Requires-Dist:').partition(';')
            extra = re.search(r'extra\s*==\s*[\'"]([^\'"]+)[\'"]', marker)
            name = re.match(r'\s*([A-Za-z0-9_.-]+)', requirement)
            assert name is not None, line
            declared.setdefault(extra.group(1) if extra else '', []).append(name.group(1).lower())
    assert declared == {'': ['httpx'], **{extra: [framework] for extra, framework in EXTRAS.items()}}
    provided = {line.split(':', 1)[1].strip() for line in metadata.splitlines() if line.startswith('Provides-Extra:')}
    assert provided == set(EXTRAS)


def run_wheel(dist: Path, body: str, *, block_frameworks: bool) -> subprocess.CompletedProcess[str]:
    """Run `body` in a fresh, isolated interpreter that imports meridian from the built wheel."""
    code = f'import importlib.abc, sys\nsys.path.insert(0, {str(wheel_of(dist))!r})\nFRAMEWORKS = {FRAMEWORKS!r}\n'
    if block_frameworks:
        code += (
            'class Block(importlib.abc.MetaPathFinder):\n'
            '    def find_spec(self, name, path=None, target=None):\n'
            "        if name.partition('.')[0] in FRAMEWORKS:\n"
            "            raise ModuleNotFoundError(f'No module named {name!r}', name=name)\n"
            'sys.meta_path.insert(0, Block())\n'
        )
    return subprocess.run(
        [sys.executable, '-I', '-c', code + body], capture_output=True, text=True, check=False, timeout=120
    )


CORE = (
    'import meridian, meridian.forward, meridian.admin, meridian.types\n'
    'assert meridian.__file__.startswith(sys.path[0]), meridian.__file__\n'
    "loaded = sorted(m for m in sys.modules if m.partition('.')[0] in FRAMEWORKS)\n"
    'assert not loaded, loaded\n'
)


@pytest.mark.parametrize('block_frameworks', [True, False])
def test_the_core_imports_without_any_framework(dist: Path, block_frameworks: bool) -> None:
    result = run_wheel(dist, CORE, block_frameworks=block_frameworks)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('extra', list(EXTRAS))
def test_an_adapter_without_its_framework_names_the_extra(dist: Path, extra: str) -> None:
    body = f'try:\n    import meridian.{extra}\nexcept ImportError as missing:\n    print(missing)\n'
    result = run_wheel(dist, body, block_frameworks=True)
    assert result.returncode == 0, result.stderr
    assert f'meridian.{extra} needs' in result.stdout and f'pip install "meridian[{extra}]"' in result.stdout


@pytest.mark.parametrize('extra', list(EXTRAS))
def test_an_adapter_with_its_framework_imports(dist: Path, extra: str) -> None:
    result = run_wheel(dist, f'import meridian.{extra}\n', block_frameworks=False)
    assert result.returncode == 0, result.stderr


def test_the_source_holds_no_invisible_character_and_no_pyright_ignore() -> None:
    # An invisible character is written as an escape (a reader cannot see a literal BOM), and the source type-checks
    # without suppressing anything
    for path in sorted((ROOT / 'src' / 'meridian').rglob('*.py')):
        text = path.read_text(encoding='utf-8')
        assert chr(0xFEFF) not in text and chr(0x200B) not in text, path
        if '_generated' not in path.parts:
            assert 'pyright: ignore' not in text, path
