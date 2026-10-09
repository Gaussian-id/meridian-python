import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from tests.support.contract_guard import (
    is_inside,
    is_loopback_url,
    paid_keys_in,
    paid_keys_in_environ,
    refusal,
    runtime_dir,
    unmarked_engine_tests,
)
from tests.support.engine import ContractEngine

pytestmark = pytest.mark.fake_only

ROOT = Path(__file__).resolve().parents[1]

GEMINI = 'GAUSSMERIDIAN_GEMINI_API_KEY'
OPENROUTER = 'GAUSSMERIDIAN_OPENROUTER_API_KEY'
SECRET = 'v' + 'q' * 24  # a stand-in value, built at run time; no message may ever contain it


# -- the env file text ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ('text', 'names'),
    [
        ('GAUSSMERIDIAN_GEMINI_API_KEY=abc', [GEMINI]),
        ('export GAUSSMERIDIAN_OPENROUTER_API_KEY="x"\r\n', [OPENROUTER]),
        ('gaussmeridian_gemini_api_key = y', [GEMINI]),
        ('set GAUSSMERIDIAN_GEMINI_API_KEY=abc\r\n', [GEMINI]),
        ('SET GAUSSMERIDIAN_OPENROUTER_API_KEY=abc', [OPENROUTER]),
        ('set "GAUSSMERIDIAN_GEMINI_API_KEY=abc"', [GEMINI]),
        ('   export    GAUSSMERIDIAN_GEMINI_API_KEY   =   "abc"   ', [GEMINI]),
        ("GAUSSMERIDIAN_GEMINI_API_KEY='abc'", [GEMINI]),
        ('GAUSSMERIDIAN_GEMINI_API_KEY: abc', [GEMINI]),
        ('\ufeffGAUSSMERIDIAN_GEMINI_API_KEY=abc', [GEMINI]),
        ('A=1\rGAUSSMERIDIAN_GEMINI_API_KEY=abc\rB=2', [GEMINI]),
        ('GAUSSMERIDIAN_GEMINI_API_KEY=abc # trailing comment', [GEMINI]),
        # a quoted value that runs over several lines is a value
        ('GAUSSMERIDIAN_GEMINI_API_KEY="line one\nline two"\nB=2', [GEMINI]),
        ("GAUSSMERIDIAN_OPENROUTER_API_KEY='a\r\nb\r\n'", [OPENROUTER]),
        ('GAUSSMERIDIAN_GEMINI_API_KEY="', [GEMINI]),
        ('GAUSSMERIDIAN_GEMINI_API_KEY="a\\"b"', [GEMINI]),
        # whitespace inside quotes is still a value, and a `#` straight after `=` is not a comment for every parser
        ('GAUSSMERIDIAN_GEMINI_API_KEY=" "', [GEMINI]),
        ('GAUSSMERIDIAN_GEMINI_API_KEY=#abc', [GEMINI]),
        ('GAUSSMERIDIAN_GEMINI_API_KEY=   # looks empty', [GEMINI]),
        # any line that sets it counts, even if a later one empties it
        ('GAUSSMERIDIAN_GEMINI_API_KEY=abc\nGAUSSMERIDIAN_GEMINI_API_KEY=\n', [GEMINI]),
        (f'{OPENROUTER}=a\n{GEMINI}=b\n{OPENROUTER}=c', [OPENROUTER, GEMINI]),
        # not keys
        ('# GAUSSMERIDIAN_GEMINI_API_KEY=abc', []),
        ('   # export GAUSSMERIDIAN_GEMINI_API_KEY=abc', []),
        ('REM set GAUSSMERIDIAN_GEMINI_API_KEY=abc', []),
        ('OTHER=1 # GAUSSMERIDIAN_GEMINI_API_KEY=abc', []),
        ('GAUSSMERIDIAN_GEMINI_API_KEY=', []),
        ('GAUSSMERIDIAN_GEMINI_API_KEY=   \r\n', []),
        ('GAUSSMERIDIAN_GEMINI_API_KEY=""', []),
        ("GAUSSMERIDIAN_GEMINI_API_KEY=''  # empty", []),
        ('set GAUSSMERIDIAN_GEMINI_API_KEY=', []),
        ('set "GAUSSMERIDIAN_GEMINI_API_KEY="', []),
        ('export GAUSSMERIDIAN_GEMINI_API_KEY', []),
        ('XGAUSSMERIDIAN_GEMINI_API_KEY=abc', []),
        ('GAUSSMERIDIAN_GEMINI_API_KEY_OTHER=abc', []),
        ('GAUSSMERIDIAN_DB_PASSWORD=abc', []),
        ('', []),
    ],
)
def test_paid_keys_are_found_by_name_only(text: str, names: list[str]) -> None:
    assert paid_keys_in(text) == names


def test_a_value_never_comes_back() -> None:
    found = paid_keys_in(f'{GEMINI}={SECRET}\n{OPENROUTER}="{SECRET}"')
    assert found == [GEMINI, OPENROUTER]
    assert SECRET not in repr(found)


def test_the_process_environment_is_judged_by_name_and_without_case() -> None:
    assert paid_keys_in_environ({'gaussmeridian_gemini_api_key': 'x', OPENROUTER: ''}) == [GEMINI]
    assert paid_keys_in_environ({OPENROUTER: ' ', GEMINI: 'x'}) == [OPENROUTER, GEMINI]
    assert paid_keys_in_environ({'GAUSSMERIDIAN_DB_PASSWORD': 'x', 'PATH': '/bin'}) == []
    assert paid_keys_in_environ({}) == []


# -- loopback ------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    'url',
    [
        'http://127.0.0.1:8180',
        'http://127.0.0.1',
        'http://127.0.0.2:8181',
        'http://127.255.255.254/',
        'http://localhost:8181',
        'http://LocalHost:8181',
        'https://127.0.0.1:8180/ready',
        'http://[::1]:8180',
        'http://[::ffff:127.0.0.1]:8180',
        'http://[::ffff:7f00:1]:8180',
    ],
)
def test_loopback_urls_are_accepted(url: str) -> None:
    assert is_loopback_url(url) is True


@pytest.mark.parametrize(
    'url',
    [
        'http://127.0.0.1.evil.com',
        'http://localhost.evil.com',
        'http://localhost@evil.com',
        'http://127.0.0.1@evil.com',
        'http://127.0.0.1:80@evil.com',
        'http://evil.com@127.0.0.1',
        'http://user:pass@127.0.0.1:8180',
        'http://127.0.0.1\\@evil.com',
        'http://evil.com\\@127.0.0.1',
        'http://127.0.0.1 .evil.com',
        'http://127.0.0.1\n.evil.com',
        'http://127.0.0.1:8180\n',
        'http://127.0.0.1:8180\t',
        ' http://127.0.0.1:8180',
        'http://127.0.0.1:8180\x7f',
        'https://api.example.com',
        'http://0.0.0.0:8180',
        'http://[::]:8180',
        'http://10.0.0.5',
        'http://192.168.1.10:8180',
        'http://[::ffff:8.8.8.8]',
        'http://[::ffff:808:808]',
        'http://2130706433',
        'http://0x7f000001',
        'http://0x7f.0.0.1',
        'http://0177.0.0.1',
        'http://127.1',
        'http://127.0.0.1.',
        'http://localhost.',
        'http://localhost%2eevil.com',
        'http://[::1%25lo]:8180',
        'http://\uff11\uff12\uff17.0.0.1',
        'http://127.0.0.1:99999',
        'http://127.0.0.1:abc',
        'http://::1',
        'http://[::1',
        'ftp://127.0.0.1',
        'file:///etc/passwd',
        'localhost:8180',
        '//127.0.0.1:8180',
        '127.0.0.1',
        '',
        'not a url',
    ],
)
def test_anything_else_is_refused(url: str) -> None:
    assert is_loopback_url(url) is False


def test_a_value_that_is_not_text_is_refused() -> None:
    not_text: Any = None
    assert is_loopback_url(not_text) is False


# -- containment ---------------------------------------------------------------------------------------------------


def link(target: Path, at: Path) -> None:
    """A directory link: a symlink where the platform allows it, a junction on Windows otherwise."""
    try:
        os.symlink(target, at, target_is_directory=True)
    except OSError:
        if os.name != 'nt':
            pytest.skip('cannot create a symlink here')
        subprocess.run(['cmd', '/c', 'mklink', '/J', str(at), str(target)], check=True, capture_output=True)


def test_containment_resolves_paths(tmp_path: Path) -> None:
    runtime = tmp_path / 'MeridianEngine'
    assert is_inside(runtime, runtime / 'meridian-engine.env')
    assert is_inside(runtime, runtime / 'a' / '..' / 'meridian-engine.env')
    assert not is_inside(runtime, runtime / '..' / 'elsewhere.env')
    assert not is_inside(runtime, runtime)
    assert not is_inside(runtime, tmp_path / 'MeridianEngine2' / 'x.env')
    assert not is_inside(runtime, tmp_path)


def test_containment_accepts_text_paths(tmp_path: Path) -> None:
    assert is_inside(str(tmp_path), str(tmp_path / 'x.env'))


def test_a_link_out_of_the_folder_is_followed(tmp_path: Path) -> None:
    runtime = tmp_path / 'MeridianEngine'
    outside = tmp_path / 'outside'
    runtime.mkdir()
    outside.mkdir()
    link(outside, runtime / 'escape')
    assert not is_inside(runtime, runtime / 'escape' / 'meridian-engine.env')
    assert not is_inside(runtime, runtime / 'escape')


def test_a_link_into_the_folder_is_inside(tmp_path: Path) -> None:
    runtime = tmp_path / 'MeridianEngine'
    runtime.mkdir()
    link(runtime, tmp_path / 'alias')
    assert is_inside(runtime, tmp_path / 'alias' / 'meridian-engine.env')
    assert is_inside(tmp_path / 'alias', runtime / 'meridian-engine.env')


@pytest.mark.skipif(os.name != 'nt', reason='Windows path spelling')
def test_windows_drive_letters_and_separators_are_not_case_or_slash_sensitive(tmp_path: Path) -> None:
    runtime = tmp_path / 'MeridianEngine'
    runtime.mkdir()
    spelled = str(runtime)
    swapped = spelled[0].swapcase() + spelled[1:]
    assert is_inside(swapped, spelled + '\\meridian-engine.env')
    assert is_inside(spelled.replace('\\', '/'), swapped + '\\a/b.env')
    assert is_inside(spelled.upper(), spelled.lower() + '/x.env')
    assert not is_inside(spelled.replace('\\', '/'), swapped + '\\..\\x.env')
    # a folder that does not exist yet (a first run) is compared by spelling alone
    assert is_inside(str(tmp_path / 'Missing'), str(tmp_path / 'MISSING' / 'x.env'))
    assert not is_inside(str(tmp_path / 'Missing'), str(tmp_path / 'MISSING'))


def test_an_unresolvable_path_is_not_inside(tmp_path: Path) -> None:
    assert not is_inside(tmp_path, str(tmp_path / 'x') + '\x00')


# -- the whole judgement -------------------------------------------------------------------------------------------


def runtime_with(tmp_path: Path, text: str = 'GAUSSMERIDIAN_DB_PASSWORD=x\n') -> tuple[dict[str, str], Path]:
    folder = tmp_path / 'Gaussian' / 'MeridianEngine'
    folder.mkdir(parents=True)
    (folder / 'meridian-engine.env').write_text(text, encoding='utf-8')
    return {'LOCALAPPDATA': str(tmp_path)}, folder


def test_a_clean_runtime_is_allowed(tmp_path: Path) -> None:
    environ, _ = runtime_with(tmp_path)
    assert refusal(environ) is None


def test_the_runtime_folder_follows_the_engine_scripts(tmp_path: Path) -> None:
    assert runtime_dir({'LOCALAPPDATA': str(tmp_path)}) == tmp_path / 'Gaussian' / 'MeridianEngine'
    assert runtime_dir({'XDG_STATE_HOME': str(tmp_path)}) == tmp_path / 'gaussian' / 'meridian-engine'
    assert runtime_dir({'HOME': str(tmp_path)}) == tmp_path / '.local' / 'state' / 'gaussian' / 'meridian-engine'
    first = runtime_dir({'LOCALAPPDATA': str(tmp_path / 'a'), 'XDG_STATE_HOME': str(tmp_path / 'b')})
    assert first == tmp_path / 'a' / 'Gaussian' / 'MeridianEngine'
    assert runtime_dir({}) is None
    assert runtime_dir({'LOCALAPPDATA': '', 'HOME': ''}) is None


def test_an_unknown_runtime_folder_is_a_refusal() -> None:
    message = refusal({})
    assert message is not None
    assert 'runtime' in message


@pytest.mark.parametrize('name', [GEMINI, OPENROUTER])
def test_a_paid_key_in_the_env_file_is_refused_by_name(tmp_path: Path, name: str) -> None:
    environ, _ = runtime_with(tmp_path, f'A=1\r\nset {name}={SECRET}\r\n')
    message = refusal(environ)
    assert message is not None
    assert name in message
    assert SECRET not in message


def test_a_paid_key_in_the_process_environment_is_refused_by_name(tmp_path: Path) -> None:
    environ, _ = runtime_with(tmp_path)
    message = refusal({**environ, GEMINI.lower(): SECRET})
    assert message is not None
    assert GEMINI in message
    assert SECRET not in message
    assert refusal({**environ, OPENROUTER: ''}) is None


def test_both_sources_are_named_together(tmp_path: Path) -> None:
    environ, _ = runtime_with(tmp_path, f'{OPENROUTER}={SECRET}')
    message = refusal({**environ, GEMINI: SECRET})
    assert message is not None
    assert GEMINI in message
    assert OPENROUTER in message
    assert SECRET not in message


def test_a_missing_or_unreadable_env_file_is_a_refusal(tmp_path: Path) -> None:
    folder = tmp_path / 'Gaussian' / 'MeridianEngine'
    folder.mkdir(parents=True)
    assert refusal({'LOCALAPPDATA': str(tmp_path)}) is not None
    (folder / 'meridian-engine.env').mkdir()  # a directory cannot be read as text
    assert refusal({'LOCALAPPDATA': str(tmp_path)}) is not None


@pytest.mark.parametrize('raw', [f'{GEMINI}={SECRET}\n'.encode('utf-16'), b'A=\xff\xfe\n', b'A=1\x00\n'])
def test_an_env_file_that_cannot_be_read_as_text_is_a_refusal(tmp_path: Path, raw: bytes) -> None:
    environ, folder = runtime_with(tmp_path)
    (folder / 'meridian-engine.env').write_bytes(raw)
    message = refusal(environ)
    assert message is not None
    assert SECRET not in message


def test_an_env_file_override_must_stay_in_the_runtime_folder(tmp_path: Path) -> None:
    environ, folder = runtime_with(tmp_path)
    (folder / 'other.env').write_text('A=1\n', encoding='utf-8')
    assert refusal({**environ, 'GAUSSMERIDIAN_ENV_FILE': str(folder / 'other.env')}) is None
    outside = tmp_path / 'clean.env'
    outside.write_text('A=1\n', encoding='utf-8')
    for bad in (outside, folder / '..' / 'clean.env', folder):
        message = refusal({**environ, 'GAUSSMERIDIAN_ENV_FILE': str(bad)})
        assert message is not None
        assert 'GAUSSMERIDIAN_ENV_FILE' in message


def test_an_env_file_that_is_a_link_out_of_the_folder_is_refused(tmp_path: Path) -> None:
    environ, folder = runtime_with(tmp_path)
    outside = tmp_path / 'outside'
    outside.mkdir()
    (outside / 'clean.env').write_text('A=1\n', encoding='utf-8')
    link(outside, folder / 'escape')
    message = refusal({**environ, 'GAUSSMERIDIAN_ENV_FILE': str(folder / 'escape' / 'clean.env')})
    assert message is not None
    assert 'GAUSSMERIDIAN_ENV_FILE' in message


def test_a_runtime_dir_override_must_stay_in_the_runtime_folder(tmp_path: Path) -> None:
    environ, folder = runtime_with(tmp_path)
    (folder / 'sub').mkdir()
    assert refusal({**environ, 'MERIDIAN_RUNTIME_DIR': str(folder)}) is None
    assert refusal({**environ, 'MERIDIAN_RUNTIME_DIR': str(folder / 'sub')}) is None
    elsewhere = tmp_path / 'elsewhere'
    elsewhere.mkdir()
    (elsewhere / 'meridian-engine.env').write_text('A=1\n', encoding='utf-8')
    for bad in (elsewhere, folder / '..', folder / 'sub' / '..' / '..', tmp_path):
        message = refusal({**environ, 'MERIDIAN_RUNTIME_DIR': str(bad)})
        assert message is not None
        assert 'MERIDIAN_RUNTIME_DIR' in message


def test_a_runtime_dir_that_is_a_link_out_of_the_folder_is_refused(tmp_path: Path) -> None:
    environ, folder = runtime_with(tmp_path)
    outside = tmp_path / 'outside'
    outside.mkdir()
    link(outside, folder / 'escape')
    message = refusal({**environ, 'MERIDIAN_RUNTIME_DIR': str(folder / 'escape')})
    assert message is not None
    assert 'MERIDIAN_RUNTIME_DIR' in message


def test_the_engines_own_env_file_is_checked_whatever_the_runtime_dir_says(tmp_path: Path) -> None:
    # the engine scripts read <runtime>/meridian-engine.env; a clean decoy in a sub-folder must not hide it
    environ, folder = runtime_with(tmp_path, f'{GEMINI}={SECRET}')
    (folder / 'sub').mkdir()
    (folder / 'sub' / 'meridian-engine.env').write_text('A=1\n', encoding='utf-8')
    message = refusal({**environ, 'MERIDIAN_RUNTIME_DIR': str(folder / 'sub')})
    assert message is not None
    assert GEMINI in message


def test_an_env_file_in_the_runtime_dir_override_is_checked_too(tmp_path: Path) -> None:
    environ, folder = runtime_with(tmp_path)
    (folder / 'sub').mkdir()
    (folder / 'sub' / 'meridian-engine.env').write_text(f'{OPENROUTER}={SECRET}', encoding='utf-8')
    message = refusal({**environ, 'MERIDIAN_RUNTIME_DIR': str(folder / 'sub')})
    assert message is not None
    assert OPENROUTER in message


@pytest.mark.parametrize('name', ['local-api-key.txt', 'local-integrator-key.txt'])
def test_a_key_file_that_links_out_of_the_runtime_folder_is_refused(tmp_path: Path, name: str) -> None:
    environ, folder = runtime_with(tmp_path)
    outside = tmp_path / 'outside'
    outside.mkdir()
    (outside / name).write_text('k', encoding='utf-8')
    assert refusal(environ) is None
    try:
        (folder / name).symlink_to(outside / name)
    except OSError:
        pytest.skip('cannot create a file symlink here')
    message = refusal(environ)
    assert message is not None
    assert name in message


@pytest.mark.parametrize('variable', ['MERIDIAN_PUBLIC_URL', 'MERIDIAN_ADMIN_URL', 'MERIDIAN_URL'])
def test_a_non_loopback_url_is_refused_by_variable_name_only(tmp_path: Path, variable: str) -> None:
    environ, _ = runtime_with(tmp_path)
    for loopback in ('http://127.0.0.1:9999', 'http://localhost:1', 'http://[::1]:2'):
        assert refusal({**environ, variable: loopback}) is None
    message = refusal({**environ, variable: f'http://user:{SECRET}@evil.example.com'})
    assert message is not None
    assert variable in message
    assert SECRET not in message
    assert 'evil.example.com' not in message


def test_the_default_urls_are_loopback(tmp_path: Path) -> None:
    environ, _ = runtime_with(tmp_path)
    assert refusal(environ) is None
    assert refusal({**environ, 'MERIDIAN_PUBLIC_URL': ''}) is not None  # set but empty: not a loopback URL


# -- the engine the contract tests talk to -------------------------------------------------------------------------


def test_the_contract_engine_meets_the_guard_even_without_the_script(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    environ, folder = runtime_with(tmp_path, f'{GEMINI}={SECRET}')
    (folder / 'local-api-key.txt').write_text('gmr_' + '0' * 64 + '\n', encoding='utf-8')
    monkeypatch.setattr(os, 'environ', environ)
    with pytest.raises(RuntimeError) as refused:
        ContractEngine()
    assert GEMINI in str(refused.value)
    assert SECRET not in str(refused.value)


def test_the_contract_engine_reads_its_keys_from_the_checked_runtime_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    environ, folder = runtime_with(tmp_path)
    (folder / 'sub').mkdir()
    project, integrator = 'gmr_' + '2' * 64, 'gmi_' + '3' * 64
    (folder / 'sub' / 'local-api-key.txt').write_text(project + '\n', encoding='utf-8')
    (folder / 'sub' / 'local-integrator-key.txt').write_text(integrator + '\n', encoding='utf-8')
    monkeypatch.setattr(os, 'environ', {**environ, 'MERIDIAN_RUNTIME_DIR': str(folder / 'sub')})
    engine = ContractEngine()
    assert (engine.project_key, engine.integrator_key) == (project, integrator)
    assert (engine.public_url, engine.admin_url) == ('http://127.0.0.1:8180', 'http://127.0.0.1:8181')


# -- which tests may meet a real engine ---------------------------------------------------------------------------


class FakeItem:
    def __init__(self, nodeid: str, fixtures: tuple[str, ...], marks: tuple[str, ...] = ()) -> None:
        self.nodeid = nodeid
        self.fixturenames = fixtures
        self._marks = marks

    def get_closest_marker(self, name: str) -> object | None:
        return object() if name in self._marks else None


def test_a_test_using_the_engine_fixture_must_say_which_engine_it_is_for() -> None:
    items: list[Any] = [
        FakeItem('t::fake', ('mode', 'engine'), ('fake_only',)),
        FakeItem('t::real', ('engine',), ('contract',)),
        FakeItem('t::unmarked', ('mode', 'engine')),
        FakeItem('t::pure', ('clock',)),
        FakeItem('t::other mark', ('engine',), ('anyio',)),
    ]
    assert unmarked_engine_tests(items) == ['t::unmarked', 't::other mark']
    assert unmarked_engine_tests([]) == []


def test_an_item_without_fixture_names_is_not_an_engine_user() -> None:
    bare: Any = object()
    assert unmarked_engine_tests([bare]) == []


MARKED_TESTS = """\
import pytest


def test_pure():
    pass


@pytest.mark.fake_only
def test_fake(engine):
    pass


@pytest.mark.contract
def test_real():
    pass
"""
UNMARKED_TEST = MARKED_TESTS + '\n\ndef test_unmarked_engine_user(engine):\n    pass\n'
NEEDS_FAKE = 'needs the in-memory fake engine'
NEEDS_REAL = 'needs the real local engine'


def pytest_on(tmp_path: Path, tests: str, contract: bool) -> subprocess.CompletedProcess[str]:
    """Our conftest, as a plugin, over a sample file. In contract mode the only test that asks for the engine
    fixture is skipped before it runs, so no engine object is built and nothing is contacted."""
    (tmp_path / 'test_sample.py').write_text(tests, encoding='utf-8')
    (tmp_path / 'pytest.ini').write_text('[pytest]\naddopts = -q -rs -p no:cacheprovider\n', encoding='utf-8')
    env = {'PYTHONPATH': str(ROOT), 'SYSTEMROOT': os.environ.get('SYSTEMROOT', ''), 'PATH': os.environ.get('PATH', '')}
    if contract:
        env['MERIDIAN_CONTRACT'] = '1'
    command = [sys.executable, '-m', 'pytest', '-p', 'tests.conftest', '-c', str(tmp_path / 'pytest.ini')]
    command += ['--rootdir', str(tmp_path), str(tmp_path / 'test_sample.py')]
    return subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)


@pytest.mark.parametrize('contract', [False, True])
def test_collection_stops_at_an_unmarked_engine_test_in_either_mode(tmp_path: Path, contract: bool) -> None:
    done = pytest_on(tmp_path, UNMARKED_TEST, contract)
    assert done.returncode == 4
    assert 'test_unmarked_engine_user' in done.stderr
    assert 'test_pure' not in done.stderr
    assert 'test_fake' not in done.stderr


def test_without_the_real_engine_contract_tests_are_skipped(tmp_path: Path) -> None:
    done = pytest_on(tmp_path, MARKED_TESTS, contract=False)
    assert done.returncode == 0
    assert NEEDS_REAL in done.stdout
    assert NEEDS_FAKE not in done.stdout
    assert '2 passed, 1 skipped' in done.stdout


def test_against_the_real_engine_fake_only_tests_are_skipped(tmp_path: Path) -> None:
    done = pytest_on(tmp_path, MARKED_TESTS, contract=True)
    assert done.returncode == 0
    assert NEEDS_FAKE in done.stdout
    assert NEEDS_REAL not in done.stdout
    assert '2 passed, 1 skipped' in done.stdout
