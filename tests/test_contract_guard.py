import json
import os
import shutil
import subprocess
import sys
import urllib.request
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
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


@pytest.fixture(autouse=True)
def no_system_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    """The guard asks the system for proxies (registry on Windows); a test must not depend on this machine's."""
    monkeypatch.setattr(urllib.request, 'getproxies', lambda: dict[str, str]())
    monkeypatch.setattr(urllib.request, 'getproxies_registry', lambda: dict[str, str](), raising=False)


# -- the env file text ---------------------------------------------------------------------------------------------

BOTH = [GEMINI, OPENROUTER]  # what a text that cannot be checked reports

# Env file texts and the names the guard must report. A paid name counts as set unless its line is exactly an empty
# assignment; a character that some reader skips, folds away or breaks a line at means neither name can be checked.
TEXTS: list[tuple[str, list[str]]] = [
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
    ('GAUSSMERIDIAN_GEMINI_API_KEY=abc # trailing comment', [GEMINI]),
    # a quoted value that runs over several lines is a value
    ('GAUSSMERIDIAN_GEMINI_API_KEY="line one\nline two"\nB=2', [GEMINI]),
    ("GAUSSMERIDIAN_OPENROUTER_API_KEY='a\r\nb\r\n'", [OPENROUTER]),
    ('GAUSSMERIDIAN_GEMINI_API_KEY="', [GEMINI]),
    ('GAUSSMERIDIAN_GEMINI_API_KEY="a\\"b"', [GEMINI]),
    ('GAUSSMERIDIAN_GEMINI_API_KEY=" "', [GEMINI]),
    ('GAUSSMERIDIAN_GEMINI_API_KEY=#abc', [GEMINI]),
    ('GAUSSMERIDIAN_GEMINI_API_KEY=   # looks empty', [GEMINI]),
    # any line that sets it counts, even if a later one empties it
    ('GAUSSMERIDIAN_GEMINI_API_KEY=abc\nGAUSSMERIDIAN_GEMINI_API_KEY=\n', [GEMINI]),
    (f'{OPENROUTER}=a\n{GEMINI}=b\n{OPENROUTER}=c', [OPENROUTER, GEMINI]),
    # a quoted value is followed by a new statement on the same line (compose)
    ('GAUSSMERIDIAN_GEMINI_API_KEY=""GAUSSMERIDIAN_GEMINI_API_KEY=VAL', [GEMINI]),
    ('A="" GAUSSMERIDIAN_OPENROUTER_API_KEY=VAL', [OPENROUTER]),
    ("A='' GAUSSMERIDIAN_OPENROUTER_API_KEY=VAL", [OPENROUTER]),
    ('A="x\nline" GAUSSMERIDIAN_OPENROUTER_API_KEY=VAL', [OPENROUTER]),
    ('A="x" GAUSSMERIDIAN_OPENROUTER_API_KEY=VAL', [OPENROUTER]),
    ('A="a\\" GAUSSMERIDIAN_OPENROUTER_API_KEY=VAL', [OPENROUTER]),
    ('A="x\nGAUSSMERIDIAN_OPENROUTER_API_KEY=VAL\n"', [OPENROUTER]),
    # a line that starts with `#` but holds a quote may close a quoted value; compose then reads what follows
    ('A="x\n#"GAUSSMERIDIAN_GEMINI_API_KEY=VAL', [GEMINI]),
    ('A="x\n# a" GAUSSMERIDIAN_GEMINI_API_KEY=VAL', [GEMINI]),
    ('GAUSSMERIDIAN_GEMINI_API_KEY=${UNSET:-VAL}', [GEMINI]),
    ('GAUSSMERIDIAN_GEMINI_API_KEY=$X', [GEMINI]),
    # a whitespace-only value is set for both local-up readers
    ('GAUSSMERIDIAN_GEMINI_API_KEY= ', [GEMINI]),
    ('GAUSSMERIDIAN_GEMINI_API_KEY=\t', [GEMINI]),
    ('GAUSSMERIDIAN_GEMINI_API_KEY=   \r\n', [GEMINI]),
    # anything but exactly an empty assignment counts, wherever the name is on the line
    ("GAUSSMERIDIAN_GEMINI_API_KEY=''  # empty", [GEMINI]),
    ('set "GAUSSMERIDIAN_GEMINI_API_KEY="', [GEMINI]),
    ('export GAUSSMERIDIAN_GEMINI_API_KEY', [GEMINI]),  # compose takes the value from its own environment
    ('GAUSSMERIDIAN_GEMINI_API_KEY', [GEMINI]),
    ('  GAUSSMERIDIAN_GEMINI_API_KEY=', [GEMINI]),
    ('GAUSSMERIDIAN_GEMINI_API_KEY=""""', [GEMINI]),
    ('A="x" # GAUSSMERIDIAN_OPENROUTER_API_KEY=VAL', [OPENROUTER]),
    ('OTHER=1 # GAUSSMERIDIAN_GEMINI_API_KEY=abc', [GEMINI]),
    ('REM set GAUSSMERIDIAN_GEMINI_API_KEY=abc', [GEMINI]),
    ('XGAUSSMERIDIAN_GEMINI_API_KEY=abc', [GEMINI]),
    ('GAUSSMERIDIAN_GEMINI_API_KEY_OTHER=abc', [GEMINI]),
    # letters that fold to the name's letters (Kelvin sign, long s); compose matches names without case
    ('GAUSSMERIDIAN_GEMINI_API_\N{KELVIN SIGN}EY=VAL', [GEMINI]),
    ('GAUS\N{LATIN SMALL LETTER LONG S}MERIDIAN_GEMINI_API_KEY=VAL', [GEMINI]),
    ('GAUSSMERIDIAN_GEMINI_API_\N{KELVIN SIGN}EY=', []),
    # a character some reader skips, folds away or breaks a line at: neither name can be checked
    ('A=1\rGAUSSMERIDIAN_GEMINI_API_KEY=abc\rB=2', BOTH),
    ('GAUSSMERIDIAN_GEMINI_API_KEY=   \rVAL', BOTH),
    ('GAUSSMERIDIAN_GEMINI_API_KEY=\rVAL', BOTH),
    ('GAUSSMERIDIAN_GEMINI_API_KEY=\r"VAL"', BOTH),
    ('GAUSSMERIDIAN_GEMINI_API_KEY=\r#VAL', BOTH),
    ('GAUSSMERIDIAN_GEMINI_API_KEY=\rA=1', BOTH),
    ('GAUSSMERIDIAN_GEMINI_API_KEY\r=VAL', BOTH),
    ('export\rGAUSSMERIDIAN_GEMINI_API_KEY=VAL', BOTH),
    ('GAUSSMERIDIAN_GEMINI_API_KEY=\r\r\n', BOTH),
    ('GAUSSMERIDIAN_GEMINI_API_KEY="\r"', BOTH),
    ('GAUSSMERIDIAN_GEMINI_API_KEY=\x1c', BOTH),
    ('GAUSSMERIDIAN_GEMINI_API_KEY=\x1c\n', BOTH),
    ('GAUSSMERIDIAN_GEMINI_API_KEY=\x85', BOTH),
    ('GAUSSMERIDIAN_GEMINI_API_KEY=\xa0', BOTH),
    ('GAUSSMERIDIAN_GEMINI_API_KEY=\N{LINE SEPARATOR}', BOTH),
    ('GAUSSMERIDIAN_GEMINI_API_KEY=\N{ZERO WIDTH SPACE}', BOTH),
    ('GAUSSMERIDIAN_GEMINI_API_KEY=\N{ZERO WIDTH NO-BREAK SPACE}', BOTH),
    ('GAUSSMERIDIAN_GEMINI_API_KEY=\vVAL', BOTH),
    ('GAUSSMERIDIAN_GEMINI_API_KEY=\v\n', BOTH),
    ('GAUSSMERIDIAN_GEMINI_API_KEY=\f', BOTH),
    ('GAUSSMERIDIAN_GEMINI_API_KEY= \t\v\f\r \r\n', BOTH),
    ('GAUSSMERIDIAN_GEMINI_API\x01_KEY=VAL', BOTH),
    ('\ufeff\ufeffGAUSSMERIDIAN_GEMINI_API_KEY=', BOTH),
    ('A=1\n# a note\xa0\nB=2', BOTH),
    # not keys
    ('# GAUSSMERIDIAN_GEMINI_API_KEY=abc', []),
    ('   # export GAUSSMERIDIAN_GEMINI_API_KEY=abc', []),
    ('\t# GAUSSMERIDIAN_GEMINI_API_KEY=abc', []),
    ('GAUSSMERIDIAN_GEMINI_API_KEY=', []),
    ('GAUSSMERIDIAN_GEMINI_API_KEY=""', []),
    ("GAUSSMERIDIAN_GEMINI_API_KEY=''", []),
    ('gaussmeridian_gemini_api_key=', []),
    ('set GAUSSMERIDIAN_GEMINI_API_KEY=', []),
    ('export GAUSSMERIDIAN_GEMINI_API_KEY=', []),
    ('EXPORT\tGAUSSMERIDIAN_GEMINI_API_KEY=', []),
    ('GAUSSMERIDIAN_GEMINI_API_KEY =', []),
    ('GAUSSMERIDIAN_GEMINI_API_KEY\t=""', []),
    ('GAUSSMERIDIAN_GEMINI_API_KEY:', []),
    ('\ufeffGAUSSMERIDIAN_GEMINI_API_KEY=', []),
    ('GAUSSMERIDIAN_GEMINI_API_KEY=\r\n', []),
    ('GAUSSMERIDIAN_GEMINI_API_KEY=\r\nA=1', []),
    ('GAUSSMERIDIAN_GEMINI_API_KEY=\r\n\r\n', []),
    (f'{GEMINI}=\n{OPENROUTER}=""\n', []),
    ('A=""\nGAUSSMERIDIAN_GEMINI_API_KEY=""\n# GAUSSMERIDIAN_GEMINI_API_KEY=abc', []),
    ('A="x\n# GAUSSMERIDIAN_GEMINI_API_KEY=VAL\n"', []),
    ('GAUSSMER\N{LATIN SMALL LETTER DOTLESS I}DIAN_GEMINI_API_KEY=VAL', []),  # no reader folds it to I
    ('GAUSSMERIDIAN_DB_PASSWORD=abc', []),
    ('', []),
]


@pytest.mark.parametrize(('text', 'names'), TEXTS)
def test_paid_keys_are_found_by_name_only(text: str, names: list[str]) -> None:
    assert paid_keys_in(text) == names


HIDING = [
    *'\x01\x08\x0b\x0c\x0e\x1c\x1f\x7f\x80\x85\x9f\xa0\xad\u034f\u061c\u115f\u1680\u17b4\u180b\u180e\u180f',
    *'\u2000\u200a\u200b\u200f\u202a\u202e\u202f\u2028\u2029\u205f\u2060\u2064\u206f\u3000\u3164\ufe00\ufe0f',
    *'\ufeff\uffa0\ufff0\ufff9\U0001bca0\U0001d173\U000e0001\U000e0020\U000e0100\U000e0fff',
    '\r',
]


@pytest.mark.parametrize('char', HIDING, ids=[f'U+{ord(c):04X}' for c in HIDING])
def test_a_hiding_character_anywhere_means_neither_name_can_be_checked(char: str) -> None:
    assert paid_keys_in(f'A=1\n# a note{char}x\nB=2\n') == BOTH
    assert paid_keys_in(f'{GEMINI}{char}=\n') == BOTH


@pytest.mark.parametrize('char', ['\t', '\n', '\u00e9', '\u0131', '\u212a', '\u4e2d', '\U0001f600'])
def test_other_characters_do_not_stop_the_check(char: str) -> None:
    assert paid_keys_in(f'A=1{char}\n# a note {char}\n{GEMINI}=\n') == []


# The security reviewer's probes (fake values): every one is set for compose or for a local-up reader, except the
# dotless i, which no reader folds to I.
REVIEW_FIXTURES = {
    'a01-name-nbsp-eq': f'{GEMINI}\xa0=FAKEA01\n',
    'a02-name-nel-eq': f'{GEMINI}\x85=FAKEA02\n',
    'a03-export-nbsp': f'export\xa0{GEMINI}=FAKEA03\n',
    'a04-nbsp-emptyq-then-key': f'A=\xa0""{GEMINI}=FAKEA04\n',
    'a05-nel-emptysq-then-key': f"A=\x85''{GEMINI}=FAKEA05\n",
    'a06-name-nbsp-colon': f'{GEMINI}\xa0:FAKEA06\n',
    'a07-nbsp-q-then-key': f'A=\xa0"x"{GEMINI}=FAKEA07\n',
    'a08-export-nel': f'export\x85{GEMINI}=FAKEA08\n',
    'a09-space-nbsp-eq': f'{GEMINI} \xa0 =FAKEA09\n',
    'a10-self-nbsp-emptyq': f'{GEMINI}=\xa0""{GEMINI}=FAKEA10\n',
    'a11-lead-nbsp': f'\xa0{GEMINI}=FAKEA11\n',
    'a12-export-sp-nbsp': f'export \xa0{GEMINI}=FAKEA12\n',
    'a13-tab-emptyq-key': f'A=\t""{GEMINI}=FAKEA13\n',
    'a14-vt-emptyq-key': f'A=\v""{GEMINI}=FAKEA14\n',
    'a15-bs-quote': f'A="\\\\"{GEMINI}=FAKEA15"\n',
    'a16-crlf-emptyq-cr-key': f'A=""\r{GEMINI}=FAKEA16\n',
    'a17-sq-esc': f"A='x\\'{GEMINI}=FAKEA17\n",
    'a18-dq-multi-then': f'A="l1\nl2"{GEMINI}=FAKEA18\n',
    'a19-nbsp-before-colon-sp': f'{GEMINI}\xa0 :FAKEA19\n',
    'a20-inherit': f'{GEMINI}\n',
    'b01-kelvin': 'GAUSSMERIDIAN_GEMINI_API_\N{KELVIN SIGN}EY=FAKEB01\n',
    'b02-longs': 'GAUS\N{LATIN SMALL LETTER LONG S}MERIDIAN_GEMINI_API_KEY=FAKEB02\n',
    'b03-dotless-i': 'GAUSSMER\N{LATIN SMALL LETTER DOTLESS I}DIAN_GEMINI_API_KEY=FAKEB03\n',
    'b04-openrouter-nbsp': f'{OPENROUTER}\xa0= FAKEB04\n',
    'b05-nel-q-then': f'A=\x85"y"{OPENROUTER}=FAKEB05\n',
    'b06-tab-nbsp-eq': f'{GEMINI}\t\xa0\t=FAKEB06\n',
    'c01-or-space_g-nbsp': f'GAUSSMERIDIAN_DB_USERNAME=fake\n{OPENROUTER}= \n{GEMINI}\xa0=FAKEC01\n',
    'c02-or-vt_g-nbsp': f'{OPENROUTER}=\v\n{GEMINI}\xa0=FAKEC02\n',
    'c03-or-softhyphen-in-quote_g-nbsp': f'A="\n{OPENROUTER}\xad=x\n"\n{GEMINI}\xa0=FAKEC03\n',
    'c04-or-tab_g-nbsp-emptyq': f'{OPENROUTER}=\t\nA=\xa0""{GEMINI}=FAKEC04\n',
    '01-cr-after-eq': f'{GEMINI}=\rFAKEVAL01\n',
    '02-spaces-cr': f'{GEMINI}=   \rFAKEVAL02\n',
    '03-cr-quoted': f'{GEMINI}=\r"FAKEVAL03"\n',
    '04-same-line': f'A="" {GEMINI}=FAKEVAL04\n',
    '05-same-line-self': f'{GEMINI}=""{GEMINI}=FAKEVAL05\n',
    '06-cr-before-eq': f'{GEMINI}\r=FAKEVAL06\n',
    '07-cr-hash': f'{GEMINI}=\r#FAKEVAL07\n',
    '08-after-multiline': f'A="x\nline" {OPENROUTER}=FAKEVAL08\n',
    '09-export-cr': f'export\r{GEMINI}=FAKEVAL09\n',
    '10-nbsp-only': f'{GEMINI}=\xa0\n',
    '11-emspace-only': f'{GEMINI}=\u2003\n',
    '12-zwsp-only': f'{GEMINI}=\u200b\n',
    '13-vt': f'{GEMINI}=\vFAKEVAL13\n',
    '14-vt-only': f'{GEMINI}=\v\n',
    '15-fs-only': f'{GEMINI}=\x1c\n',
    '16-nel-only': f'{GEMINI}=\x85\n',
    '17-ff-only': f'{GEMINI}=\f\n',
    '18-last-empty': f'{GEMINI}=FAKEVAL18\n{GEMINI}=\n',
    '19-first-empty': f'{GEMINI}=\n{GEMINI}=FAKEVAL19\n',
    '20-lower': f'{GEMINI.lower()}=FAKEVAL20\n',
    '21-u2028': f'{GEMINI}=\u2028FAKEVAL21\n',
    '22-fs-val': f'{GEMINI}=\x1cFAKEVAL22\n',
    '23-quoted-cr': f'{GEMINI}="\r"\n',
    '24-bom-in-value': f'{GEMINI}=\ufeffFAKEVAL24\n',
    '25-bom': f'\ufeff{GEMINI}=FAKEVAL25\n',
    '26-interp': f'{GEMINI}=${{UNSET_THING:-FAKEVAL26}}\n',
    '27-ref': f'X=FAKEVAL27\n{GEMINI}=$X\n',
}
NOT_SET_FOR_ANY_READER = {'b03-dotless-i'}


@pytest.mark.parametrize('fixture', sorted(REVIEW_FIXTURES))
def test_every_reviewer_probe_is_refused(tmp_path: Path, fixture: str) -> None:
    environ, folder = runtime_with(tmp_path)
    (folder / 'meridian-engine.env').write_bytes(REVIEW_FIXTURES[fixture].encode('utf-8'))
    message = refusal(environ)
    if fixture in NOT_SET_FOR_ANY_READER:
        assert message is None
        return
    assert message is not None
    named = OPENROUTER if OPENROUTER in REVIEW_FIXTURES[fixture] else GEMINI
    assert named in message
    assert 'FAKE' not in message


def test_a_file_with_two_byte_order_marks_cannot_be_checked(tmp_path: Path) -> None:
    environ, folder = runtime_with(tmp_path)
    (folder / 'meridian-engine.env').write_bytes(b'\xef\xbb\xbf\xef\xbb\xbf' + f'{GEMINI}=\n'.encode())
    message = refusal(environ)
    assert message is not None
    assert GEMINI in message
    (folder / 'meridian-engine.env').write_bytes(b'\xef\xbb\xbf' + f'{GEMINI}=\n'.encode())
    assert refusal(environ) is None


# -- the guard against compose itself ------------------------------------------------------------------------------

# The engine's deploy/compose.yaml hands the keys over the same way
COMPOSE_FILE = (
    'services:\n  probe:\n    image: busybox\n    environment:\n'
    f'      G: ${{{GEMINI}:-}}\n      O: ${{{OPENROUTER}:-}}\n'
)
# What docker needs to find its compose plugin. No other variable of this process reaches compose, so a bare
# `NAME` line cannot pick up a value from here.
DOCKER_ENV = ('PATH', 'SYSTEMROOT', 'PROGRAMFILES', 'PROGRAMDATA', 'HOME', 'USERPROFILE', 'TEMP', 'TMP')
Resolver = Callable[[int, str], set[str] | None]


def docker_env() -> dict[str, str]:
    return {name: value for name, value in os.environ.items() if name.upper() in DOCKER_ENV}


@pytest.fixture(scope='module')
def compose_resolves(tmp_path_factory: pytest.TempPathFactory) -> Resolver:
    """The paid names `docker compose config` resolves to a non-empty value for an env file text, or None when compose
    rejects the file. It only parses: nothing is pulled, built or started. Only names leave it, never a value."""
    if shutil.which('docker') is None:
        pytest.skip('docker is not on PATH')
    try:
        version = subprocess.run(['docker', 'compose', 'version'], env=docker_env(), capture_output=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        pytest.skip('docker compose did not answer')
    if version.returncode != 0:
        pytest.skip('docker compose is not available')
    folder = tmp_path_factory.mktemp('compose')
    (folder / 'compose.yaml').write_text(COMPOSE_FILE, encoding='utf-8')

    def resolves(index: int, text: str) -> set[str] | None:
        env_file = folder / f'{index}.env'
        env_file.write_bytes(text.encode('utf-8'))
        command = ['docker', 'compose', '-p', 'meridian-guard-probe', '--env-file', str(env_file)]
        command += ['-f', str(folder / 'compose.yaml'), 'config', '--format', 'json']
        done = subprocess.run(command, env=docker_env(), cwd=folder, capture_output=True, timeout=120)
        if done.returncode != 0:
            return None  # compose refuses the file, so it would hand the engine nothing
        environment: dict[str, str | None] = json.loads(done.stdout)['services']['probe'].get('environment') or {}
        return {name for key, name in (('G', GEMINI), ('O', OPENROUTER)) if environment.get(key)}

    return resolves


def compose_texts() -> dict[str, str]:
    texts = {**REVIEW_FIXTURES, **{f'texts[{i}]': text for i, (text, _) in enumerate(TEXTS)}}
    for char in HIDING:
        texts[f'U+{ord(char):04X} after the name'] = f'{GEMINI}{char}=FAKE\n'
        texts[f'U+{ord(char):04X} before the name'] = f'{char}{OPENROUTER}=FAKE\n'
        texts[f'U+{ord(char):04X} as the value'] = f'{GEMINI}={char}\n'
    return texts


def test_whenever_compose_resolves_a_paid_key_the_guard_refuses_it(compose_resolves: Resolver, tmp_path: Path) -> None:
    texts = compose_texts()
    with ThreadPoolExecutor(8) as pool:
        resolved = dict(zip(texts, pool.map(compose_resolves, range(len(texts)), texts.values()), strict=True))
    missed: list[str] = []
    for index, (label, names) in enumerate(resolved.items()):
        if names:
            environ, folder = runtime_with(tmp_path / str(index))
            (folder / 'meridian-engine.env').write_bytes(texts[label].encode('utf-8'))
            message = refusal(environ) or ''
            missed += [f'{label}: {name}' for name in sorted(names) if name not in message]
    assert missed == []
    # not vacuous: compose read these files, so a docker that resolves nothing cannot pass
    assert resolved['20-lower'] == {GEMINI}
    assert resolved['08-after-multiline'] == {OPENROUTER}
    assert resolved['b03-dotless-i'] == set()
    assert sum(names is not None for names in resolved.values()) > len(texts) // 2


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


def test_a_missing_env_file_is_a_refusal_that_says_it_is_missing(tmp_path: Path) -> None:
    folder = tmp_path / 'Gaussian' / 'MeridianEngine'
    folder.mkdir(parents=True)
    message = refusal({'LOCALAPPDATA': str(tmp_path)})
    assert message is not None
    assert 'missing' in message
    assert 'UTF-8' not in message
    # the file the engine would read is the override when there is one
    (folder / 'meridian-engine.env').write_text('A=1\n', encoding='utf-8')
    message = refusal({'LOCALAPPDATA': str(tmp_path), 'GAUSSMERIDIAN_ENV_FILE': str(folder / 'nope.env')})
    assert message is not None
    assert 'missing' in message


def test_an_unreadable_env_file_is_a_refusal(tmp_path: Path) -> None:
    folder = tmp_path / 'Gaussian' / 'MeridianEngine'
    folder.mkdir(parents=True)
    (folder / 'meridian-engine.env').mkdir()  # a directory cannot be read as text
    message = refusal({'LOCALAPPDATA': str(tmp_path)})
    assert message is not None
    assert 'cannot be read' in message


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


def test_the_canonical_env_file_is_checked_when_both_overrides_point_at_a_clean_decoy(tmp_path: Path) -> None:
    environ, folder = runtime_with(tmp_path, f'{GEMINI}={SECRET}')
    (folder / 'sub').mkdir()
    (folder / 'sub' / 'clean.env').write_text('A=1\n', encoding='utf-8')
    message = refusal(
        {
            **environ,
            'MERIDIAN_RUNTIME_DIR': str(folder / 'sub'),
            'GAUSSMERIDIAN_ENV_FILE': str(folder / 'sub' / 'clean.env'),
        }
    )
    assert message is not None
    assert GEMINI in message
    assert SECRET not in message


def test_the_canonical_env_file_is_checked_when_only_the_env_file_points_elsewhere(tmp_path: Path) -> None:
    environ, folder = runtime_with(tmp_path, f'{OPENROUTER}={SECRET}')
    (folder / 'clean.env').write_text('A=1\n', encoding='utf-8')
    message = refusal({**environ, 'GAUSSMERIDIAN_ENV_FILE': str(folder / 'clean.env')})
    assert message is not None
    assert OPENROUTER in message


def test_an_env_file_named_by_two_spellings_is_read_once_and_a_canonical_one_that_is_absent_is_fine(
    tmp_path: Path,
) -> None:
    environ, folder = runtime_with(tmp_path)
    spelled = str(folder / 'sub' / '..' / 'meridian-engine.env')
    assert refusal({**environ, 'GAUSSMERIDIAN_ENV_FILE': spelled}) is None
    (folder / 'meridian-engine.env').unlink()
    (folder / 'clean.env').write_text('A=1\n', encoding='utf-8')
    assert refusal({**environ, 'GAUSSMERIDIAN_ENV_FILE': str(folder / 'clean.env')}) is None


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


@pytest.mark.parametrize('name', ['http_proxy', 'HTTP_PROXY', 'https_proxy', 'Https_Proxy', 'ALL_PROXY', 'all_proxy'])
def test_a_proxy_in_the_environment_is_refused_by_name_only(tmp_path: Path, name: str) -> None:
    environ, _ = runtime_with(tmp_path)
    message = refusal({**environ, name: f'http://user:{SECRET}@proxy.example:3128'})
    assert message is not None
    assert 'proxy' in message
    assert SECRET not in message
    assert 'proxy.example' not in message


def test_a_proxy_variable_that_is_empty_or_only_a_bypass_list_is_not_a_proxy(tmp_path: Path) -> None:
    environ, _ = runtime_with(tmp_path)
    assert refusal({**environ, 'HTTP_PROXY': '', 'all_proxy': ''}) is None
    assert refusal({**environ, 'NO_PROXY': '*', 'no_proxy': '127.0.0.1'}) is None


@pytest.mark.parametrize('scheme', ['http', 'https', 'all'])
def test_a_system_proxy_is_refused_by_scheme_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scheme: str) -> None:
    environ, _ = runtime_with(tmp_path)
    monkeypatch.setattr(urllib.request, 'getproxies', lambda: {scheme: f'http://u:{SECRET}@sys.example:8080'})
    message = refusal(environ)
    assert message is not None
    assert scheme in message
    assert SECRET not in message
    assert 'sys.example' not in message


def test_system_proxy_entries_for_other_schemes_do_not_matter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    environ, _ = runtime_with(tmp_path)
    monkeypatch.setattr(urllib.request, 'getproxies', lambda: {'ftp': 'http://sys.example:21', 'no': '*'})
    assert refusal(environ) is None


@pytest.mark.parametrize(
    'process', [{}, {'no': 'localhost'}, {'ftp': 'http://sys.example:21'}], ids=['none', 'no_proxy', 'ftp_proxy']
)
@pytest.mark.parametrize('scheme', ['http', 'https', 'all'])
def test_a_registry_proxy_is_refused_whatever_the_process_environment_says(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, process: dict[str, str], scheme: str
) -> None:
    # on Windows getproxies() reads the registry only when no *_proxy variable is set, so no_proxy alone hid it
    environ, _ = runtime_with(tmp_path)
    monkeypatch.setattr(urllib.request, 'getproxies', lambda: process)
    registry = {scheme: f'http://u:{SECRET}@reg.example:8080'}
    monkeypatch.setattr(urllib.request, 'getproxies_registry', lambda: registry, raising=False)
    message = refusal(environ)
    assert message is not None
    assert scheme in message
    assert SECRET not in message
    assert 'reg.example' not in message


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
        ContractEngine('t::node')
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
    engine = ContractEngine('t::node')
    assert engine.requested_by == 't::node'
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
NEEDS_REAL = 'needs the real local engine'
# the fixture reached by name at run time, which no collection-time look at fixture names can see
UNMARKED_DYNAMIC = (
    MARKED_TESTS + '\n\ndef test_unmarked_dynamic_user(request):\n    request.getfixturevalue("engine")\n'
)
MARKED_DYNAMIC = """\
import pytest


@pytest.mark.fake_only
def test_fake_by_name(request):
    assert request.getfixturevalue("engine").is_fake
"""
FAKE_ONLY_THAT_MUST_NOT_RUN = """\
import pytest


def test_pure():
    pass


@pytest.mark.fake_only
def test_fake_that_would_touch_the_network():
    raise AssertionError("ran")
"""


def pytest_on(
    tmp_path: Path, tests: str, contract: bool, *args: str, addopts: str = ''
) -> subprocess.CompletedProcess[str]:
    """Our conftest, as a plugin, over a sample file. In contract mode the only test that asks for the engine
    fixture is deselected before it runs, so no engine object is built and nothing is contacted."""
    (tmp_path / 'test_sample.py').write_text(tests, encoding='utf-8')
    (tmp_path / 'pytest.ini').write_text('[pytest]\naddopts = -q -rs -p no:cacheprovider\n', encoding='utf-8')
    env = {'PYTHONPATH': str(ROOT), 'SYSTEMROOT': os.environ.get('SYSTEMROOT', ''), 'PATH': os.environ.get('PATH', '')}
    if contract:
        env['MERIDIAN_CONTRACT'] = '1'
    if addopts:
        env['PYTEST_ADDOPTS'] = addopts
    command = [sys.executable, '-m', 'pytest', '-p', 'tests.conftest', '-c', str(tmp_path / 'pytest.ini'), *args]
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
    assert '2 passed, 1 skipped' in done.stdout


def test_against_the_real_engine_fake_only_tests_are_deselected(tmp_path: Path) -> None:
    done = pytest_on(tmp_path, MARKED_TESTS, contract=True)
    assert done.returncode == 0
    assert '2 passed, 1 deselected' in done.stdout
    assert 'skipped' not in done.stdout


@pytest.mark.parametrize(
    ('args', 'addopts'),
    [(('-p', 'no:skipping'), ''), ((), '-p no:skipping'), (('-m', ''), ''), ((), '-m "" -p no:skipping')],
)
def test_no_option_makes_a_fake_only_test_run_against_the_real_engine(
    tmp_path: Path, args: tuple[str, ...], addopts: str
) -> None:
    done = pytest_on(tmp_path, FAKE_ONLY_THAT_MUST_NOT_RUN, True, *args, addopts=addopts)
    assert done.returncode == 0, done.stdout
    assert 'ran' not in done.stdout
    assert '1 passed, 1 deselected' in done.stdout


@pytest.mark.parametrize('contract', [False, True])
def test_the_engine_fixture_reached_by_name_in_an_unmarked_test_fails_it(tmp_path: Path, contract: bool) -> None:
    done = pytest_on(tmp_path, UNMARKED_DYNAMIC, contract)
    assert done.returncode == 1, done.stdout
    assert 'test_unmarked_dynamic_user uses the engine without @pytest.mark.fake_only' in done.stdout
    assert '1 failed' in done.stdout


def test_the_engine_fixture_reached_by_name_in_a_marked_test_works(tmp_path: Path) -> None:
    done = pytest_on(tmp_path, MARKED_DYNAMIC, contract=False)
    assert done.returncode == 0, done.stdout
    assert '1 passed' in done.stdout
