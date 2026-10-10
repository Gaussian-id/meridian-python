import functools
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unicodedata
import urllib.request
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from tests.support import contract_guard
from tests.support.contract_guard import (
    SIM_ENV,
    SIM_TEMPLATE,
    DockerRunner,
    is_inside,
    is_loopback_url,
    paid_keys_in,
    paid_keys_in_environ,
    runtime_dir,
    simulator_proof,
    unmarked_engine_tests,
)
from tests.support.contract_guard import refusal as guard_refusal
from tests.support.engine import ContractEngine
from tests.support.fake_docker import ENGINE_ID, FakeDocker

pytestmark = pytest.mark.fake_only


def refusal(environ: dict[str, str], docker: DockerRunner | None = None) -> str | None:
    """The guard's judgement over a healthy simulator stack (unless the test brings its own docker): the real docker is
    never asked."""
    return guard_refusal(environ, docker or FakeDocker.for_environ(environ))


ROOT = Path(__file__).resolve().parents[1]

GEMINI = 'GAUSSMERIDIAN_GEMINI_API_KEY'
OPENROUTER = 'GAUSSMERIDIAN_OPENROUTER_API_KEY'
SECRET = 'v' + 'q' * 24  # a stand-in value, built at run time; no message may ever contain it
REAL_REGISTRY = getattr(urllib.request, 'getproxies_registry', None)  # before any test replaces it


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
    # both local-up readers strip only double quotes, so two single quotes are a two-character value
    ("GAUSSMERIDIAN_GEMINI_API_KEY=''", [GEMINI]),
    ("export GAUSSMERIDIAN_GEMINI_API_KEY=''", [GEMINI]),
    ("GAUSSMERIDIAN_GEMINI_API_KEY\t:''", [GEMINI]),
    (f"{GEMINI}=\n{OPENROUTER}=''\n", [OPENROUTER]),
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
    # anything outside printable ASCII and tab, anywhere in the file: neither name can be checked. Some reader folds
    # it to a name's letter (Kelvin sign, long s, dotted capital I), skips it (marks, tatweel), or breaks a line at it.
    ('GAUSSMERIDIAN_GEMINI_API_\N{KELVIN SIGN}EY=VAL', BOTH),
    ('GAUS\N{LATIN SMALL LETTER LONG S}MERIDIAN_GEMINI_API_KEY=VAL', BOTH),
    ('GAUSSMERIDIAN_GEMINI_API_\N{KELVIN SIGN}EY=', BOTH),
    ('GAUSSMER\N{LATIN CAPITAL LETTER I WITH DOT ABOVE}DIAN_GEMINI_API_KEY=VAL', BOTH),
    ('GAUSS\N{ARABIC TATWEEL}MERIDIAN_GEMINI_API_KEY=VAL', BOTH),
    ('GAUSSMERIDIAN_GEMINI_API_KEY\N{HEBREW ACCENT ETNAHTA}=VAL', BOTH),
    ('A=caf\N{LATIN SMALL LETTER E WITH ACUTE}\n', BOTH),
    ('# \N{LATIN SMALL LETTER E WITH ACUTE}\n', BOTH),
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
    ("A='x\n# GAUSSMERIDIAN_GEMINI_API_KEY=VAL\n'", []),
    ('A=1\tB=~ !"#$%&()*+,-./:;<>?@[\\]^_`{|}\n', []),
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
    # what the culture-sensitive StartsWith of local-common.ps1 skips (ICU), or what folds to a name's letter
    *'\x00\u0300\u0591\u0640\u07fa\u180a\u1cd3\ufe73\u0903\u20dd\u0130\u0131\u017f\u212a\u00e9\u4e2d\U0001f600',
]


@pytest.mark.parametrize('char', HIDING, ids=[f'U+{ord(c):04X}' for c in HIDING])
def test_a_character_outside_printable_ascii_anywhere_means_neither_name_can_be_checked(char: str) -> None:
    assert paid_keys_in(f'A=1\n# a note{char}x\nB=2\n') == BOTH
    assert paid_keys_in(f'{GEMINI}{char}=\n') == BOTH


PRINTABLE = '\t' + ''.join(map(chr, range(0x20, 0x7F)))


@pytest.mark.parametrize('char', PRINTABLE, ids=[f'U+{ord(c):04X}' for c in PRINTABLE])
def test_printable_ascii_and_tab_do_not_stop_the_check(char: str) -> None:
    assert paid_keys_in(f'A=1{char}\n# a note {char}\n{GEMINI}=\n') == []
    assert paid_keys_in(f'{GEMINI}={char}\n') == [GEMINI]


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
# The third round (fake values): two single quotes are a value for both local-up readers; Windows compose matches a
# dotted capital I to I; the culture-sensitive StartsWith skips marks and a few letters (U+0640, U+07FA, U+FE73) and
# punctuation (U+180A, U+1CD3)
SQ = "''"
DOTTED = '\N{LATIN CAPITAL LETTER I WITH DOT ABOVE}'
DB = 'GAUSSMERIDIAN_DB_USERNAME=fakeuser\nGAUSSMERIDIAN_DB_PASSWORD=fakepw\n'
ROUND_3 = {
    'r3-sq-G': f'{GEMINI}={SQ}\n',
    'r3-sq-O': f'{OPENROUTER}={SQ}\n',
    'r3-sq-both': f'{GEMINI}={SQ}\n{OPENROUTER}={SQ}\n',
    'r3-sq-kelvin': f'GAUSSMERIDIAN_GEMINI_API_\N{KELVIN SIGN}EY={SQ}\n',
    'r3-sq-export': f'export {GEMINI}={SQ}\n',
    'r3-sq-set': f'set {GEMINI}={SQ}\n',
    'r3-sq-colon': f'{GEMINI}:{SQ}\n',
    'r3-sq-tab': f'{GEMINI}\t={SQ}\n',
    'r3-sq-lower': f'{GEMINI.lower()}={SQ}\n',
    'r3-sq-then-empty': f'{GEMINI}={SQ}\n{GEMINI}=\n',
    'r3-empty-then-sq': f'{GEMINI}=\n{GEMINI}={SQ}\n',
    'r3-sq-comment': f'{GEMINI}={SQ} # c\n',
    'r3-chain-sq-dotted': f'{DB}{OPENROUTER}={SQ}\nGAUSSMER{DOTTED}DIAN_GEMINI_API_KEY=FAKER3A\n',
    'r3-chain-icu-dotted': f'{DB}GAUSS\N{ARABIC TATWEEL}MERIDIAN_OPENROUTER_API_KEY=\n'
    f'GAUSSMER{DOTTED}DIAN_GEMINI_API_KEY=FAKER3B\n',
    'r3-chain-dotted-lower': f'{DB}gaussmer{DOTTED}dian_gemini_api_key=FAKER3C\n',
    'r3-chain-all-dotted': f'{DB}{OPENROUTER}={SQ}\n{GEMINI.replace("I", DOTTED)}=FAKER3D\n',
    'r3-export-dotted': f'{OPENROUTER}={SQ}\nexport GAUSSMER{DOTTED}DIAN_GEMINI_API_KEY=FAKER3E\n',
    'r3-dotted-colon': f'{OPENROUTER}={SQ}\nGAUSSMER{DOTTED}DIAN_GEMINI_API_KEY: FAKER3F\n',
    'r3-tatweel-val': 'GAUSS\N{ARABIC TATWEEL}MERIDIAN_GEMINI_API_KEY=FAKER3G\n',
    'r3-fe73-val': 'GAUSS\N{ARABIC TAIL FRAGMENT}MERIDIAN_GEMINI_API_KEY=FAKER3H\n',
    'r3-07fa-val': 'GAUSS\N{NKO LAJANYALAN}MERIDIAN_GEMINI_API_KEY=FAKER3I\n',
    'r3-180a-val': 'GAUSS\N{MONGOLIAN NIRUGU}MERIDIAN_GEMINI_API_KEY=FAKER3J\n',
    'r3-1cd3-val': 'GAUSS\N{VEDIC SIGN NIHSHVASA}MERIDIAN_GEMINI_API_KEY=FAKER3K\n',
    'r3-0591-val': 'GAUSS\N{HEBREW ACCENT ETNAHTA}MERIDIAN_GEMINI_API_KEY=FAKER3L\n',
    'r3-mark-in-multiline': 'A="\nGAUSS\N{HEBREW ACCENT ETNAHTA}MERIDIAN_OPENROUTER_API_KEY=\n"\n',
    'r3-interp-other': f'X=${{{GEMINI}:-FAKER3M}}\n',
    'r3-self-ref': f'{GEMINI}=${{{GEMINI}}}\n',
    # a `#` line inside a quoted value: compose reads it as part of the value (or rejects the unclosed one), and
    # neither local-up reader reads it
    'r3-sq-multiline-hash': f"A='x\n# {GEMINI}=FAKER3N\n'\n",
    'r3-dq-multiline-hash': f'A="x\n# {GEMINI}=FAKER3O\n"\n',
    'r3-unclosed-then-hash': f'A="x\n# {GEMINI}=FAKER3P\n',
    'r3-backtick-hash': f'A=`x\n# {GEMINI}=FAKER3Q\n`\n',
}
for _name in BOTH:
    for _at, _letter in enumerate(_name):
        if _letter == 'I':
            ROUND_3[f'r3-dotted-{_name[14]}-{_at}'] = f'{_name[:_at]}{DOTTED}{_name[_at + 1 :]}=FAKEDOT\n'
REVIEW_FIXTURES.update(ROUND_3)
NOT_SET_FOR_ANY_READER = {'r3-sq-multiline-hash', 'r3-dq-multiline-hash', 'r3-unclosed-then-hash', 'r3-backtick-hash'}


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


# -- the guard against the readers themselves ----------------------------------------------------------------------

SUPPORT = Path(__file__).resolve().parent / 'support'
# The engine's deploy/compose.yaml hands the keys over the same way
COMPOSE_FILE = (
    'services:\n  probe:\n    image: busybox\n    environment:\n'
    f'      G: ${{{GEMINI}:-}}\n      O: ${{{OPENROUTER}:-}}\n'
)
# What docker, pwsh and bash need to start. No other variable of this process reaches them, so a bare `NAME` line
# cannot pick up a value from here.
TOOL_ENV = ('PATH', 'SYSTEMROOT', 'PROGRAMFILES', 'PROGRAMDATA', 'HOME', 'USERPROFILE', 'TEMP', 'TMP', 'LANG', 'LC_ALL')
# The paid names a reader sets to a non-empty value, per env file text; None where it rejects the file
Reader = Callable[[dict[str, str]], dict[str, set[str] | None]]


def tool_env() -> dict[str, str]:
    return {name: value for name, value in os.environ.items() if name.upper() in TOOL_ENV}


@pytest.fixture(scope='module')
def compose_reader(tmp_path_factory: pytest.TempPathFactory) -> Reader:
    """`docker compose config`, which hands the engine its keys. It only parses: nothing is pulled, built or started.
    Only names leave it, never a value."""
    if shutil.which('docker') is None:
        pytest.skip('docker is not on PATH')
    try:
        version = subprocess.run(['docker', 'compose', 'version'], env=tool_env(), capture_output=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        pytest.skip('docker compose did not answer')
    if version.returncode != 0:
        pytest.skip('docker compose is not available')
    folder = tmp_path_factory.mktemp('compose')
    (folder / 'compose.yaml').write_text(COMPOSE_FILE, encoding='utf-8')

    def resolves(item: tuple[int, str]) -> set[str] | None:
        env_file = folder / f'{item[0]}.env'
        env_file.write_bytes(item[1].encode('utf-8'))
        command = ['docker', 'compose', '-p', 'meridian-guard-probe', '--env-file', str(env_file)]
        command += ['-f', str(folder / 'compose.yaml'), 'config', '--format', 'json']
        done = subprocess.run(command, env=tool_env(), cwd=folder, capture_output=True, timeout=120)
        if done.returncode != 0:
            return None  # compose refuses the file, so it would hand the engine nothing
        environment: dict[str, str | None] = json.loads(done.stdout)['services']['probe'].get('environment') or {}
        return {name for key, name in (('G', GEMINI), ('O', OPENROUTER)) if environment.get(key)}

    def read(texts: dict[str, str]) -> dict[str, set[str] | None]:
        with ThreadPoolExecutor(8) as pool:
            return dict(zip(texts, pool.map(resolves, enumerate(texts.values())), strict=True))

    return read


def script_reader(command: list[str], folder: Path) -> Reader:
    """A copied local-up reader, run over every text in eight parallel batches. Two flags per file come back."""

    def read(texts: dict[str, str]) -> dict[str, set[str] | None]:
        labels = list(texts)

        def run(start: int) -> str:
            batch = Path(tempfile.mkdtemp(dir=folder))
            for index in range(start, len(labels), 8):
                (batch / f'{index:06d}.env').write_bytes(texts[labels[index]].encode('utf-8'))
            done = subprocess.run([*command, batch.as_posix()], env=tool_env(), capture_output=True, timeout=600)
            assert done.returncode == 0, f'{Path(command[-1]).name} failed'  # not its output, which might hold a value
            return done.stdout.decode('ascii')

        found: dict[str, set[str] | None] = {}
        with ThreadPoolExecutor(8) as pool:
            for row in ''.join(pool.map(run, range(8))).splitlines():
                stem, gemini, openrouter = row.split('\t')
                flags = ((GEMINI, gemini), (OPENROUTER, openrouter))
                found[labels[int(stem)]] = {name for name, flag in flags if flag == '1'}
        return found

    return read


@pytest.fixture(scope='module')
def pwsh_reader(tmp_path_factory: pytest.TempPathFactory) -> Reader:
    """local-up.ps1's Get-EnvValue under pwsh, with its culture-sensitive StartsWith (ICU, the default culture)."""
    pwsh = shutil.which('pwsh')
    if pwsh is None:
        pytest.skip('pwsh is not on PATH')
    command = [pwsh, '-NoProfile', '-NonInteractive', '-File', str(SUPPORT / 'get_env_value.ps1')]
    return script_reader(command, tmp_path_factory.mktemp('pwsh'))


@pytest.fixture(scope='module')
def bash_reader(tmp_path_factory: pytest.TempPathFactory) -> Reader:
    """local-up.sh's env_value under bash."""
    bash = shutil.which('bash')
    if bash is None:
        pytest.skip('bash is not on PATH')
    if os.name == 'nt' and is_inside(os.environ.get('SYSTEMROOT', 'C:\\Windows'), bash):
        pytest.skip('the bash on PATH is the WSL launcher, which cannot read these Windows paths')
    return script_reader([bash, (SUPPORT / 'env_value.sh').as_posix()], tmp_path_factory.mktemp('bash'))


@pytest.fixture(params=['compose', 'pwsh', 'bash'])
def reader(request: pytest.FixtureRequest) -> tuple[str, Reader]:
    name: str = request.param
    return name, request.getfixturevalue(f'{name}_reader')


def some_of_each_category() -> list[str]:
    """About 24 code points from each category the ICU StartsWith skips most of: marks, format and control
    characters."""
    categories: dict[str, list[str]] = {}
    for point in range(0x80, 0x110000):
        category = unicodedata.category(chr(point))
        if category in ('Mn', 'Mc', 'Me', 'Cf', 'Cc'):
            categories.setdefault(category, []).append(chr(point))
    return [char for chars in categories.values() for char in chars[:: -(-len(chars) // 24)]]


@functools.cache
def differential_texts() -> dict[str, str]:
    """The reviewers' probes, the unit table, and suspicious characters around, inside and at the end of a name."""
    texts = {**REVIEW_FIXTURES, **{f'texts[{i}]': text for i, (text, _) in enumerate(TEXTS)}}
    for char in HIDING:
        texts[f'U+{ord(char):04X} after the name'] = f'{GEMINI}{char}=FAKE\n'
        texts[f'U+{ord(char):04X} before the name'] = f'{char}{OPENROUTER}=FAKE\n'
        texts[f'U+{ord(char):04X} as the value'] = f'{GEMINI}={char}\n'
    for char in [chr(point) for point in range(0x80) if point != 0x0A] + HIDING:
        texts[f'U+{ord(char):04X} inside the name'] = f'{GEMINI[:5]}{char}{GEMINI[5:]}=FAKE\n'
        texts[f'U+{ord(char):04X} ends the name'] = f'{OPENROUTER}{char}=FAKE\n'
    for char in some_of_each_category():
        texts[f'U+{ord(char):04X} inside the name'] = f'{GEMINI[:5]}{char}{GEMINI[5:]}=FAKE\n'
    return texts


@pytest.fixture(scope='module')
def guard_allows(tmp_path_factory: pytest.TempPathFactory) -> set[str]:
    """The texts the guard lets a contract run start with, as the runtime's env file."""
    allowed: set[str] = set()
    for index, (label, text) in enumerate(differential_texts().items()):
        environ, folder = runtime_with(tmp_path_factory.mktemp(str(index)))
        (folder / 'meridian-engine.env').write_bytes(text.encode('utf-8'))
        if refusal(environ) is None:
            allowed.add(label)
    return allowed


# What each reader must find, so one that reads nothing cannot pass. Compose takes '' as empty, matches a name without
# case and keeps the last line; both local-up readers keep '', match case and take the first line; pwsh splits lines at
# a lone CR, bash deletes it.
ANCHORS: dict[str, dict[str, set[str]]] = {
    'compose': {'r3-sq-G': set(), '20-lower': {GEMINI}, '08-after-multiline': {OPENROUTER}, '19-first-empty': {GEMINI}},
    'pwsh': {'r3-sq-G': {GEMINI}, '20-lower': set(), '19-first-empty': set(), '01-cr-after-eq': set()},
    'bash': {'r3-sq-G': {GEMINI}, '20-lower': set(), '19-first-empty': set(), '01-cr-after-eq': {GEMINI}},
}


def test_whenever_a_reader_sets_a_paid_key_the_guard_refuses(
    reader: tuple[str, Reader], guard_allows: set[str]
) -> None:
    name, read = reader
    texts = differential_texts()
    found = read(texts)
    assert found.keys() == texts.keys()
    missed = [f'{label}: {key}' for label in sorted(guard_allows) for key in sorted(found[label] or ())]
    assert missed == []
    assert {label: found[label] for label in ANCHORS[name]} == ANCHORS[name]
    assert sum(names is not None for names in found.values()) > len(texts) // 2


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


@pytest.mark.parametrize(
    'registry',
    [{'socks': 'socks://reg.example:1080'}, {'ftp': 'http://reg.example:21'}, {'gopher': 'reg.example:70'}],
    ids=['socks', 'ftp', 'other'],
)
def test_any_registry_proxy_entry_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, registry: dict[str, str]
) -> None:
    environ, _ = runtime_with(tmp_path)
    monkeypatch.setattr(urllib.request, 'getproxies_registry', lambda: registry, raising=False)
    message = refusal(environ)
    assert message is not None
    assert 'registry' in message
    assert 'reg.example' not in message
    monkeypatch.setattr(urllib.request, 'getproxies_registry', lambda: {'socks': ''}, raising=False)
    assert refusal(environ) is None


@pytest.mark.parametrize('server', ['socks=reg.example:1080', 'ftp=reg.example:21', 'reg.example:3128'])
def test_a_proxy_server_entry_the_standard_library_reads_from_the_registry_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, server: str
) -> None:
    if sys.platform != 'win32':
        pytest.skip('the Windows registry')
    import winreg

    class Key:
        def Close(self) -> None:
            pass

    def open_key(*_: object) -> Key:
        return Key()

    def query(_: object, name: str) -> tuple[object, int]:
        return {'ProxyEnable': (1, winreg.REG_DWORD), 'ProxyServer': (server, winreg.REG_SZ)}[name]

    environ, _ = runtime_with(tmp_path)
    monkeypatch.setattr(winreg, 'OpenKey', open_key)  # nothing of this machine's registry is read
    monkeypatch.setattr(winreg, 'QueryValueEx', query)
    monkeypatch.setattr(urllib.request, 'getproxies_registry', REAL_REGISTRY)
    message = refusal(environ)
    assert message is not None
    assert 'reg.example' not in message


def test_the_default_urls_are_loopback(tmp_path: Path) -> None:
    environ, _ = runtime_with(tmp_path)
    assert refusal(environ) is None
    assert refusal({**environ, 'MERIDIAN_PUBLIC_URL': ''}) is not None  # set but empty: not a loopback URL


# -- the simulator proof (S1 to S4) --------------------------------------------------------------------------------


def stack(tmp_path: Path, **environ: str) -> tuple[dict[str, str], Path, FakeDocker]:
    env, folder = runtime_with(tmp_path)
    env.update(environ)
    return env, folder, FakeDocker.for_environ(env)


def test_a_healthy_simulator_stack_is_proven_with_nothing_but_ps_port_and_formatted_inspect(tmp_path: Path) -> None:
    environ, folder, docker = stack(tmp_path)
    assert simulator_proof(environ, folder, docker) is None
    assert refusal(environ, docker) is None
    assert {call[0] for call in docker.calls} == {'ps', 'port', 'inspect'}
    for call in docker.calls:
        assert call[0] != 'inspect' or (call[2] == '--format' and len(call) == 4)  # never a raw inspect
        assert 'logs' not in call
    assert docker.calls[0][:2] == ['ps', '-q']


def test_the_real_docker_is_what_runs_when_none_is_injected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    environ, _, docker = stack(tmp_path)
    monkeypatch.setattr(contract_guard, 'run_docker', docker)
    assert guard_refusal(environ) is None
    assert docker.calls


def test_the_template_compares_only_the_allowlisted_entries_and_prints_tokens() -> None:
    assert [entry.split('=')[0] for _, entry in SIM_ENV] == [
        'GAUSSMERIDIAN_GEMINI_BASE_URL',
        'GAUSSMERIDIAN_OPENROUTER_BASE_URL',
        'GAUSSMERIDIAN_PROVIDER_SIMULATOR',
        'GAUSSMERIDIAN_GEMINI_API_KEY',
        'GAUSSMERIDIAN_OPENROUTER_API_KEY',
    ]
    assert [entry for _, entry in SIM_ENV if entry.split('=')[0].endswith('_API_KEY')] == [
        'GAUSSMERIDIAN_GEMINI_API_KEY=sim-local-key',
        'GAUSSMERIDIAN_OPENROUTER_API_KEY=sim-local-key',
    ]
    assert SIM_TEMPLATE.startswith('{{range .Config.Env}}')
    assert '{{if eq . "GAUSSMERIDIAN_PROVIDER_SIMULATOR=true"}}sim_flag {{end}}' in SIM_TEMPLATE


def test_the_project_follows_the_local_project_variable(tmp_path: Path) -> None:
    environ, folder, docker = stack(tmp_path, GAUSSMERIDIAN_LOCAL_PROJECT='meridian-sim')
    assert simulator_proof(environ, folder, docker) is None
    assert 'label=com.docker.compose.project=meridian-sim' in docker.calls[0]


@pytest.mark.parametrize('project', ['Meridian', 'a b', 'a,label=x', '-x', '../x'])
def test_a_project_that_is_not_a_plain_name_is_refused(tmp_path: Path, project: str) -> None:
    environ, folder, docker = stack(tmp_path, GAUSSMERIDIAN_LOCAL_PROJECT=project)
    assert simulator_proof(environ, folder, docker) == 'GAUSSMERIDIAN_LOCAL_PROJECT is not a plain project name.'
    assert docker.calls == []


BREAKS: dict[str, tuple[Callable[[FakeDocker], None], str]] = {
    'no simulator': (lambda d: d.providers.clear(), 'provider-sim is not running exactly once'),
    'two simulators': (lambda d: d.providers.append('c' * 12), 'provider-sim is not running exactly once'),
    'a simulator id that is no id': (lambda d: d.providers.__setitem__(0, 'not-an-id'), 'provider-sim is not running'),
    'no engine': (lambda d: d.engines.clear(), 'not exactly one engine container'),
    'two engines': (lambda d: d.engines.append('c' * 64), 'not exactly one engine container'),
    'an engine id that is no id': (lambda d: d.engines.__setitem__(0, 'ENGINE'), 'not exactly one engine container'),
    'the public port elsewhere': (
        lambda d: d.ports.__setitem__('8180/tcp', '0.0.0.0:8180'),
        'does not own the public and admin ports',
    ),
    'the admin port not published': (
        lambda d: d.ports.__delitem__('8181/tcp'),
        'does not own the public and admin ports',
    ),
    'the public port on another host port': (
        lambda d: d.ports.__setitem__('8180/tcp', '127.0.0.1:9180'),
        'does not own the public and admin ports',
    ),
    'two bindings': (
        lambda d: d.ports.__setitem__('8180/tcp', '127.0.0.1:8180\n[::]:8180'),
        'does not own the public and admin ports',
    ),
    'a real gemini base url': (lambda d: d.tokens.remove('gemini_base'), 'missing: gemini_base)'),
    'a real openrouter base url': (lambda d: d.tokens.remove('openrouter_base'), 'missing: openrouter_base)'),
    'no simulator flag': (lambda d: d.tokens.remove('sim_flag'), 'missing: sim_flag)'),
    'a real gemini key': (lambda d: d.tokens.remove('gemini_dummy'), 'missing: gemini_dummy)'),
    'a real openrouter key': (lambda d: d.tokens.remove('openrouter_dummy'), 'missing: openrouter_dummy)'),
    'no token at all': (lambda d: d.tokens.clear(), 'missing: gemini_base, openrouter_base, sim_flag'),
    'a token that is not one of ours': (lambda d: d.tokens.append('real_key'), 'not set up for the simulator'),
    'a label pointing elsewhere': (
        lambda d: setattr(d, 'label', str(Path(d.label).parent / 'other.env')),
        'not started from the env file of this runtime',
    ),
    'a label naming two files': (
        lambda d: setattr(d, 'label', d.label + ',' + d.label),
        'not started from the env file of this runtime',
    ),
    'no label and no marker': (lambda d: setattr(d, 'label', ''), 'not started from the env file of this runtime'),
}


@pytest.mark.parametrize('name', list(BREAKS))
def test_anything_short_of_a_proven_simulator_is_refused_by_name_only(tmp_path: Path, name: str) -> None:
    change, expected = BREAKS[name]
    environ, folder, docker = stack(tmp_path)
    change(docker)
    reason = simulator_proof(environ, folder, docker)
    assert reason is not None
    assert expected in reason
    assert refusal(environ, docker) == reason  # the whole judgement says the same
    assert 'sim-local-key' not in reason and str(tmp_path) not in reason


@pytest.mark.parametrize('command', ['ps', 'port', 'inspect'])
def test_docker_that_cannot_be_asked_is_a_refusal(tmp_path: Path, command: str) -> None:
    environ, folder, docker = stack(tmp_path)
    docker.fail = command
    assert simulator_proof(environ, folder, docker) == 'the simulator cannot be proven: docker cannot be asked.'


def test_a_missing_docker_program_is_a_refusal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    environ, _, _ = stack(tmp_path)

    def no_docker(args: object) -> str:
        raise FileNotFoundError('docker')

    monkeypatch.setattr(contract_guard, 'run_docker', no_docker)
    assert guard_refusal(environ) == 'the simulator cannot be proven: docker cannot be asked.'


def test_the_real_runner_runs_docker_with_a_fixed_argv_and_no_shell(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[Any, dict[str, Any]]] = []

    def fake_run(argv: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        seen.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, stdout='out', stderr='')

    monkeypatch.setattr(subprocess, 'run', fake_run)
    assert contract_guard.run_docker(['ps', '-q']) == 'out'
    [(argv, kwargs)] = seen
    assert argv == ['docker', 'ps', '-q']
    assert not kwargs.get('shell') and kwargs['check'] is True and kwargs['timeout'] > 0


def test_the_label_may_be_a_path_spelled_differently(tmp_path: Path) -> None:
    environ, folder, docker = stack(tmp_path)
    docker.label = str(folder / '.' / 'sub' / '..' / 'meridian-engine.env')
    assert simulator_proof(environ, folder, docker) is None


def test_the_env_file_override_is_the_file_the_label_must_name(tmp_path: Path) -> None:
    environ, folder, _ = stack(tmp_path)
    other = folder / 'other.env'
    other.write_text('GAUSSMERIDIAN_DB_PASSWORD=x\n', encoding='utf-8')
    environ['GAUSSMERIDIAN_ENV_FILE'] = str(other)
    assert simulator_proof(environ, folder, FakeDocker(folder / 'meridian-engine.env')) is not None
    assert simulator_proof(environ, folder, FakeDocker(other)) is None


def test_without_a_label_the_runbooks_marker_for_this_engine_proves_the_binding(tmp_path: Path) -> None:
    environ, folder, docker = stack(tmp_path)
    docker.label = ''
    (folder / f'.sim-started-{ENGINE_ID}').write_text(ENGINE_ID + '\n', encoding='utf-8')
    assert simulator_proof(environ, folder, docker) is None
    docker.label = '<no value>'
    assert simulator_proof(environ, folder, docker) is None


def test_a_marker_for_another_engine_proves_nothing(tmp_path: Path) -> None:
    environ, folder, docker = stack(tmp_path)
    docker.label = ''
    (folder / f'.sim-started-{ENGINE_ID}').write_text('d' * 64, encoding='utf-8')
    assert simulator_proof(environ, folder, docker) == 'the engine was not started from the env file of this runtime.'


def test_a_label_that_names_another_file_is_not_rescued_by_a_marker(tmp_path: Path) -> None:
    environ, folder, docker = stack(tmp_path)
    docker.label = str(tmp_path / 'elsewhere.env')
    (folder / f'.sim-started-{ENGINE_ID}').write_text(ENGINE_ID, encoding='utf-8')
    assert simulator_proof(environ, folder, docker) is not None


def test_a_real_mode_stack_is_refused_whatever_else_is_right(tmp_path: Path) -> None:
    # real providers: the gemini and openrouter base urls and both keys are not the simulator's, no flag
    environ, _, docker = stack(tmp_path)
    docker.tokens = []
    docker.providers = []
    assert refusal(environ, docker) is not None


def test_the_proof_comes_after_the_files_and_urls_and_before_the_proxy(tmp_path: Path) -> None:
    environ, _, docker = stack(tmp_path)
    docker.providers.clear()
    assert 'provider-sim' in str(refusal({**environ, 'HTTPS_PROXY': 'http://p.example:1'}, docker))
    assert 'is not a loopback URL' in str(refusal({**environ, 'MERIDIAN_ADMIN_URL': 'http://x.example'}, docker))
    assert GEMINI in str(refusal({**environ, GEMINI: SECRET}, docker))


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
    monkeypatch.setattr(contract_guard, 'run_docker', FakeDocker.for_environ(environ))
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
