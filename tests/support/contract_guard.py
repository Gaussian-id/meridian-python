"""The contract guard: a contract run talks to the real local engine and must never spend money.

Everything here reports names only. No value of an environment variable or env file ever reaches a message.
"""

from __future__ import annotations

import heapq
import ipaddress
import os
import re
import urllib.request
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

PAID_KEYS = ('GAUSSMERIDIAN_GEMINI_API_KEY', 'GAUSSMERIDIAN_OPENROUTER_API_KEY')
KEY_FILES = ('local-api-key.txt', 'local-integrator-key.txt')
ENV_FILE = 'meridian-engine.env'
DEFAULT_URLS = {'MERIDIAN_PUBLIC_URL': 'http://127.0.0.1:8180', 'MERIDIAN_ADMIN_URL': 'http://127.0.0.1:8181'}
PROXY_SCHEMES = ('http', 'https', 'all')

# What compose skips after `=`. Every other character, \x1c and \xa0 included, is a value (Python's \s is wider).
BLANK = ' \t\v\f\r'
_B = r'[ \t\v\f\r]'
# [export|set] ["]NAME [=:] , the name being the whole word; or any other statement `anything =:`
_PAID_HEAD = re.compile(rf'(?:(?:export|set){_B}+)?(")?({"|".join(PAID_KEYS)}){_B}*[=:]{_B}*', re.IGNORECASE)
_ANY_HEAD = re.compile(rf'[^=:\n]*[=:]{_B}*')


def _line_end(text: str, start: int) -> int:
    end = text.find('\n', start)
    return len(text) if end < 0 else end


def _closings(text: str, start: int, quote: str) -> set[int]:
    """Where a quote opened at `start` may close. Parsers disagree on escapes, so every reading is followed: the next
    quote; the next one no backslash precedes (compose); the next one a backslash-pair escape does not hide."""
    ends: set[int] = set()
    first = text.find(quote, start + 1)
    if first >= 0:
        ends.add(first)
    i = first
    while i >= 0 and text[i - 1] == '\\':
        i = text.find(quote, i + 1)
    if i >= 0:
        ends.add(i)
    i = start + 1
    while i < len(text) and text[i] != quote:
        i += 2 if text[i] == '\\' else 1
    if i < len(text):
        ends.add(i)
    return ends


def _scan(text: str) -> list[str]:
    """Paid keys that get a value in `text`, read as compose reads it: statements `NAME=value`, a value running to
    the end of the line (a line ends at \\n), a quoted value ending at its closing quote, and after that quote
    another statement can follow on the same line. A quote may also be read as ending its line. Where a statement
    may start in more than one way, all of them are followed."""
    names: list[str] = []
    seen: set[int] = set()
    todo = [0]
    while todo:
        start = heapq.heappop(todo)
        if start in seen:
            continue
        seen.add(start)
        pos = start
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if pos >= len(text):
            continue
        line_end = _line_end(text, pos)
        paid = _PAID_HEAD.match(text, pos)
        head = paid or _ANY_HEAD.match(text, pos)
        heapq.heappush(todo, line_end)
        if text[pos] == '#' or head is None:  # a comment, or a line that is not a statement
            continue
        value = head.end()
        if paid and paid.group(1):  # set "NAME=value"
            rest = text[value:line_end].strip(BLANK)
            rest = rest[:-1].rstrip(BLANK) if rest.endswith('"') else rest
            has_value = rest not in ('', '""', "''")
        elif value >= line_end:
            has_value = False
        elif text[value] in '"\'':
            has_value = text[value + 1 : value + 2] != text[value]  # `""` is empty; an unclosed quote is not
            for end in _closings(text, value, text[value]):
                heapq.heappush(todo, end + 1)  # the statement after the closing quote
        else:
            has_value = True  # even `#abc`: whether it is a comment depends on the parser
        if paid and has_value and paid.group(2).upper() not in names:
            names.append(paid.group(2).upper())
    return names


def paid_keys_in(text: str) -> list[str]:
    """Names of the paid provider keys that have a value in an env file's text. Never returns values. The text is
    read as compose reads it (lines end at \\n) and as local-up.ps1 does (at \\r\\n, \\r and \\n); a name set under
    either reading is set."""
    text = text.lstrip('﻿')
    names: list[str] = []
    for reading in (text, re.sub(r'\r\n|\r', '\n', text)):
        names += [name for name in _scan(reading) if name not in names]
    return names


def paid_keys_in_environ(environ: Mapping[str, str]) -> list[str]:
    """Names of the paid provider keys that are non-empty in a process environment, in any letter case."""
    names: list[str] = []
    for name, value in environ.items():
        if name.upper() in PAID_KEYS and value != '' and name.upper() not in names:
            names.append(name.upper())
    return names


def _loopback_host(host: str) -> bool:
    # `localhost` is accepted because the engine's own docs use it, but it is resolved by the system resolver (hosts
    # file, DNS, a VPN client), which the guard does not control. Literal 127.0.0.1 or ::1 is preferred.
    if host == 'localhost':
        return True
    try:
        return ipaddress.ip_address(host).is_loopback  # IPv4-mapped IPv6 is judged by its IPv4 part
    except ValueError:  # also every decimal, hex, octal and short IPv4 spelling
        return False


def is_loopback_url(value: str) -> bool:
    """An http(s) URL whose host, as written, is localhost, an address in 127.0.0.0/8 or ::1. No userinfo, no
    whitespace or control characters, no `%`, no alternative IP spellings."""
    if not isinstance(value, str) or any(c <= ' ' or c == '%' for c in value):  # pyright: ignore[reportUnnecessaryIsInstance]
        return False
    try:
        parts = urlsplit(value)
        host = parts.hostname
        _ = parts.port  # a bad port raises
    except ValueError:
        return False
    if parts.scheme not in ('http', 'https') or '@' in parts.netloc or host is None:
        return False
    return _loopback_host(host)


def _real(path: str | os.PathLike[str]) -> str:
    return os.path.normcase(os.path.realpath(path))


def is_inside(parent: str | os.PathLike[str], child: str | os.PathLike[str]) -> bool:
    """Whether child, after symlinks, junctions and `..`, is strictly below parent. Anything unresolvable is not."""
    try:
        if '\x00' in os.fspath(parent) or '\x00' in os.fspath(child):
            return False
        p, c = _real(parent), _real(child)
        return c != p and os.path.commonpath([p, c]) == p
    except (OSError, ValueError, RuntimeError):  # another drive, a symlink loop, an unrepresentable path
        return False


def runtime_dir(environ: Mapping[str, str]) -> Path | None:
    """The folder the engine's local scripts keep their secrets in, found the way local-common does."""
    local = environ.get('LOCALAPPDATA')
    if local:
        return Path(local) / 'Gaussian' / 'MeridianEngine'
    state = environ.get('XDG_STATE_HOME')
    home = environ.get('HOME')
    if state:
        return Path(state) / 'gaussian' / 'meridian-engine'
    if home:
        return Path(home) / '.local' / 'state' / 'gaussian' / 'meridian-engine'
    return None


def proxy_schemes(environ: Mapping[str, str]) -> list[str]:
    """Schemes (http, https, all) for which a proxy is configured: in this environment, in the process's, or in the
    system (the registry on Windows). Names only; a proxy would carry the suite's traffic and local keys off-machine."""
    system = urllib.request.getproxies()
    return [
        scheme
        for scheme in PROXY_SCHEMES
        if system.get(scheme) or any(v and k.lower() == f'{scheme}_proxy' for k, v in environ.items())
    ]


def _paid_keys_in_file(path: Path) -> list[str] | None:
    """Names in an env file, or None when it cannot be read as UTF-8 text (so it cannot be checked)."""
    try:
        data = path.read_bytes()
        if b'\x00' in data:  # UTF-16 and binary files hide names from a text scan
            return None
        return paid_keys_in(data.decode('utf-8-sig'))
    except (OSError, UnicodeDecodeError):
        return None


def refusal(environ: Mapping[str, str]) -> str | None:
    """Why a contract run must not start, or None when it may. Names only; never a value, path or URL."""
    # [1] The runtime folder the engine's scripts use is the only place the guard trusts
    runtime = runtime_dir(environ)
    if runtime is None:
        return 'the local runtime folder cannot be located (set LOCALAPPDATA, XDG_STATE_HOME or HOME).'

    # [2] Every path the run reads from must resolve inside it
    folder = environ.get('MERIDIAN_RUNTIME_DIR') or runtime
    if _real(folder) != _real(runtime) and not is_inside(runtime, folder):
        return 'MERIDIAN_RUNTIME_DIR resolves outside the local runtime folder.'
    canonical = runtime / ENV_FILE
    engine_env = Path(environ.get('GAUSSMERIDIAN_ENV_FILE') or canonical)
    if not is_inside(runtime, engine_env):
        return 'GAUSSMERIDIAN_ENV_FILE resolves outside the local runtime folder.'
    for name in KEY_FILES:
        key_file = Path(folder) / name
        if os.path.lexists(key_file) and not is_inside(runtime, key_file):
            return f'{name} resolves outside the local runtime folder.'

    # [3] No paid provider key may be set: not in the process, and not in any env file the engine or the suite could
    #     read: the canonical one, the runtime dir's own, and the override (a clean decoy must not hide the others).
    #     Only the file the engine would read has to exist.
    names = paid_keys_in_environ(environ)
    required = _real(engine_env)
    checked: set[str] = set()
    for env_file in (engine_env, canonical, Path(folder) / ENV_FILE):
        if _real(env_file) in checked:
            continue
        checked.add(_real(env_file))
        if not os.path.lexists(env_file):
            if _real(env_file) == required:
                return 'the engine env file is missing, so it cannot be checked for provider keys.'
            continue
        found = _paid_keys_in_file(env_file)
        if found is None:
            return 'the engine env file cannot be read as UTF-8 text, so it cannot be checked for provider keys.'
        names += [name for name in found if name not in names]
    if names:
        return f'{", ".join(names)} is set, so the engine would call a paid provider.'

    # [4] Only a loopback engine
    for name in ('MERIDIAN_PUBLIC_URL', 'MERIDIAN_ADMIN_URL', 'MERIDIAN_URL'):
        value = environ.get(name, DEFAULT_URLS.get(name))
        if value is not None and not is_loopback_url(value):
            return f'{name} is not a loopback URL (localhost, 127.0.0.0/8 or ::1).'

    # [5] No proxy between the suite and the engine
    proxies = proxy_schemes(environ)
    if proxies:
        return f'a proxy is configured ({", ".join(proxies)}), so traffic and the local keys could leave this machine.'
    return None


def unmarked_engine_tests(items: Iterable[Any]) -> list[str]:
    """Collected tests that use the `engine` fixture without saying whether they need the in-memory engine
    (`fake_only`) or the real one (`contract`); such a test would fail, or reach the network, in contract mode."""
    return [
        item.nodeid
        for item in items
        if 'engine' in getattr(item, 'fixturenames', ())
        and not any(item.get_closest_marker(name) for name in ('fake_only', 'contract'))
    ]
