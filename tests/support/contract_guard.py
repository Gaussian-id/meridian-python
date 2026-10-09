"""The contract guard: a contract run talks to the real local engine and must never spend money.

Everything here reports names only. No value of an environment variable or env file ever reaches a message.
"""

from __future__ import annotations

import ipaddress
import os
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

PAID_KEYS = ('GAUSSMERIDIAN_GEMINI_API_KEY', 'GAUSSMERIDIAN_OPENROUTER_API_KEY')
KEY_FILES = ('local-api-key.txt', 'local-integrator-key.txt')
ENV_FILE = 'meridian-engine.env'
DEFAULT_URLS = {'MERIDIAN_PUBLIC_URL': 'http://127.0.0.1:8180', 'MERIDIAN_ADMIN_URL': 'http://127.0.0.1:8181'}

# NAME=value, export NAME=value, set NAME=value, set "NAME=value", NAME: value; the name is the whole word
_ASSIGNMENT = re.compile(r'^\s*(?:(?:export|set)\s+)?(")?(' + '|'.join(PAID_KEYS) + r')\s*[=:]\s*(.*)$', re.IGNORECASE)


def _quoted_inner(rest: str, quote: str) -> str | None:
    """The text before the closing quote, or None when the quote never closes on this line."""
    i = 0
    while i < len(rest):
        if rest[i] == '\\':
            i += 2
        elif rest[i] == quote:
            return rest[:i]
        else:
            i += 1
    return None


def _has_value(raw: str, wrapped: bool) -> bool:
    """Whether what follows `NAME=` is a value. Where parsers disagree (`#abc`, a space inside quotes) it is one."""
    value = raw.strip()
    if wrapped and value.endswith('"'):
        value = value[:-1].rstrip()
    if not value:
        return False
    if value[0] in '"\'':
        inner = _quoted_inner(value[1:], value[0])
        return inner is None or inner != ''  # no closing quote: the value runs over several lines
    return True


def paid_keys_in(text: str) -> list[str]:
    """Names of the paid provider keys that have a value in an env file's text. Never returns values."""
    names: list[str] = []
    for line in re.split(r'\r\n|\r|\n', text.lstrip('﻿')):
        match = _ASSIGNMENT.match(line)
        if match and _has_value(match.group(3), match.group(1) is not None):
            name = match.group(2).upper()
            if name not in names:
                names.append(name)
    return names


def paid_keys_in_environ(environ: Mapping[str, str]) -> list[str]:
    """Names of the paid provider keys that are non-empty in a process environment, in any letter case."""
    names: list[str] = []
    for name, value in environ.items():
        if name.upper() in PAID_KEYS and value != '' and name.upper() not in names:
            names.append(name.upper())
    return names


def _loopback_host(host: str) -> bool:
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
    engine_env = environ.get('GAUSSMERIDIAN_ENV_FILE') or runtime / ENV_FILE
    if not is_inside(runtime, engine_env):
        return 'GAUSSMERIDIAN_ENV_FILE resolves outside the local runtime folder.'
    for name in KEY_FILES:
        key_file = Path(folder) / name
        if os.path.lexists(key_file) and not is_inside(runtime, key_file):
            return f'{name} resolves outside the local runtime folder.'

    # [3] No paid provider key may be set: not in the process, not in the file the engine reads, not in the
    #     runtime dir's own env file (a clean decoy next to the real one must not hide it)
    names = paid_keys_in_environ(environ)
    env_files = [Path(engine_env)]
    own = Path(folder) / ENV_FILE
    if os.path.lexists(own) and _real(own) != _real(engine_env):
        env_files.append(own)
    for env_file in env_files:
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
