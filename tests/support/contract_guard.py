"""The contract guard: a contract run talks to the real local engine and must never spend money.

Everything here reports names only. No value of an environment variable or env file ever reaches a message.
"""

from __future__ import annotations

import ipaddress
import os
import re
import unicodedata
import urllib.request
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

PAID_KEYS = ('GAUSSMERIDIAN_GEMINI_API_KEY', 'GAUSSMERIDIAN_OPENROUTER_API_KEY')
KEY_FILES = ('local-api-key.txt', 'local-integrator-key.txt')
ENV_FILE = 'meridian-engine.env'
DEFAULT_URLS = {'MERIDIAN_PUBLIC_URL': 'http://127.0.0.1:8180', 'MERIDIAN_ADMIN_URL': 'http://127.0.0.1:8181'}
PROXY_SCHEMES = ('http', 'https', 'all')

# Characters that one reader skips, ignores or breaks a line at while another does not (compose treats NBSP and NEL as
# blank; a culture-sensitive StartsWith ignores soft hyphens, joiners and C0 controls; a lone CR ends a line for
# local-up but not for compose): control characters other than tab, LF and the CR of a CRLF, and every default-ignorable
# code point. Any other non-ASCII space or format character (Zs, Zl, Zp, Cf) is refused too.
_HIDING = re.compile(
    r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xa0\xad\U0000034f\U0000061c\U0000115f\U00001160\U000017b4\U000017b5'
    r'\U0000180b-\U0000180f\U0000200b-\U0000200f\U00002028-\U0000202e\U00002060-\U0000206f\U00003164'
    r'\U0000fe00-\U0000fe0f\U0000feff\U0000ffa0\U0000fff0-\U0000fff8\U0001bca0-\U0001bca3'
    r'\U0001d173-\U0001d17a\U000e0000-\U000e0fff]|\r(?!\n)'
)
_FOLDED = [name.casefold() for name in PAID_KEYS]
# The only line with a paid name that leaves it unset: `[export |set ]NAME =` (or `:`), then nothing, `""` or `''`
_EMPTY = re.compile(rf'(?:(?:export|set)[ \t]+)?(?:{"|".join(_FOLDED)})[ \t]*[=:](?:""|\'\')?')


def _hides(text: str) -> bool:
    return _HIDING.search(text) is not None or any(
        c > '\x7f' and unicodedata.category(c) in ('Zs', 'Zl', 'Zp', 'Cf') for c in text
    )


def paid_keys_in(text: str) -> list[str]:
    """Names of the paid provider keys an env file's text may set. Never returns values. Not a parser but an
    allowlist: a line that names a paid key (without case, Kelvin sign and long s folded) sets it unless the line is
    exactly an empty assignment, so a whitespace-only value is set. A text with a hiding character cannot be checked
    and reports both names. A comment line is skipped unless it holds a quote, which may close a quoted value."""
    text = text.removeprefix('\N{BYTE ORDER MARK}')
    if _hides(text):
        return list(PAID_KEYS)
    names: list[str] = []
    for line in re.split(r'\r\n|\r|\n', text):
        folded = line.casefold()
        comment = folded.lstrip(' \t').startswith('#') and not any(quote in line for quote in '"\'`')
        if comment or _EMPTY.fullmatch(folded):
            continue
        names += [name for name, key in zip(PAID_KEYS, _FOLDED, strict=True) if key in folded and name not in names]
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
    Windows registry. The registry is always read, because getproxies() skips it once any *_proxy variable (even
    no_proxy) is set. Names only; a proxy would carry the suite's traffic and local keys off-machine."""
    registry: Callable[[], Mapping[str, str]] | None = getattr(urllib.request, 'getproxies_registry', None)
    systems = [urllib.request.getproxies(), registry() if registry else {}]
    return [
        scheme
        for scheme in PROXY_SCHEMES
        if any(system.get(scheme) for system in systems)
        or any(v and k.lower() == f'{scheme}_proxy' for k, v in environ.items())
    ]


def _paid_keys_in_file(path: Path) -> list[str] | None:
    """Names in an env file, or None when it cannot be read as UTF-8 text (so it cannot be checked)."""
    try:
        data = path.read_bytes()
        if b'\x00' in data:  # UTF-16 and binary files hide names from a text scan
            return None
        return paid_keys_in(data.decode('utf-8'))  # paid_keys_in drops one byte order mark; a second one hides
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
