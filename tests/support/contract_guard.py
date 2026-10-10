"""The contract guard: a contract run talks to the real local engine and must never spend money.

Everything here reports names only. No value of an environment variable or env file ever reaches a message.
"""

from __future__ import annotations

import ipaddress
import os
import re
import subprocess
import urllib.request
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

PAID_KEYS = ('GAUSSMERIDIAN_GEMINI_API_KEY', 'GAUSSMERIDIAN_OPENROUTER_API_KEY')
KEY_FILES = ('local-api-key.txt', 'local-integrator-key.txt')
ENV_FILE = 'meridian-engine.env'
DEFAULT_URLS = {'MERIDIAN_PUBLIC_URL': 'http://127.0.0.1:8180', 'MERIDIAN_ADMIN_URL': 'http://127.0.0.1:8181'}
PROXY_SCHEMES = ('http', 'https', 'all')

# What a line may hold. Anything else is skipped, folded or split at differently by one of the readers (compose takes
# NBSP and NEL as blank and matches a dotted capital I to I; the culture-sensitive StartsWith of local-common.ps1 skips
# controls, marks and some letters; a lone CR ends a line for pwsh but not for compose or bash), so it is refused.
_PRINTABLE = re.compile(r'[\t\x20-\x7e]*')
_LOWER = [name.lower() for name in PAID_KEYS]
# The only line with a paid name that leaves it unset: `[export |set ]NAME =` (or `:`), then nothing or `""`. Two
# single quotes are a value: both local-up readers strip double quotes only.
_EMPTY = re.compile(rf'(?:(?:export|set)[ \t]+)?(?:{"|".join(_LOWER)})[ \t]*[=:](?:"")?')


def paid_keys_in(text: str) -> list[str]:
    """Names of the paid provider keys an env file's text may set. Never returns values. Not a parser but an
    allowlist: a text with any character outside printable ASCII and tab (after one leading byte order mark; a lone
    CR included) cannot be checked and reports both names. Otherwise a line that names a paid key, in any letter case,
    sets it unless the line is exactly an empty assignment, so a whitespace-only value and `''` are set. A comment line
    is skipped unless it holds a quote, which may close a quoted value."""
    lines = re.split(r'\r\n|\n', text.removeprefix('\N{BYTE ORDER MARK}'))
    if not all(_PRINTABLE.fullmatch(line) for line in lines):
        return list(PAID_KEYS)
    names: list[str] = []
    for line in lines:
        lowered = line.lower()
        comment = lowered.lstrip(' \t').startswith('#') and not any(quote in line for quote in '"\'`')
        if comment or _EMPTY.fullmatch(lowered):
            continue
        names += [name for name, key in zip(PAID_KEYS, _LOWER, strict=True) if key in lowered and name not in names]
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
    Windows registry, plus `registry` for any other proxy server entry there (socks, ftp, ...). The registry is always
    read, because getproxies() skips it once any *_proxy variable (even no_proxy) is set. Names only; a proxy would
    carry the suite's traffic and local keys off-machine."""
    registry: Callable[[], Mapping[str, str]] | None = getattr(urllib.request, 'getproxies_registry', None)
    entries: Mapping[str, str] = registry() if registry else {}
    systems = [urllib.request.getproxies(), entries]
    schemes = [
        scheme
        for scheme in PROXY_SCHEMES
        if any(system.get(scheme) for system in systems)
        or any(v and k.lower() == f'{scheme}_proxy' for k, v in environ.items())
    ]
    if any(v for k, v in entries.items() if k not in PROXY_SCHEMES):
        schemes.append('registry')
    return schemes


def _paid_keys_in_file(path: Path) -> list[str] | None:
    """Names in an env file, or None when it cannot be read as UTF-8 text (so it cannot be checked)."""
    try:
        data = path.read_bytes()
        if b'\x00' in data:  # UTF-16 and binary files hide names from a text scan
            return None
        return paid_keys_in(data.decode('utf-8'))  # paid_keys_in drops one byte order mark and refuses a second
    except (OSError, UnicodeDecodeError):
        return None


DEFAULT_PROJECT = 'meridian-engine'
_PROJECT_NAME = re.compile(r'[a-z0-9][a-z0-9_-]*')
_CONTAINER_ID = re.compile(r'[0-9a-f]{12,64}')

# The only environment entries the proof may compare, each by exact equality inside the docker template, so the output
# is a token and never a value. The two paid keys are compared with the public dummy of deploy/compose.simulator.yaml
# and with nothing else.
SIM_ENV = (
    ('gemini_base', 'GAUSSMERIDIAN_GEMINI_BASE_URL=http://provider-sim:8184/v1beta'),
    ('openrouter_base', 'GAUSSMERIDIAN_OPENROUTER_BASE_URL=http://provider-sim:8184/api/v1'),
    ('sim_flag', 'GAUSSMERIDIAN_PROVIDER_SIMULATOR=true'),
    ('gemini_dummy', 'GAUSSMERIDIAN_GEMINI_API_KEY=sim-local-key'),
    ('openrouter_dummy', 'GAUSSMERIDIAN_OPENROUTER_API_KEY=sim-local-key'),
)
SIM_TEMPLATE = (
    '{{range .Config.Env}}' + ''.join(f'{{{{if eq . "{e}"}}}}{t} {{{{end}}}}' for t, e in SIM_ENV) + '{{end}}'
)
ENV_FILE_TEMPLATE = '{{index .Config.Labels "com.docker.compose.project.environment_file"}}'

DockerRunner = Callable[[Sequence[str]], str]
"""Runs `docker` with a fixed argv (no shell) and returns its stdout; any failure raises."""


def run_docker(args: Sequence[str]) -> str:
    """The real docker. Its output is read by the guard and never printed; nothing here is a network call."""
    done = subprocess.run(
        ['docker', *args],
        capture_output=True,
        text=True,
        encoding='utf-8',
        timeout=20,
        check=True,
        stdin=subprocess.DEVNULL,
    )
    return done.stdout


def _lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def _url_port(value: str) -> str | None:
    try:
        parts = urlsplit(value)
        return str(parts.port or (443 if parts.scheme == 'https' else 80))
    except ValueError:
        return None


def simulator_proof(environ: Mapping[str, str], runtime: Path, docker: DockerRunner | None = None) -> str | None:
    """Why the engine on this machine is not proven to be in simulator mode and started from this runtime's env file,
    or None when it is. S1 provider-sim runs; S2 exactly one engine owns the public and admin ports on 127.0.0.1;
    S3 its provider settings are the simulator's (exact comparisons inside the docker template, tokens out); S4 it was
    started from the env file of this runtime (the compose label, or the runbook's marker when the label is absent).
    Any error counts as a refusal. Names only: no value of the engine's environment ever reaches this process's output,
    and no command but ps, port and a formatted inspect is run (never `logs`, never a raw `inspect`)."""
    ask = docker or run_docker
    project = environ.get('GAUSSMERIDIAN_LOCAL_PROJECT') or DEFAULT_PROJECT
    if not _PROJECT_NAME.fullmatch(project):
        return 'GAUSSMERIDIAN_LOCAL_PROJECT is not a plain project name.'
    unasked = 'the simulator cannot be proven: docker cannot be asked.'

    def running(service: str) -> list[str] | None:
        try:
            return _lines(
                ask(
                    [
                        'ps',
                        '-q',
                        '--filter',
                        f'label=com.docker.compose.project={project}',
                        '--filter',
                        f'label=com.docker.compose.service={service}',
                        '--filter',
                        'status=running',
                    ]
                )
            )
        except Exception:
            return None

    def tell(*args: str) -> str | None:
        try:
            return ask(args)
        except Exception:
            return None

    # [S1]
    providers = running('provider-sim')
    if providers is None:
        return unasked
    if len(providers) != 1 or not _CONTAINER_ID.fullmatch(providers[0]):
        return 'provider-sim is not running exactly once, so this is not the simulator.'

    # [S2]
    engines = running('engine')
    if engines is None:
        return unasked
    if len(engines) != 1 or not _CONTAINER_ID.fullmatch(engines[0]):
        return 'there is not exactly one engine container running.'
    engine = engines[0]
    wanted = {
        '8180/tcp': _url_port(environ.get('MERIDIAN_PUBLIC_URL', DEFAULT_URLS['MERIDIAN_PUBLIC_URL'])),
        '8181/tcp': _url_port(environ.get('MERIDIAN_ADMIN_URL', DEFAULT_URLS['MERIDIAN_ADMIN_URL'])),
    }
    if None in wanted.values():
        return 'MERIDIAN_PUBLIC_URL or MERIDIAN_ADMIN_URL cannot be read, so the engine ports cannot be compared.'
    for port, host_port in wanted.items():
        bound = tell('port', engine, port)
        if bound is None:
            return unasked
        if _lines(bound) != [f'127.0.0.1:{host_port}']:
            return 'the engine does not own the public and admin ports on 127.0.0.1 (ports differ from the URLs).'

    # [S3]
    printed = tell('inspect', engine, '--format', SIM_TEMPLATE)
    if printed is None:
        return unasked
    tokens = printed.split()
    missing = [token for token, _ in SIM_ENV if token not in tokens]
    if missing:
        return f'the engine is not set up for the simulator (missing: {", ".join(missing)}).'
    if len(tokens) != len(SIM_ENV):
        return 'the engine is not set up for the simulator.'

    # [S4]
    label = tell('inspect', engine, '--format', ENV_FILE_TEMPLATE)
    if label is None:
        return unasked
    mine = Path(environ.get('GAUSSMERIDIAN_ENV_FILE') or runtime / ENV_FILE)
    stated = label.strip()
    refused = 'the engine was not started from the env file of this runtime.'
    if stated and stated != '<no value>':
        return None if _real(stated) == _real(mine) else refused
    # No label: the runbook's marker, written in the runtime folder after local-up, holds this engine's id
    try:
        marker = (runtime / f'.sim-started-{engine}').read_text(encoding='utf-8').strip()
    except (OSError, UnicodeDecodeError):
        return refused
    ok = _CONTAINER_ID.fullmatch(marker) and (marker.startswith(engine) or engine.startswith(marker))
    return None if ok else refused


def refusal(environ: Mapping[str, str], docker: DockerRunner | None = None) -> str | None:
    """Why a contract run must not start, or None when it may. Names only; never a value, path or URL. `docker` is
    injectable: tests never call the real one."""
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

    # [4.5] The engine must be proven to be the simulator, started from this runtime's env file. After the URLs, so the
    #       ports it compares are loopback ones.
    unproven = simulator_proof(environ, runtime, docker)
    if unproven is not None:
        return unproven

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
