"""Contract run against the real local engine. It refuses on a paid provider key, a path outside the runtime folder or
a non-loopback URL; it never prints a value. The real run is deferred by the owner: this script only refuses, or
launches the suite when the guard allows it.

    uv run python scripts/contract.py
"""

import os
import subprocess
import sys
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tests.support.contract_guard import DEFAULT_URLS, refusal, runtime_dir  # noqa: E402

COMMAND = ['uv', 'run', 'pytest', '-m', 'not fake_only']
LOOPBACK_HOSTS = '127.0.0.1,localhost,::1'


def ready_endpoint(public_url: str) -> str:
    return public_url.rstrip('/') + '/ready'


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *_: Any, **__: Any) -> None:
        return None


def engine_ready(public_url: str) -> bool:
    """A plain 200 from /ready. No proxy from the environment and no redirect, so the answer is the engine's own."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect)
    try:
        with opener.open(ready_endpoint(public_url), timeout=5) as response:
            return response.status == 200
    except (OSError, ValueError, urllib.error.URLError):
        return False


def main(
    environ: Mapping[str, str] | None = None,
    *,
    ready: Callable[[str], bool] = engine_ready,
    runner: Callable[..., int] = subprocess.call,
) -> int:
    environ = os.environ if environ is None else environ

    # [1] The guard decides first: nothing is contacted or launched on a refusal
    reason = refusal(environ)
    if reason is not None:
        print(f'Contract run refused: {reason}', file=sys.stderr)
        return 1

    # [2] The engine must be ready
    public_url = environ.get('MERIDIAN_PUBLIC_URL', DEFAULT_URLS['MERIDIAN_PUBLIC_URL'])
    if not ready(public_url):
        print('Contract run refused: the local engine is not ready (or did not answer in 5 s).', file=sys.stderr)
        return 1

    # [3] The same suite, in contract mode, against the engine; no proxy may sit between the suite and it
    runtime = environ.get('MERIDIAN_RUNTIME_DIR') or str(runtime_dir(environ))
    child = {
        key: value for key, value in environ.items() if key.upper() not in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY')
    }
    child.update(MERIDIAN_CONTRACT='1', MERIDIAN_RUNTIME_DIR=runtime, NO_PROXY=LOOPBACK_HOSTS, no_proxy=LOOPBACK_HOSTS)
    return int(runner(COMMAND, env=child, cwd=str(ROOT)))


if __name__ == '__main__':
    sys.exit(main())
