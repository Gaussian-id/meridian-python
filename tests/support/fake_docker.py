"""A stand-in for the docker command line, for the guard's simulator proof. No test may call the real docker: this
answers from a description of the engine stack and records every argv it was asked."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from urllib.parse import urlsplit

from .contract_guard import DEFAULT_PROJECT, ENV_FILE, ENV_FILE_TEMPLATE, SIM_ENV, SIM_TEMPLATE, runtime_dir

PROVIDER_ID = 'a' * 12
ENGINE_ID = 'b' * 64
SIM_TOKENS = [token for token, _ in SIM_ENV]


class FakeDocker:
    """By default a healthy simulator stack: one provider-sim, one engine owning 8180 and 8181 on 127.0.0.1, all five
    simulator tokens, and the compose label naming `env_file`. Each field can be changed to break one thing."""

    def __init__(self, env_file: str | Path | None = None, *, project: str = DEFAULT_PROJECT) -> None:
        self.project = project
        self.providers: list[str] = [PROVIDER_ID]
        self.engines: list[str] = [ENGINE_ID]
        self.ports: dict[str, str] = {'8180/tcp': '127.0.0.1:8180', '8181/tcp': '127.0.0.1:8181'}
        self.tokens: list[str] = list(SIM_TOKENS)
        self.label: str = str(env_file) if env_file is not None else ''
        self.fail: str | None = None  # a command word (ps, port, inspect) that raises
        self.calls: list[list[str]] = []

    @classmethod
    def for_environ(cls, environ: Mapping[str, str]) -> FakeDocker:
        """A healthy stack started from the env file this environment's runtime folder holds."""
        runtime = runtime_dir(environ)
        mine = environ.get('GAUSSMERIDIAN_ENV_FILE') or (runtime / ENV_FILE if runtime else '')
        fake = cls(mine, project=environ.get('GAUSSMERIDIAN_LOCAL_PROJECT') or DEFAULT_PROJECT)
        for port, name in (('8180/tcp', 'MERIDIAN_PUBLIC_URL'), ('8181/tcp', 'MERIDIAN_ADMIN_URL')):
            try:
                host_port = urlsplit(environ.get(name, '')).port
            except ValueError:
                host_port = None
            if host_port:  # the stack publishes the port the URL names
                fake.ports[port] = f'127.0.0.1:{host_port}'
        return fake

    def __call__(self, args: Sequence[str]) -> str:
        argv = list(args)
        self.calls.append(argv)
        if self.fail is not None and argv[:1] == [self.fail]:
            raise OSError('docker is not available')
        if argv[:2] == ['ps', '-q']:
            return self._ps(argv)
        if argv[:1] == ['port'] and len(argv) == 3:
            bound = self.ports.get(argv[2], '')
            return bound + '\n' if bound else ''
        if argv[:1] == ['inspect'] and len(argv) == 4 and argv[2] == '--format':
            if argv[3] == SIM_TEMPLATE:
                return ''.join(f'{token} ' for token in self.tokens)
            if argv[3] == ENV_FILE_TEMPLATE:
                return self.label + '\n'
        raise AssertionError(f'a command the guard may not run: {argv[:2]}')

    def _ps(self, argv: list[str]) -> str:
        service = argv[5].removeprefix('label=com.docker.compose.service=')
        assert argv[2:5] == ['--filter', f'label=com.docker.compose.project={self.project}', '--filter']
        assert argv[6:] == ['--filter', 'status=running']
        ids = {'provider-sim': self.providers, 'engine': self.engines}[service]
        return ''.join(f'{one}\n' for one in ids)
