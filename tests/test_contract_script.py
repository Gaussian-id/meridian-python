"""scripts/contract.py: only its refusals and the command it would run are tested. Nothing here starts a pytest
subprocess against an engine, and the only socket is a throwaway server on this machine."""

import importlib.util
import os
import subprocess
import sys
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import pytest

pytestmark = pytest.mark.fake_only

ROOT = Path(__file__).resolve().parents[1]
GEMINI = 'GAUSSMERIDIAN_GEMINI_API_KEY'
SECRET = 'v' + 'q' * 24  # built at run time; nothing the script prints may contain it


def load_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location('meridian_contract_script', ROOT / 'scripts' / 'contract.py')
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


script: Any = load_script()


class Calls:
    """A runner and a readiness probe that record what they were asked and launch nothing."""

    def __init__(self, ready: bool = True, code: int = 0) -> None:
        self.ready_urls: list[str] = []
        self.runs: list[tuple[list[str], dict[str, str], str]] = []
        self._ready = ready
        self._code = code

    def ready(self, url: str) -> bool:
        self.ready_urls.append(url)
        return self._ready

    def run(self, command: list[str], env: dict[str, str], cwd: str) -> int:
        self.runs.append((command, env, cwd))
        return self._code


def runtime_with(tmp_path: Path, text: str = 'GAUSSMERIDIAN_DB_PASSWORD=x\n') -> tuple[dict[str, str], Path]:
    folder = tmp_path / 'Gaussian' / 'MeridianEngine'
    folder.mkdir(parents=True)
    (folder / 'meridian-engine.env').write_text(text, encoding='utf-8')
    return {'LOCALAPPDATA': str(tmp_path), 'PATH': os.environ.get('PATH', '')}, folder


def run_main(environ: dict[str, str], calls: Calls) -> int:
    return int(script.main(environ, ready=calls.ready, runner=calls.run))


def test_the_command_it_would_run_is_the_suite_without_the_fake_only_tests() -> None:
    assert script.COMMAND == ['uv', 'run', 'pytest', '-m', 'not fake_only']


def test_a_paid_key_in_the_env_file_refuses_before_anything_runs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    environ, _ = runtime_with(tmp_path, f'export {GEMINI}="{SECRET}"\r\n')
    calls = Calls()
    assert run_main(environ, calls) == 1
    shown = capsys.readouterr()
    assert GEMINI in shown.err + shown.out
    assert SECRET not in shown.err + shown.out
    assert calls.ready_urls == []
    assert calls.runs == []


def test_a_paid_key_in_the_process_environment_refuses_before_anything_runs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    environ, _ = runtime_with(tmp_path)
    calls = Calls()
    assert run_main({**environ, GEMINI: SECRET}, calls) == 1
    shown = capsys.readouterr()
    assert GEMINI in shown.err + shown.out
    assert SECRET not in shown.err + shown.out
    assert (calls.ready_urls, calls.runs) == ([], [])


def test_a_non_loopback_url_refuses_before_anything_runs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    environ, _ = runtime_with(tmp_path)
    calls = Calls()
    assert run_main({**environ, 'MERIDIAN_ADMIN_URL': 'http://localhost.evil.com:8181'}, calls) == 1
    shown = capsys.readouterr()
    assert 'MERIDIAN_ADMIN_URL' in shown.err + shown.out
    assert 'evil' not in shown.err + shown.out
    assert (calls.ready_urls, calls.runs) == ([], [])


def test_an_engine_that_is_not_ready_refuses_without_running(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    environ, _ = runtime_with(tmp_path)
    calls = Calls(ready=False)
    assert run_main(environ, calls) == 1
    assert 'not ready' in capsys.readouterr().err
    assert calls.ready_urls == ['http://127.0.0.1:8180']
    assert calls.runs == []


def test_an_allowed_run_launches_the_suite_in_contract_mode_with_the_code_it_returns(tmp_path: Path) -> None:
    environ, folder = runtime_with(tmp_path)
    calls = Calls(code=3)
    env = {
        **environ,
        'MERIDIAN_PUBLIC_URL': 'http://localhost:9180',
        'HTTP_PROXY': 'http://proxy.example:3128',
        'https_proxy': 'http://proxy.example:3128',
        'All_Proxy': 'http://proxy.example:3128',
        'MERIDIAN_CONTRACT': '0',
    }
    assert run_main(env, calls) == 3
    assert calls.ready_urls == ['http://localhost:9180']
    [(command, child, cwd)] = calls.runs
    assert command == ['uv', 'run', 'pytest', '-m', 'not fake_only']
    assert cwd == str(ROOT)
    assert child['MERIDIAN_CONTRACT'] == '1'
    assert Path(child['MERIDIAN_RUNTIME_DIR']) == folder
    assert child['PATH'] == environ['PATH']
    assert {key.upper() for key in child} & {'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY'} == set()
    assert child['NO_PROXY'] == child['no_proxy'] == '127.0.0.1,localhost,::1'


def test_the_child_gets_the_runtime_dir_that_was_checked(tmp_path: Path) -> None:
    environ, folder = runtime_with(tmp_path)
    (folder / 'sub').mkdir()
    calls = Calls()
    assert run_main({**environ, 'MERIDIAN_RUNTIME_DIR': str(folder / 'sub')}, calls) == 0
    assert Path(calls.runs[0][1]['MERIDIAN_RUNTIME_DIR']) == folder / 'sub'


def test_a_trailing_slash_on_the_url_still_asks_for_ready(tmp_path: Path) -> None:
    environ, _ = runtime_with(tmp_path)
    seen: list[str] = []

    def ready(url: str) -> bool:
        seen.append(url)
        return False

    script.main({**environ, 'MERIDIAN_PUBLIC_URL': 'http://127.0.0.1:8180/'}, ready=ready, runner=Calls().run)
    assert seen == ['http://127.0.0.1:8180/']
    assert script.ready_endpoint('http://127.0.0.1:8180/') == 'http://127.0.0.1:8180/ready'
    assert script.ready_endpoint('http://127.0.0.1:8180') == 'http://127.0.0.1:8180/ready'


def test_run_as_a_program_it_refuses_with_a_non_zero_exit_and_no_values(tmp_path: Path) -> None:
    environ, _ = runtime_with(tmp_path)
    # port 1 answers nobody: even if the guard were wrong, the readiness probe would still refuse
    child_env = {
        **environ,
        GEMINI: SECRET,
        'MERIDIAN_PUBLIC_URL': 'http://127.0.0.1:1',
        'SYSTEMROOT': os.environ.get('SYSTEMROOT', ''),
    }
    done = subprocess.run(
        [sys.executable, str(ROOT / 'scripts' / 'contract.py')],
        env=child_env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert done.returncode != 0
    assert GEMINI in done.stdout + done.stderr
    assert SECRET not in done.stdout + done.stderr


# -- the readiness probe, against a throwaway server on this machine -----------------------------------------------


class Answers(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == '/ready':
            self.send_response(200)
            self.end_headers()
        elif self.path == '/redirect/ready':
            self.send_response(302)
            port = cast(HTTPServer, self.server).server_port
            self.send_header('Location', f'http://127.0.0.1:{port}/ready')  # which would answer 200
            self.end_headers()
        elif self.path == '/empty/ready':
            self.send_response(204)
            self.end_headers()
        else:
            self.send_response(503)
            self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        return None


@pytest.fixture
def server() -> Iterator[str]:
    httpd = HTTPServer(('127.0.0.1', 0), Answers)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f'http://127.0.0.1:{httpd.server_port}'
    httpd.shutdown()
    httpd.server_close()


def test_ready_means_a_plain_200(server: str) -> None:
    assert script.engine_ready(server) is True
    assert script.engine_ready(server + '/') is True


def test_ready_is_false_for_other_answers_redirects_and_nobody_listening(server: str) -> None:
    assert script.engine_ready(server + '/redirect') is False
    assert script.engine_ready(server + '/elsewhere') is False
    assert script.engine_ready(server + '/empty') is False
    assert script.engine_ready('http://127.0.0.1:1') is False


def test_ready_does_not_go_through_a_proxy_from_the_environment(server: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('HTTP_PROXY', 'http://127.0.0.1:1')
    monkeypatch.setenv('http_proxy', 'http://127.0.0.1:1')
    assert script.engine_ready(server) is True
