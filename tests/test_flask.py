"""The Flask adapter, driven the way a WSGI server drives it: the server iterates the response and then calls its
`close()`, however the iteration ended (PEP 3333). Flask's own test client covers the ordinary round trip."""

from __future__ import annotations

import io
import json
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any, cast

import pytest
from flask import Flask, jsonify, make_response
from werkzeug.wrappers import Response as WerkzeugResponse

import meridian.flask
from meridian import Meridian
from meridian._client import shared_client
from meridian.flask import MeridianExtension, forward_blueprint, meridian_client
from meridian.forward import ReplyBody
from tests.support.adapter_contract import CHAT, CHECKS, MESSAGES, STREAM, Wire, allow
from tests.support.fake_engine import Scripted, chunk, engine_error, frame
from tests.support.recorder import Recorder

pytestmark = [pytest.mark.anyio, pytest.mark.fake_only]
ROUTE = '/api/meridian/chat'


@dataclass
class Served:
    status: int
    headers: dict[str, str]
    body: bytes


class Upload:
    """`wsgi.input`: hands the app the body's chunks only as it reads them, as a socket would."""

    def __init__(self, body: bytes | Iterator[bytes]) -> None:
        self._chunks = iter([body]) if isinstance(body, bytes) else body
        self._pending = b''

    def read(self, size: int = -1) -> bytes:
        if not self._pending:
            self._pending = next(self._chunks, b'')
        if size < 0:
            size = len(self._pending)
        piece, self._pending = self._pending[:size], self._pending[size:]
        return piece

    def readinto(self, buffer: bytearray | memoryview) -> int:
        piece = self.read(len(buffer))
        buffer[: len(piece)] = piece
        return len(piece)

    def readline(self, size: int = -1) -> bytes:  # the route never reads lines
        raise AssertionError('the body was read by lines')


def wsgi(
    app: Flask,
    *,
    method: str = 'POST',
    content_type: str | None = 'application/json',
    headers: Mapping[str, str] | None = None,
    body: bytes | Iterator[bytes] = CHAT,
    leave_after: int | None = None,
    raise_on_leave: bool = False,
) -> Served:
    """One request served as a WSGI server would. `leave_after=n`: the server stops after n body chunks (0: before the
    first), because the browser left; with `raise_on_leave` its write raises OSError instead. It calls `close()` in
    every case."""
    environ: dict[str, Any] = {
        'REQUEST_METHOD': method,
        'SCRIPT_NAME': '',
        'PATH_INFO': ROUTE,
        'QUERY_STRING': '',
        'SERVER_NAME': 'testserver',
        'SERVER_PORT': '80',
        'SERVER_PROTOCOL': 'HTTP/1.1',
        'HTTP_HOST': 'testserver',
        'wsgi.version': (1, 0),
        'wsgi.url_scheme': 'http',
        'wsgi.input': Upload(body),
        'wsgi.errors': io.StringIO(),
        'wsgi.multithread': True,
        'wsgi.multiprocess': False,
        'wsgi.run_once': False,
        # A streamed body's length is announced up front, as a browser does: here, more than any body that is sent
        'CONTENT_LENGTH': str(len(body) if isinstance(body, bytes) else 1 << 30),
    }
    if content_type is not None:
        environ['CONTENT_TYPE'] = content_type
    for name, value in (headers or {}).items():
        environ['HTTP_' + name.upper().replace('-', '_')] = value
    started: list[tuple[str, list[tuple[str, str]]]] = []

    def start_response(status: str, response_headers: list[tuple[str, str]], *_: object) -> Callable[[bytes], None]:
        started.append((status, response_headers))
        return lambda data: None

    response = app(environ, start_response)
    received: list[bytes] = []
    try:
        if leave_after != 0:
            for piece in response:
                if piece:
                    received.append(piece)
                if leave_after is not None and len(received) >= leave_after:
                    if raise_on_leave:
                        raise OSError('the client is gone')
                    break
    finally:
        close = getattr(response, 'close', None)
        if callable(close):
            close()
    status, raw_headers = started[0]
    return Served(int(status.split()[0]), {k.lower(): v for k, v in raw_headers}, b''.join(received))


def sync_client(engine: Any) -> Meridian:
    return Meridian(engine.public_url, engine.project_key, transport=engine.transport, max_retries=0)


def app_with(blueprint: Any, sent: list[WerkzeugResponse] | None = None) -> Flask:
    app = Flask(__name__)
    app.register_blueprint(blueprint)
    if sent is not None:

        @app.after_request
        def keep(response: WerkzeugResponse) -> WerkzeugResponse:
            sent.append(response)
            return response

    return app


def app_for(engine: Any, authorize: Callable[[Any], object] = allow) -> Flask:
    return app_with(forward_blueprint(authorize=authorize, client=sync_client(engine)))


class FlaskHarness:
    spools_the_body = False

    def __init__(self, engine: Any) -> None:
        self.engine = engine
        self.client = sync_client(engine)

    def own_response(self, status: int, data: dict[str, Any]) -> object:
        return make_response(jsonify(data), status)

    async def send(
        self,
        *,
        authorize: Callable[[Any], object] = allow,
        method: str = 'POST',
        content_type: str | None = 'application/json',
        body: bytes | Iterator[bytes] = CHAT,
        leave_after: int | None = None,
    ) -> Wire:
        sent: list[WerkzeugResponse] = []
        app = app_with(forward_blueprint(authorize=authorize, client=self.client), sent)
        served = wsgi(app, method=method, content_type=content_type, body=body, leave_after=leave_after)

        async def close_again() -> int:
            bodies = [response.response for response in sent if isinstance(response.response, ReplyBody)]
            for body in bodies:
                body.close()
            return len(bodies)

        return Wire(served.status, served.headers, served.body, close_again)


@pytest.mark.parametrize('name', list(CHECKS))
async def test_the_adapter_contract(name: str, engine: Any) -> None:
    await CHECKS[name](FlaskHarness(engine))


# --- Flask's own test client ------------------------------------------------------------------------------------------


def test_flasks_test_client_streams_through(engine: Any) -> None:
    engine.next(Scripted(chunks=[frame(chunk('a')), frame('[DONE]')]))
    r = app_for(engine).test_client().post(ROUTE, json={'messages': MESSAGES, 'stream': True})
    assert r.status_code == 200 and r.data == (frame(chunk('a')) + frame('[DONE]')).encode()
    assert engine.upstream_closed()


def test_flasks_test_client_buffered_and_415(engine: Any) -> None:
    client = app_for(engine).test_client()
    assert client.post(ROUTE, json={'messages': MESSAGES}).get_json()['choices'][0]['message']['content'] == 'hello'
    refused = client.post(ROUTE, data='{}', content_type='text/plain')
    assert refused.status_code == 415 and len(engine.requests) == 1


# --- authorize: True or a response; a tuple is no response ------------------------------------------------------------


def test_a_tuple_from_authorize_is_403(engine: Any) -> None:
    r = app_for(engine, authorize=lambda _: (jsonify(login=True), 401)).test_client().post(ROUTE, json={'messages': 1})
    assert r.status_code == 403 and r.get_json()['error']['code'] == 'permission_denied'
    assert engine.requests == []


def test_make_response_of_a_tuple_is_the_apps_response(engine: Any) -> None:
    app = app_for(engine, authorize=lambda _: make_response(jsonify(login=True), 401))
    r = app.test_client().post(ROUTE, json={'messages': MESSAGES})
    assert r.status_code == 401 and r.get_json() == {'login': True} and engine.requests == []


def test_a_plain_werkzeug_response_is_the_apps_response(engine: Any) -> None:
    app = app_for(engine, authorize=lambda _: WerkzeugResponse('pay first', status=402))
    r = app.test_client().post(ROUTE, json={'messages': MESSAGES})
    assert r.status_code == 402 and r.data == b'pay first' and engine.requests == []


# --- The engine's stream is closed on every exit ----------------------------------------------------------------------


def test_a_write_that_raises_mid_stream_still_closes_upstream(engine: Any) -> None:
    engine.next(Scripted(chunks=[*(frame(chunk(str(i))) for i in range(20)), frame('[DONE]')]))
    with pytest.raises(OSError, match='gone'):
        wsgi(app_for(engine), body=STREAM, leave_after=1, raise_on_leave=True)
    assert engine.upstream_closed() and engine.parts_read() < 21


def test_an_engine_stream_that_raises_mid_way_is_closed(engine: Any) -> None:
    engine.next(Scripted(chunks=[frame(chunk('a')), frame(chunk('b')), frame('[DONE]')], fail_after=1))
    with pytest.raises(RuntimeError, match='boom'):
        wsgi(app_for(engine), body=STREAM)
    assert engine.upstream_closed()


# --- The body goes through the core's cap -----------------------------------------------------------------------------


def test_flasks_own_lower_body_limit_applies_first(engine: Any) -> None:
    app = app_for(engine)
    app.config['MAX_CONTENT_LENGTH'] = 64
    r = app.test_client().post(ROUTE, json={'messages': [{'role': 'user', 'content': 'x' * 100}]})
    assert r.status_code == 413 and engine.requests == []


def test_flasks_body_limit_comes_after_authorize(engine: Any) -> None:
    app = app_for(engine, authorize=lambda _: False)
    app.config['MAX_CONTENT_LENGTH'] = 64
    r = app.test_client().post(ROUTE, json={'messages': [{'role': 'user', 'content': 'x' * 100}]})
    assert r.status_code == 403 and engine.requests == []


# --- Clients from app config ------------------------------------------------------------------------------------------


def config_of(app: Flask) -> dict[str, object]:
    return app.config  # Flask's Config is an untyped dict


def configure(app: Flask, values: Mapping[str, object]) -> None:
    config_of(app).update(values)


def test_the_config_client_is_the_shared_one(engine: Any) -> None:
    app = Flask(__name__)
    configure(app, {'MERIDIAN_URL': engine.public_url, 'MERIDIAN_KEY': engine.project_key})
    with app.app_context():
        client = meridian_client()
        assert client is meridian_client() and client is shared_client(engine.public_url, engine.project_key)
        assert MeridianExtension(app).client() is client
    assert app.extensions['meridian'] is not None


def test_missing_config_falls_back_to_the_environment(engine: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv('MERIDIAN_URL', engine.public_url)
    monkeypatch.setenv('MERIDIAN_KEY', engine.project_key)
    app = Flask(__name__)
    with app.app_context():
        assert meridian_client() is shared_client()
        app.config['MERIDIAN_URL'] = engine.public_url
        assert meridian_client() is shared_client()


UNUSABLE = [{}, {'MERIDIAN_URL': 5, 'MERIDIAN_KEY': 'k'}, {'MERIDIAN_URL': '', 'MERIDIAN_KEY': ''}]


@pytest.mark.parametrize('config', UNUSABLE)
def test_unusable_config_is_never_cached_nor_replaced_by_the_environment(
    config: dict[str, object], engine: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    if config:
        monkeypatch.setenv('MERIDIAN_URL', engine.public_url)
        monkeypatch.setenv('MERIDIAN_KEY', engine.project_key)
    else:
        monkeypatch.delenv('MERIDIAN_URL', raising=False)
        monkeypatch.delenv('MERIDIAN_KEY', raising=False)
    app = app_with(forward_blueprint(authorize=allow))
    configure(app, config)
    with app.app_context():
        assert meridian_client() is not meridian_client()
    r = app.test_client().post(ROUTE, json={'messages': MESSAGES})
    assert r.status_code == 503 and r.get_json()['error']['code'] == 'service_unavailable'


def test_the_blueprints_default_client_is_meridian_client(engine: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    # The shared client cannot reach the fake engine, so the public `meridian_client` stands in for it here
    made: list[Meridian] = []

    def stand_in() -> Meridian:
        made.append(sync_client(engine))
        return made[-1]

    monkeypatch.setattr(meridian.flask, 'meridian_client', stand_in)
    r = app_with(forward_blueprint(authorize=allow)).test_client().post(ROUTE, json={'messages': MESSAGES})
    assert r.status_code == 200 and len(made) == 1 and len(engine.requests) == 1


def test_a_closed_config_client_is_replaced_so_the_route_keeps_answering(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder = Recorder(monkeypatch)
    app = app_with(forward_blueprint(authorize=allow))
    configure(app, {'MERIDIAN_URL': 'http://flask-closed.engine.test', 'MERIDIAN_KEY': 'gmr_' + '8' * 64})
    with app.app_context(), meridian_client() as client:
        assert client.chat.create({'messages': cast('Any', MESSAGES)}).error is None
    r = app.test_client().post(ROUTE, json={'messages': MESSAGES})
    assert r.status_code == 200 and len(recorder.senders) == 2


def test_outside_an_app_the_client_refuses_rather_than_raises() -> None:
    refused = meridian_client().chat.create({'messages': [{'role': 'user', 'content': 'hi'}]})
    assert refused.error is not None and refused.error.type == 'client_error'


# --- CSRF -------------------------------------------------------------------------------------------------------------


class FakeCsrf:
    """What Flask-WTF's CSRFProtect offers: `exempt(view_or_blueprint)`."""

    def __init__(self) -> None:
        self.exempted: list[object] = []

    def exempt(self, view: object) -> object:
        self.exempted.append(view)
        return view


@pytest.mark.parametrize('exempt', [True, False])
def test_csrf_protect_exempts_the_blueprint_unless_told_not_to(engine: Any, exempt: bool) -> None:
    app = Flask(__name__)
    csrf = FakeCsrf()
    app.extensions['csrf'] = csrf
    blueprint = forward_blueprint(authorize=allow, client=sync_client(engine), csrf_exempt=exempt)
    app.register_blueprint(blueprint)
    assert csrf.exempted == ([blueprint] if exempt else [])


def test_without_a_usable_csrf_extension_nothing_happens(engine: Any) -> None:
    app = Flask(__name__)
    app.extensions['csrf'] = object()
    app.register_blueprint(forward_blueprint(authorize=allow, client=sync_client(engine)))
    assert app.test_client().post(ROUTE, json={'messages': MESSAGES}).status_code == 200


# --- Set-up mistakes raise; nothing else does -------------------------------------------------------------------------


def test_a_missing_authorize_raises() -> None:
    with pytest.raises(TypeError):
        forward_blueprint(authorize=cast('Any', None))


def test_an_engine_failure_is_a_response_not_an_exception(engine: Any) -> None:
    engine.next(Scripted(network_error=True))
    r = app_for(engine).test_client().post(ROUTE, json={'messages': MESSAGES})
    assert r.status_code == 503 and r.get_json()['error']['code'] == 'service_unavailable'


def test_credential_refusal_is_a_generic_503(engine: Any) -> None:
    engine.next(engine_error('key_expired'))
    r = app_for(engine).test_client().post(ROUTE, json={'messages': MESSAGES})
    assert r.status_code == 503 and b'key_expired' not in r.data


def test_the_browsers_idempotency_key_and_messages_reach_the_engine(engine: Any) -> None:
    wsgi(app_for(engine), headers={'idempotency-key': 'abc-123'})
    assert engine.requests[0].headers['idempotency-key'] == 'abc-123'
    assert engine.bodies()[0]['messages'] == MESSAGES


def test_the_url_and_name_are_the_apps_to_choose(engine: Any) -> None:
    app = app_with(forward_blueprint(authorize=allow, client=sync_client(engine), url='/chat', name='chat'))
    assert app.test_client().post('/chat', json={'messages': MESSAGES}).status_code == 200
    assert 'chat' in app.blueprints


def test_a_raw_body_is_never_json_decoded_by_flask(engine: Any) -> None:
    # The body reaches the core as bytes: Flask's request.json (which would accept a BOM or other encodings) is unused
    utf16 = json.dumps({'messages': MESSAGES}).encode('utf-16')
    r = app_for(engine).test_client().post(ROUTE, data=utf16, content_type='application/json')
    assert r.status_code == 400 and engine.requests == []
