"""Flask glue: a client from app config and a ready forwarding blueprint. Install with `gauss-meridian[flask]`.

`app.config['MERIDIAN_URL']` and `app.config['MERIDIAN_KEY']` configure the client; a missing one falls back to the
`MERIDIAN_URL` / `MERIDIAN_KEY` environment variables. The client is one long-lived shared client per settings, never
one per request.

`forward_blueprint` reads the request body in chunks through the core's 8 MiB cap, never through `request.data` or
`request.json`. A lower `MAX_CONTENT_LENGTH` of the app's own applies first: past it Flask answers its own 413 (after
`authorize`, when the body is read). The streamed reply is the response's body, and a WSGI server closes it however
sending ends (finished, the browser gone, a failed write, or never started), which closes the engine's stream.

Flask has no CSRF check of its own. When Flask-WTF's `CSRFProtect` (or any extension registered as
`app.extensions['csrf']` with an `exempt` method) is set up before the blueprint is registered, the blueprint is exempt
by default: the route takes JSON only and answers anything else with 415 before `authorize` runs, so a cross-site page
cannot send it without a CORS preflight, and `authorize` decides who may use it. Pass `csrf_exempt=False` to keep the
check.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping

try:
    from flask import Blueprint, Flask, Request, Response, current_app, request
    from flask.blueprints import BlueprintSetupState
    from werkzeug.exceptions import RequestEntityTooLarge
    from werkzeug.wrappers import Response as BaseResponse
except ImportError as missing:
    raise ImportError(
        'gauss_meridian.flask needs Flask: install it with `pip install "gauss-meridian[flask]"`.'
    ) from missing

from ._client import Meridian, shared_client
from .forward import AllowRule, Forwarder, Reply, ReplyBody

__all__ = ['MeridianExtension', 'forward_blueprint', 'meridian_client']

# How much of the browser's body is read at a time (the core stops reading past its 8 MiB cap)
_CHUNK = 65536


# --- The client from app config -------------------------------------------------------------------------------------


def _setting(value: object) -> str | None:
    # None falls back to the environment; any other value that is no string can never work, and '' is refused as such
    return value if value is None or isinstance(value, str) else ''


def _config() -> Mapping[str, object]:
    return current_app.config  # Flask's Config is an untyped dict: read as a mapping of names to values


def meridian_client() -> Meridian:
    """The shared client for the current app's `MERIDIAN_URL` / `MERIDIAN_KEY` (or the environment): one long-lived
    client per settings, never one per request. Settings that cannot work, or no app to read them from, give a client
    that refuses every call, which is never kept."""
    try:
        config = _config()
        url, key = _setting(config.get('MERIDIAN_URL')), _setting(config.get('MERIDIAN_KEY'))
    except Exception:
        url, key = '', ''
    return shared_client(url, key)


class MeridianExtension:
    """The Flask extension convention: `MeridianExtension(app)` or `init_app(app)` registers it as
    `app.extensions['meridian']`; `client()` is `meridian_client()` for the current app."""

    def __init__(self, app: Flask | None = None) -> None:
        if app is not None:
            self.init_app(app)

    def init_app(self, app: Flask) -> None:
        app.extensions['meridian'] = self

    def client(self) -> Meridian:
        return meridian_client()


# --- The blueprint --------------------------------------------------------------------------------------------------


def _response(reply: Reply[ReplyBody]) -> BaseResponse:
    # [1] The app's own response from authorize (any werkzeug response, Flask's included), returned as it is
    if isinstance(reply.passthrough, BaseResponse):
        return reply.passthrough

    # [2] Bytes, or a stream: the server's close() of the response closes the body (werkzeug's Response.close calls
    #     the body's close), however the iteration ended
    return Response(reply.body, status=reply.status, headers=reply.headers)


def forward_blueprint(
    *,
    authorize: Callable[[Request], object],
    allow: AllowRule | None = None,
    client: Meridian | Callable[[], Meridian] | None = None,
    url: str = '/api/meridian/chat',
    name: str = 'meridian',
    csrf_exempt: bool = True,
) -> Blueprint:
    """A blueprint with the forwarding route: `app.register_blueprint(forward_blueprint(authorize=...))`.

    Only the literal `True` from `authorize(request)` lets the request through; a werkzeug/Flask `Response` is returned
    as the app's own answer. A tuple such as `(jsonify(...), 401)` is no response, so it is a 403: build one with
    `make_response(jsonify(...), 401)`. Anything else, or a raise, is a 403. `client` is a Meridian, or a function
    returning the same long-lived one every time; by default the shared client for the app's config. The route never
    raises for an API, network or configuration problem: each is a response."""
    forwarder: Forwarder[Request] = Forwarder(
        client=meridian_client if client is None else client,
        authorize=authorize,
        allow=allow,
        response_type=BaseResponse,
    )
    blueprint = Blueprint(name, __name__)

    @blueprint.route(url, methods=['GET', 'POST', 'PUT', 'PATCH', 'DELETE'])
    def forward() -> BaseResponse:
        # [1] The body in chunks through the core's cap, never `request.data` / `request.json`; the app's own lower
        #     MAX_CONTENT_LENGTH is noted, and answered by Flask itself
        refused: list[RequestEntityTooLarge] = []

        def chunks() -> Iterator[bytes]:
            try:
                stream = request.stream
                while piece := stream.read(_CHUNK):
                    yield piece
            except RequestEntityTooLarge as limit:
                refused.append(limit)
                raise

        reply = forwarder.handle(
            request,
            method=request.method,
            content_type=request.headers.get('Content-Type'),
            idempotency_key=request.headers.get('Idempotency-Key'),
            read_body=chunks,
        )
        if refused:
            raise refused[0]
        return _response(reply)

    # [2] Exempt from an installed CSRFProtect (Flask-WTF, found by its `exempt` method, never imported)
    if csrf_exempt:

        def exempt(state: BlueprintSetupState) -> None:
            mark = getattr(state.app.extensions.get('csrf'), 'exempt', None)
            if callable(mark):
                mark(blueprint)

        blueprint.record_once(exempt)
    return blueprint
