"""Provisioning with an integrator key (`gmi_`) on the engine's admin listener. Never in a browser.

`MeridianAdmin` and `AsyncMeridianAdmin` wrap the 21 operations an integrator key may call: organisations, projects,
keys, bring-your-own provider keys, a project's model list, the model catalogue, usage and logs. The supplier
catalogue is not wrapped (it returns only supplier prices, and no SDK exposes a price), nor is renaming or deleting
an organisation (a person does that in the console).

Every call returns a `Result` and never raises for an API, network, decode or configuration problem. A client built
with a missing, malformed or wrong-typed URL or key, with both `http_client` and `transport`, or with either of the
wrong kind, answers every call with an `invalid_request` / `client_error` and sends nothing. So does a call whose body
is not a mapping or holds NaN or infinity, or whose `page_size` is not an int. A delete answers `Ok(None)`. A query
value that is `None` or `''` is left out.

Retries follow one rule, decided in the transport from the call itself: a read or a delete is retried normally; every
write (POST, PUT, PATCH) is retried only on `rate_limited`, because a repeat could do the work twice. The one
exception is `orgs.create` with a non-blank `external_reference`, which the engine replays. Admin calls send no
`Idempotency-Key`. A delete that succeeded but whose 204 was lost on the way back is retried, and the retry may then
answer `not_found`: the resource is gone either way.

Path ids are each one encoded segment. An id that is empty, all white space, `.` or `..` (as given, or after one
percent-decode), or an organisation id of `by-reference` (the engine's lookup route), is refused with
`invalid_request` and nothing is sent. The refusal is the same for every spelling and never quotes the id.

No result carries a price: every key naming a cost, a price, a per-million rate, a recovery rate or a write-off rate
is removed, at any depth.

The issued key and the credentials you store are secrets:

* `keys.create` returns the new key once. `result.data['api_key']` is a `Secret`, a `str` that prints as
  `[redacted]` through `repr()`, which `str()` of a result or of a dict holding it, an f-string of either, `%s` / `%r`
  logging and pytest's assertion output all use. The key itself is the value: `result.data['api_key']` compares,
  concatenates and encodes as the real key, and `json.dumps(result.data)` carries it, because handing the key to
  its owner as data is the point of issuing it. Printing the value on its own (`print(key)`, `f'{key}'`) shows the
  key; extracting it is the choice to use it.
* `provider_credentials.put` holds a text `credential` as a `Secret` in its own copy of your body (your dict is not
  changed) and sends it in full; the bytes on the wire are the same. Wrap it yourself (`{'credential': Secret(raw)}`)
  to hide it in your own code too. The SDK never echoes a request body in an error.

What a `Secret` cannot hide: only `repr()` is redacted, so anything that makes a new plain `str` from it, or reads its
characters, shows the key. That is `str(key)`, an f-string `f'{key}'`, `format(key)` / `'{}'.format(key)`,
`'%s' % key`, `+` and `''.join(...)`, slicing (`key[:]`), and every string method that returns text (`.strip()`,
`.upper()`, ...); `%(api_key)s` and `'{api_key}'.format(**data)` formatting of a dict holding it; logging the bare value
with `%s` (`log.info('%s', key)`, or `log.info(key)`); an exception built from the value (`ValueError(key)`, whose
`str()` and traceback print it); JSON and structured log formatters (which encode the value, not its `repr`); the
bytes of `pickle.dumps`; and pytest's line diff of two strings (`assert key == 'x'` prints both texts).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast
from urllib.parse import quote, unquote

import httpx

from ._config import ADMIN, ASYNC_IO, SYNC_IO, refusal, resolve_settings
from ._errors import Err, MeridianError, Ok, Result, config_error
from ._prices import without_prices
from ._transport import AsyncRefusedTransport, AsyncTransport, Call, RefusedTransport, SyncTransport

if TYPE_CHECKING:
    from ._generated.admin import (
        CreateOrganisation,
        CreateProject,
        IssueKey,
        ModelAllowlist,
        SetCredential,
        UpdateProject,
    )

__all__ = ['AsyncMeridianAdmin', 'MeridianAdmin', 'Secret']

R = TypeVar('R')
BY_REFERENCE = (
    'by-reference'  # the engine's /v1/orgs/by-reference/{reference} route, which an organisation id would shadow
)
_UNSAFE_ID = config_error(
    f'A resource id must be a string that is not blank, "." or "..", and an organisation id is not "{BY_REFERENCE}".'
)
_NO_PROJECT = config_error('A project_id is required.')
_NOT_AN_OBJECT = config_error('A request body must be a mapping (a JSON object).')
_BAD_PAGE_SIZE = config_error('page_size must be an int.')


class Secret(str):
    """A `str` that prints as `[redacted]` through `repr()` (so in anything holding it); otherwise it is the value."""

    __slots__ = ()

    def __repr__(self) -> str:
        return '[redacted]'


@dataclass(frozen=True)
class _Op:
    """One operation as the Call to make (or the refusal to answer with)."""

    call: Call | MeridianError
    hides_key: bool = False  # the reply holds a secret that is shown once


def _decoded(text: str) -> str:
    try:
        return unquote(text, errors='strict')
    except UnicodeDecodeError:  # not valid percent-encoded UTF-8: the text itself is what gets encoded
        return text


def _route(template: str, *ids: object) -> str | MeridianError:
    """The path with each id as one encoded segment, or the refusal for an id that could reach another route."""
    org_first = template.startswith('/v1/orgs/{}')
    segments: list[str] = []
    for position, value in enumerate(ids):
        # [1] An id is a string; look at it as given and after one percent-decode (%2e%2e is "..")
        if not isinstance(value, str):
            return _UNSAFE_ID
        # [2] Refuse what would reach, collapse into or shadow another route; nothing is sent
        for candidate in (value, _decoded(value)):
            blank = not candidate.strip() or candidate in ('.', '..')
            if blank or (org_first and position == 0 and candidate == BY_REFERENCE):
                return _UNSAFE_ID
        # [3] Everything else is one segment: "/", "?", "#" and "%" are escaped, never interpreted
        try:
            segments.append(quote(value, safe=''))
        except UnicodeError:  # a lone surrogate has no UTF-8
            return _UNSAFE_ID
    return template.format(*segments)


def _get(path: str | MeridianError, query: dict[str, Any] | None = None) -> _Op:
    return _Op(path if isinstance(path, MeridianError) else Call('GET', path, query=query))


def _write(
    method: str, path: str | MeridianError, body: object, *, repeatable: bool = False, hides_key: bool = False
) -> _Op:
    # A body is a mapping (a JSON object), sent as a copy; anything else is refused and nothing is sent
    if isinstance(path, MeridianError):
        return _Op(path)
    if not isinstance(body, Mapping):
        return _Op(_NOT_AN_OBJECT)
    return _Op(Call(method, path, body=dict(cast('Mapping[str, Any]', body)), repeatable=repeatable), hides_key)


def _with_secret(body: object) -> object:
    """A credential body with a text `credential` held as a `Secret`, in a new dict (the caller's stays as it is); the
    wire bytes are the same. Any other body as it is."""
    fields = cast('Mapping[str, Any]', body)
    credential = fields.get('credential') if isinstance(body, Mapping) else None
    if isinstance(credential, str) and not isinstance(credential, Secret):
        return {**fields, 'credential': Secret(credential)}
    return fields


def _delete(path: str | MeridianError) -> _Op:
    return _Op(path if isinstance(path, MeridianError) else Call('DELETE', path, no_content=True))


def _scoped(path: str, project_id: str, **query: Any) -> _Op:
    if not isinstance(project_id, str) or not project_id.strip():  # pyright: ignore[reportUnnecessaryIsInstance]
        return _Op(_NO_PROJECT)
    return _get(path, {'project_id': project_id, **query})


class _Group(Generic[R]):
    def __init__(self, send: Callable[[_Op], R]) -> None:
        self._send = send


class _Orgs(_Group[R]):
    def create(self, body: CreateOrganisation) -> R:
        # The engine replays a repeat by external_reference, so only with one is the call retried like a read
        # (a body that is no mapping is refused by _write)
        reference = body.get('external_reference') if isinstance(cast('object', body), Mapping) else None
        return self._send(
            _write('POST', '/v1/orgs', body, repeatable=isinstance(reference, str) and bool(reference.strip()))
        )

    def get(self, org: str) -> R:
        return self._send(_get(_route('/v1/orgs/{}', org)))

    def get_by_reference(self, reference: str) -> R:
        return self._send(_get(_route('/v1/orgs/by-reference/{}', reference)))

    def list(self) -> R:
        return self._send(_get('/v1/orgs'))


class _Projects(_Group[R]):
    def create(self, org: str, body: CreateProject) -> R:
        return self._send(_write('POST', _route('/v1/orgs/{}/projects', org), body))

    def get(self, org: str, project: str) -> R:
        return self._send(_get(_route('/v1/orgs/{}/projects/{}', org, project)))

    def list(self, org: str) -> R:
        return self._send(_get(_route('/v1/orgs/{}/projects', org)))

    def update(self, org: str, project: str, body: UpdateProject) -> R:
        return self._send(_write('PATCH', _route('/v1/orgs/{}/projects/{}', org, project), body))

    def delete(self, org: str, project: str) -> R:
        return self._send(_delete(_route('/v1/orgs/{}/projects/{}', org, project)))


class _Keys(_Group[R]):
    def create(self, org: str, project: str, body: IssueKey | None = None) -> R:
        """Issue a key. The reply's `api_key` is shown once, as a `Secret`: store it now."""
        path = _route('/v1/orgs/{}/projects/{}/keys', org, project)
        return self._send(_write('POST', path, {} if body is None else body, hides_key=True))

    def list(self, org: str, project: str) -> R:
        return self._send(_get(_route('/v1/orgs/{}/projects/{}/keys', org, project)))

    def revoke(self, org: str, project: str, key: str) -> R:
        return self._send(_delete(_route('/v1/orgs/{}/projects/{}/keys/{}', org, project, key)))


class _ProviderCredentials(_Group[R]):
    def list(self, org: str) -> R:
        return self._send(_get(_route('/v1/orgs/{}/provider-credentials', org)))

    def put(self, org: str, provider: str, body: SetCredential) -> R:
        """Store or replace the organisation's key for a provider. The SDK holds a text `credential` as a `Secret`
        (in its own copy of the body; yours is untouched) and sends it unchanged."""
        path = _route('/v1/orgs/{}/provider-credentials/{}', org, provider)
        return self._send(_write('PUT', path, _with_secret(body)))

    def delete(self, org: str, provider: str) -> R:
        return self._send(_delete(_route('/v1/orgs/{}/provider-credentials/{}', org, provider)))


class _ProjectModels(_Group[R]):
    def get(self, org: str, project: str) -> R:
        return self._send(_get(_route('/v1/orgs/{}/projects/{}/models', org, project)))

    def put(self, org: str, project: str, body: ModelAllowlist) -> R:
        return self._send(_write('PUT', _route('/v1/orgs/{}/projects/{}/models', org, project), body))


class _Models(_Group[R]):
    def list(self) -> R:
        return self._send(_get('/v1/models'))

    def get(self, model: str) -> R:
        return self._send(_get(_route('/v1/models/{}', model)))


class _Analytics(_Group[R]):
    def usage(self, project_id: str, *, from_: str | None = None, to: str | None = None) -> R:
        return self._send(_scoped('/v1/analytics/usage', project_id, **{'from': from_, 'to': to}))


class _Logs(_Group[R]):
    def list(self, project_id: str, *, page_size: int | None = None, cursor: str | None = None) -> R:
        size = cast('object', page_size)
        if size is not None and (isinstance(size, bool) or not isinstance(size, int)):
            return self._send(_Op(_BAD_PAGE_SIZE))
        return self._send(_scoped('/v1/logs', project_id, page_size=page_size, cursor=cursor))


def _hide_key(data: Any) -> Any:
    """A key-issue reply with its `api_key` as a Secret; a reply of any other shape as it is."""
    if not isinstance(data, dict):
        return data
    fields = cast('dict[str, Any]', data)
    if isinstance(fields.get('api_key'), str):
        return {**fields, 'api_key': Secret(fields['api_key'])}
    return fields


def _finish(result: Result[Any], hides_key: bool) -> Result[Any]:
    # [1] An error, or no content (a delete), is the answer as it is
    if result.error is not None or result.data is None:
        return result
    # [2] No price in any result; an issued key is a Secret
    data: Any = without_prices(result.data)
    return Ok(_hide_key(data) if hides_key else data, request_id=result.request_id)


class MeridianAdmin:
    """The admin client: provisioning with an integrator key (`gmi_`). Server side only."""

    def __init__(
        self,
        url: str | None = None,
        key: str | None = None,
        *,
        timeout: float | None = None,
        max_retries: int | None = None,
        http_client: httpx.Client | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        """`url` and `key` default to `MERIDIAN_ADMIN_URL` and `MERIDIAN_INTEGRATOR_KEY`."""
        base, token = resolve_settings(url, key, ADMIN)
        refused = refusal(base, token, http_client, transport, ADMIN, SYNC_IO)
        self._transport: RefusedTransport | SyncTransport = (
            RefusedTransport(refused)
            if refused
            else SyncTransport(
                base, token, timeout=timeout, max_retries=max_retries, http_client=http_client, transport=transport
            )
        )
        self.orgs: _Orgs[Result[Any]] = _Orgs(self._send)
        self.projects: _Projects[Result[Any]] = _Projects(self._send)
        self.keys: _Keys[Result[Any]] = _Keys(self._send)
        self.provider_credentials: _ProviderCredentials[Result[Any]] = _ProviderCredentials(self._send)
        self.project_models: _ProjectModels[Result[Any]] = _ProjectModels(self._send)
        self.models: _Models[Result[Any]] = _Models(self._send)
        self.analytics: _Analytics[Result[Any]] = _Analytics(self._send)
        self.logs: _Logs[Result[Any]] = _Logs(self._send)

    def _send(self, op: _Op) -> Result[Any]:
        if isinstance(op.call, MeridianError):
            return Err(op.call)
        return _finish(self._transport.json(op.call), op.hides_key)

    def close(self) -> None:
        self._transport.close()

    def __enter__(self) -> MeridianAdmin:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class AsyncMeridianAdmin:
    """The async admin client: the same API as MeridianAdmin, awaited."""

    def __init__(
        self,
        url: str | None = None,
        key: str | None = None,
        *,
        timeout: float | None = None,
        max_retries: int | None = None,
        http_client: httpx.AsyncClient | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        """`url` and `key` default to `MERIDIAN_ADMIN_URL` and `MERIDIAN_INTEGRATOR_KEY`."""
        base, token = resolve_settings(url, key, ADMIN)
        refused = refusal(base, token, http_client, transport, ADMIN, ASYNC_IO)
        self._transport: AsyncRefusedTransport | AsyncTransport = (
            AsyncRefusedTransport(refused)
            if refused
            else AsyncTransport(
                base, token, timeout=timeout, max_retries=max_retries, http_client=http_client, transport=transport
            )
        )
        self.orgs: _Orgs[Awaitable[Result[Any]]] = _Orgs(self._send)
        self.projects: _Projects[Awaitable[Result[Any]]] = _Projects(self._send)
        self.keys: _Keys[Awaitable[Result[Any]]] = _Keys(self._send)
        self.provider_credentials: _ProviderCredentials[Awaitable[Result[Any]]] = _ProviderCredentials(self._send)
        self.project_models: _ProjectModels[Awaitable[Result[Any]]] = _ProjectModels(self._send)
        self.models: _Models[Awaitable[Result[Any]]] = _Models(self._send)
        self.analytics: _Analytics[Awaitable[Result[Any]]] = _Analytics(self._send)
        self.logs: _Logs[Awaitable[Result[Any]]] = _Logs(self._send)

    async def _send(self, op: _Op) -> Result[Any]:
        if isinstance(op.call, MeridianError):
            return Err(op.call)
        return _finish(await self._transport.json(op.call), op.hides_key)

    async def aclose(self) -> None:
        await self._transport.aclose()

    async def __aenter__(self) -> AsyncMeridianAdmin:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()
