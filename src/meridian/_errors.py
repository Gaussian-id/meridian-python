from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Final, Generic, TypeVar, Union, cast

T = TypeVar('T')

ENGINE_ERROR_CODES: Final = (
    'invalid_request',
    'invalid_api_key',
    'key_revoked',
    'key_expired',
    'project_unavailable',
    'organisation_unavailable',
    'model_not_allowed',
    'byok_required',
    'project_host_mismatch',
    'model_not_found',
    'route_not_found',
    'method_not_allowed',
    'capability_unsupported',
    'body_too_large',
    'rate_limited',
    'insufficient_credit',
    'hard_limit_reached',
    'entitlement_missing',
    'idempotency_conflict',
    'upstream_unavailable',
    'upstream_error',
    'deadline_exceeded',
    'service_unavailable',
    'internal',
    'invalid_token',
    'permission_denied',
    'not_found',
    'conflict',
)

STATUS_BY_CODE: Final[Mapping[str, int]] = {
    'invalid_request': 400,
    'capability_unsupported': 400,
    'invalid_api_key': 401,
    'key_revoked': 401,
    'key_expired': 401,
    'invalid_token': 401,
    'insufficient_credit': 402,
    'project_unavailable': 403,
    'organisation_unavailable': 403,
    'model_not_allowed': 403,
    'byok_required': 403,
    'project_host_mismatch': 403,
    'permission_denied': 403,
    'entitlement_missing': 403,
    'model_not_found': 404,
    'route_not_found': 404,
    'not_found': 404,
    'method_not_allowed': 405,
    'idempotency_conflict': 409,
    'conflict': 409,
    'body_too_large': 413,
    'rate_limited': 429,
    'hard_limit_reached': 429,
    'internal': 500,
    'upstream_unavailable': 502,
    'upstream_error': 502,
    'service_unavailable': 503,
    'deadline_exceeded': 504,
}

# The five types an engine envelope may claim; `client_error` is the SDK's own and never comes off the wire
ENGINE_ERROR_TYPES: Final = frozenset(
    {'invalid_request_error', 'authentication_error', 'permission_error', 'rate_limit_error', 'api_error'}
)
RETRYABLE: Final = frozenset(
    {'rate_limited', 'upstream_unavailable', 'service_unavailable', 'deadline_exceeded', 'network_error'}
)


@dataclass(frozen=True)
class MeridianError:
    code: str
    type: str
    status: int
    message: str
    retryable: bool
    request_id: str | None = None
    # The engine's own code when this SDK does not recognise it (`code` is then `internal`); None for every other
    # error. Not part of the error's identity: it appears in neither `repr` nor `==`.
    engine_code: str | None = field(default=None, repr=False, compare=False)
    # The response's `Retry-After` header when it was valid (delta-seconds or an HTTP date), as sent; None otherwise.
    # Not part of the error's identity either.
    retry_after: str | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class Ok(Generic[T]):
    """A successful call: `data` is the result and `error` is None.

    `request_id` is the engine's `x-request-id` for the call, when the response had one (give it to Gaussian support
    with any question about the call). It is not part of the result's identity: it is in neither `repr` nor `==`.
    """

    data: T
    error: None = None
    request_id: str | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class Err:
    error: MeridianError
    data: None = None


Result = Union[Ok[T], Err]  # noqa: UP007  (a runtime alias: `X | Y` on a generic is 3.10-fragile with TypeVars)


_REQUEST_ID = re.compile(r'[ -~]{1,255}')


def valid_request_id(value: object) -> str | None:
    """A request id fit for a header and a body: 1-255 printable ASCII characters (spaces allowed, never a line
    break). Anything else is None."""
    return value if isinstance(value, str) and _REQUEST_ID.fullmatch(value) else None


def client_error(code: str, message: str, status: int = 0, request_id: str | None = None) -> MeridianError:
    """`network_error` or `aborted`: a failure on the SDK's side of the wire."""
    return MeridianError(code, 'client_error', status, message, code == 'network_error', request_id)


def config_error(message: str) -> MeridianError:
    """A mistake in how the SDK was set up: never sent, never retried."""
    return MeridianError('invalid_request', 'client_error', 0, message, False)


NOT_JSON: Final = object()  # what load_json gives for text that is not JSON
MALFORMED: Final = object()  # what error_member gives for an error member it cannot read


def load_json(text: str, **hooks: Any) -> Any:
    """The decoded JSON, or NOT_JSON when the text is not JSON or nests deeper than the parser allows
    (RecursionError). `hooks` go to `json.loads` (parse_int, parse_float, parse_constant)."""
    try:
        return json.loads(text, **hooks)
    except (ValueError, RecursionError):
        return NOT_JSON


# The members the SDK reads by name: at the top level of a reply, an envelope or a frame, and inside its error object.
# A member spelled in another letter case would be read by a case-insensitive decoder (Go's encoding/json) as the
# real one, so a body holding one can be read two ways and is never judged
_TOP_LEVEL = ('error', 'choices')
_INSIDE_ERROR = ('message', 'type', 'code', 'request_id')


def _has_case_variant(fields: Mapping[Any, Any], names: tuple[str, ...]) -> bool:
    # casefold, not lower: Go's strings.EqualFold also folds the long s (U+017F) to s, so `choice` + long s is `choices`
    # to it
    return any(isinstance(key, str) and key not in names and key.casefold() in names for key in fields)


def error_member(value: Any) -> dict[str, Any] | object | None:
    """What a decoded reply or frame says through its top-level `error` member: None when it has none (absent or
    null), the error object when it is one, and MALFORMED when it is anything else or a member the SDK reads comes in
    a case variant (`Error`, `CHOICES`, or `Code` inside the error object). Nested content (such as
    `choices[].delta`) is never inspected."""
    # [1] Only an object has members; a case variant at the top level fails closed whatever the error member holds
    if not isinstance(value, dict):
        return None
    top = cast('dict[str, Any]', value)
    if _has_case_variant(top, _TOP_LEVEL):
        return MALFORMED

    # [2] No error, a readable error object, or something that cannot be read as one
    error: Any = top.get('error')
    if error is None:
        return None
    if not isinstance(error, dict) or _has_case_variant(cast('dict[str, Any]', error), _INSIDE_ERROR):
        return MALFORMED
    return cast('dict[str, Any]', error)


def envelope_of(member: object) -> dict[str, Any] | None:
    """An error member (from error_member) as the engine's envelope: its string fields (and a `retryable` of false)
    when it has a string code and message; None for anything else."""
    if not isinstance(member, dict):
        return None
    fields = cast('dict[str, Any]', member)
    if not (isinstance(fields.get('code'), str) and isinstance(fields.get('message'), str)):
        return None
    kept: dict[str, Any] = {k: v for k, v in fields.items() if isinstance(v, str)}
    if fields.get('retryable') is False:
        kept['retryable'] = False
    return kept


def parse_envelope(text: str) -> dict[str, Any] | None:
    """The engine's `error` object, or None when the text is not Meridian's envelope."""
    return envelope_of(error_member(load_json(text)))


def error_from_envelope(error: Mapping[str, Any], status: int | None = None) -> MeridianError:
    # [1] A code this SDK does not know (a newer engine) is reported as internal, and its raw code kept in the message
    raw = error['code']
    known = raw in STATUS_BY_CODE
    code = raw if known else 'internal'
    message = error['message'] if known else f'{error["message"]} (engine code: {raw})'
    kind: str = error.get('type', 'api_error')

    # [2] Status from the response, or from the code when it arrived in a stream frame
    resolved = status if status is not None else STATUS_BY_CODE.get(code, 0)
    retryable = code in RETRYABLE and error.get('retryable') is not False
    return MeridianError(
        code,
        kind if kind in ENGINE_ERROR_TYPES else 'api_error',
        resolved,
        message,
        retryable,
        error.get('request_id'),
        None if known else raw,
    )


def error_from_body(status: int, header_request_id: str | None, text: str) -> MeridianError:
    # [1] Meridian's own envelope, with the header request id as fallback
    envelope = parse_envelope(text)
    if envelope is not None:
        error = error_from_envelope(envelope, status)
        return error if error.request_id or not header_request_id else replace(error, request_id=header_request_id)

    # [2] Anything else (a proxy page, an empty body) is a transport failure with the real status
    return MeridianError(
        'network_error',
        'client_error',
        status,
        f'Unexpected {status} response that is not a Meridian error.',
        status == 429 or status >= 500,
        header_request_id,
    )
