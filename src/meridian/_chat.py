from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, cast

from ._errors import Err, MeridianError, Ok, Result, config_error
from ._stream import AsyncChatStream, ChatStream, token_counts
from ._transport import Call, new_idempotency_key, valid_idempotency_key

if TYPE_CHECKING:
    from .types import ChatCompletion, ChatParams, ModelList

T = TypeVar('T')
CHAT_PATH = '/v1/chat/completions'
MODELS_CALL = Call('GET', '/v1/models')


class JsonTransport(Protocol):
    def json(self, call: Call) -> Result[Any]: ...

    def open(self, call: Call) -> Result[Any]: ...


class AsyncJsonTransport(Protocol):
    async def json(self, call: Call) -> Result[Any]: ...

    async def open(self, call: Call) -> Result[Any]: ...


def chat_body(params: Mapping[str, Any], stream: bool) -> dict[str, Any]:
    # The SDK owns stream and stream_options: set after the caller's fields, so the caller can never flip them
    body = dict(params)
    body['stream'] = stream
    if stream:
        body['stream_options'] = {'include_usage': True}
    else:
        body.pop('stream_options', None)
    return body


_BAD_KEY = config_error('idempotency_key must be 1-255 visible ASCII characters (no space or control character).')


def idempotency(key: str | None) -> str | MeridianError:
    """The call's key: the caller's when the engine would accept it, a new one for None or '', else a refusal."""
    if key is None or key == '':
        return new_idempotency_key()
    return key if valid_idempotency_key(key) else _BAD_KEY


def chat_call(params: Mapping[str, Any], idempotency_key: str | None) -> Call | MeridianError:
    key = idempotency(idempotency_key)
    if isinstance(key, MeridianError):
        return key
    return Call('POST', CHAT_PATH, body=chat_body(params, False), idempotency_key=key, object_body=True)


def stream_call(params: Mapping[str, Any], idempotency_key: str | None) -> Call | MeridianError:
    # [1] chat_body asks for the usage chunk, so stream.usage is filled (the engine sends it only on include_usage)
    key = idempotency(idempotency_key)
    if isinstance(key, MeridianError):
        return key
    return Call('POST', CHAT_PATH, body=chat_body(params, True), idempotency_key=key)


def stream_result(opened: Result[Any], make: Callable[[Any], T]) -> Result[T]:
    # [2] A refusal before any byte is an error value; otherwise the stream
    return Err(opened.error) if opened.error is not None else Ok(make(opened.data))


def with_counts(result: Result[Any]) -> Result[Any]:
    if result.error is not None:
        return result
    data = dict(cast('dict[str, Any]', result.data))  # the transport guarantees an object (Call.object_body)
    data['usage'] = token_counts(data.get('usage'))
    return Ok(data, request_id=result.request_id)


class ChatApi:
    def __init__(self, transport: JsonTransport) -> None:
        self._transport = transport

    def create(self, params: ChatParams, *, idempotency_key: str | None = None) -> Result[ChatCompletion]:
        """`idempotency_key`: 1-255 visible ASCII characters, reused across this call's retries; None or '' makes one.
        Any other key is refused (`invalid_request`, nothing sent)."""
        call = chat_call(params, idempotency_key)
        if isinstance(call, MeridianError):
            return Err(call)
        return with_counts(self._transport.json(call))

    def stream(self, params: ChatParams, *, idempotency_key: str | None = None) -> Result[ChatStream]:
        call = stream_call(params, idempotency_key)
        if isinstance(call, MeridianError):
            return Err(call)
        return stream_result(self._transport.open(call), ChatStream)


class ModelsApi:
    def __init__(self, transport: JsonTransport) -> None:
        self._transport = transport

    def list(self) -> Result[ModelList]:
        return self._transport.json(MODELS_CALL)


class AsyncChatApi:
    def __init__(self, transport: AsyncJsonTransport) -> None:
        self._transport = transport

    async def create(self, params: ChatParams, *, idempotency_key: str | None = None) -> Result[ChatCompletion]:
        call = chat_call(params, idempotency_key)
        if isinstance(call, MeridianError):
            return Err(call)
        return with_counts(await self._transport.json(call))

    async def stream(self, params: ChatParams, *, idempotency_key: str | None = None) -> Result[AsyncChatStream]:
        call = stream_call(params, idempotency_key)
        if isinstance(call, MeridianError):
            return Err(call)
        return stream_result(await self._transport.open(call), AsyncChatStream)


class AsyncModelsApi:
    def __init__(self, transport: AsyncJsonTransport) -> None:
        self._transport = transport

    async def list(self) -> Result[ModelList]:
        return await self._transport.json(MODELS_CALL)
