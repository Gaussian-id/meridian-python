from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, Protocol, TypeVar, cast

from ._errors import Err, Ok, Result
from ._stream import AsyncChatStream, ChatStream, token_counts
from ._transport import Call, new_idempotency_key

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


def idempotency(key: str | None) -> dict[str, str]:
    return {'idempotency-key': key or new_idempotency_key()}


def chat_call(params: Mapping[str, Any], idempotency_key: str | None) -> Call:
    return Call(
        'POST', CHAT_PATH, body=chat_body(params, False), headers=idempotency(idempotency_key), object_body=True
    )


def stream_call(params: Mapping[str, Any], idempotency_key: str | None) -> Call:
    # [1] chat_body asks for the usage chunk, so stream.usage is filled (the engine sends it only on include_usage)
    return Call('POST', CHAT_PATH, body=chat_body(params, True), headers=idempotency(idempotency_key))


def stream_result(opened: Result[Any], make: Callable[[Any], T]) -> Result[T]:
    # [2] A refusal before any byte is an error value; otherwise the stream
    return Err(opened.error) if opened.error is not None else Ok(make(opened.data))


def with_counts(result: Result[Any]) -> Result[Any]:
    if result.error is not None:
        return result
    data = dict(cast('dict[str, Any]', result.data))  # the transport guarantees an object (Call.object_body)
    data['usage'] = token_counts(data.get('usage'))
    return Ok(data)


class ChatApi:
    def __init__(self, transport: JsonTransport) -> None:
        self._transport = transport

    def create(self, params: ChatParams, *, idempotency_key: str | None = None) -> Result[ChatCompletion]:
        return with_counts(self._transport.json(chat_call(params, idempotency_key)))

    def stream(self, params: ChatParams, *, idempotency_key: str | None = None) -> Result[ChatStream]:
        return stream_result(self._transport.open(stream_call(params, idempotency_key)), ChatStream)


class ModelsApi:
    def __init__(self, transport: JsonTransport) -> None:
        self._transport = transport

    def list(self) -> Result[ModelList]:
        return self._transport.json(MODELS_CALL)


class AsyncChatApi:
    def __init__(self, transport: AsyncJsonTransport) -> None:
        self._transport = transport

    async def create(self, params: ChatParams, *, idempotency_key: str | None = None) -> Result[ChatCompletion]:
        return with_counts(await self._transport.json(chat_call(params, idempotency_key)))

    async def stream(self, params: ChatParams, *, idempotency_key: str | None = None) -> Result[AsyncChatStream]:
        return stream_result(await self._transport.open(stream_call(params, idempotency_key)), AsyncChatStream)


class AsyncModelsApi:
    def __init__(self, transport: AsyncJsonTransport) -> None:
        self._transport = transport

    async def list(self) -> Result[ModelList]:
        return await self._transport.json(MODELS_CALL)
