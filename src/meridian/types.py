"""Public request and reply types: aliases over the TypedDicts generated from the engine's documents.

A type checker sees the generated TypedDicts. At run time each name is a plain alias of `dict[str, Any]`, so
`from meridian.types import ChatParams` never raises outside `TYPE_CHECKING` (the generated module imports
`typing_extensions`, a development dependency, and is never imported at run time).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ._generated.public import (
        ChatCompletion,
        ChatCompletionChunk,
        ChatMessage,
        ModelList,
        Usage,
    )
    from ._generated.public import ChatCompletionRequest as ChatParams
else:
    ChatCompletion = ChatCompletionChunk = ChatMessage = ChatParams = ModelList = Usage = dict[str, Any]

__all__ = ['ChatCompletion', 'ChatCompletionChunk', 'ChatMessage', 'ChatParams', 'ModelList', 'Usage']
