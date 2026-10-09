"""Public request and reply types: aliases over the TypedDicts generated from the engine's documents.

They exist for type checkers only (the generated module imports `typing_extensions`, a development dependency),
so import them under `if TYPE_CHECKING:` or with `from __future__ import annotations`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._generated.public import (
        ChatCompletion,
        ChatCompletionChunk,
        ChatMessage,
        ModelList,
        Usage,
    )
    from ._generated.public import ChatCompletionRequest as ChatParams

    __all__ = ['ChatCompletion', 'ChatCompletionChunk', 'ChatMessage', 'ChatParams', 'ModelList', 'Usage']
