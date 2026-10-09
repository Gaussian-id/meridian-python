from __future__ import annotations

import codecs
import re

_LINE_ENDINGS = re.compile(r'\r\n?')


def _normalise(buffer: str) -> str:
    """Line endings to LF, holding back a trailing CR that may start a CRLF split across reads."""
    held = '\r' if buffer.endswith('\r') else ''
    body = buffer[:-1] if held else buffer
    return _LINE_ENDINGS.sub('\n', body) + held


def _data_of(block: str) -> str | None:
    lines = [line for line in block.split('\n') if line.startswith('data:')]
    if not lines:
        return None
    return '\n'.join(line[6:] if line.startswith('data: ') else line[5:] for line in lines)


class EventDecoder:
    """Turns raw SSE bytes into the data of each event, across any split."""

    def __init__(self) -> None:
        self._utf8 = codecs.getincrementaldecoder('utf-8')('replace')
        self._buffer = ''

    def feed(self, data: bytes) -> list[str]:
        # [1] Decode, keeping a split UTF-8 character for the next read
        self._buffer = _normalise(self._buffer + self._utf8.decode(data))

        # [2] Emit every complete event
        events: list[str] = []
        while (end := self._buffer.find('\n\n')) != -1:
            block, self._buffer = self._buffer[:end], self._buffer[end + 2 :]
            data_text = _data_of(block)
            if data_text is not None:
                events.append(data_text)
        return events

    def close(self) -> list[str]:
        # [3] A final event without its blank line still counts
        tail = _normalise(self._buffer + self._utf8.decode(b'', final=True)).rstrip('\r').strip()
        self._buffer = ''
        data_text = _data_of(tail)
        return [data_text] if data_text is not None else []
