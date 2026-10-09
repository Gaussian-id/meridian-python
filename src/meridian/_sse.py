from __future__ import annotations

import codecs
import re

_LINE_ENDINGS = re.compile(r'\r\n?')
_BOM = '﻿'

# One event may be this many characters (about 16 MiB of text) before the stream is failed closed.
MAX_EVENT_CHARS = 16 * 1024 * 1024


class EventTooLarge(ValueError):
    """A single event outgrew MAX_EVENT_CHARS: nothing more may be read from this stream."""


def _data_of(block: str) -> str | None:
    lines = [line for line in block.split('\n') if line.startswith('data:')]
    if not lines:
        return None
    return '\n'.join(line[6:] if line.startswith('data: ') else line[5:] for line in lines)


class EventDecoder:
    """Turns raw SSE bytes into the data of each event, across any split."""

    def __init__(self) -> None:
        self._utf8 = codecs.getincrementaldecoder('utf-8')('replace')
        self._started = False  # any text decoded yet: only a BOM at the very start is dropped
        self._held_cr = False  # a CR ended the last read: its LF may open the next one
        self._pending: list[str] = []  # text of the event being read, as pieces (no quadratic re-joining)
        self._size = 0
        self._last = ''  # last character of the pending text: a blank line may straddle two reads

    def _text(self, data: bytes, *, final: bool = False) -> str:
        # [1] Decode, keeping a split UTF-8 character for the next read, and drop one leading BOM
        text = self._utf8.decode(data, final)
        if text and not self._started:
            self._started = True
            text = text.removeprefix(_BOM)

        # [2] Line endings to LF, holding back a trailing CR that may start a CRLF split across reads
        if self._held_cr:
            text = '\r' + text
            self._held_cr = False
        if text.endswith('\r') and not final:
            self._held_cr = True
            text = text[:-1]
        return _LINE_ENDINGS.sub('\n', text)

    def _split(self, text: str) -> list[str]:
        # [1] No blank line in the new text (or straddling the last read): just keep it, searching nothing old
        if '\n\n' not in self._last + text:
            self._pending.append(text)
            self._size += len(text)
            self._last = (self._last + text)[-1:]
            self._check(self._size)
            return []

        # [2] Cut every complete event, searching only from where the old text could still start a blank line
        whole = ''.join(self._pending) + text
        start = max(self._size - 1, 0)
        events: list[str] = []
        begin = 0
        while (end := whole.find('\n\n', start)) != -1:
            self._check(end - begin)
            data_text = _data_of(whole[begin:end])
            if data_text is not None:
                events.append(data_text)
            begin = start = end + 2

        # [3] What follows the last blank line is the start of the next event
        rest = whole[begin:]
        self._check(len(rest))
        self._pending = [rest] if rest else []
        self._size = len(rest)
        self._last = rest[-1:]
        return events

    @staticmethod
    def _check(size: int) -> None:
        if size > MAX_EVENT_CHARS:
            raise EventTooLarge

    def feed(self, data: bytes) -> list[str]:
        return self._split(self._text(data))

    def close(self) -> list[str]:
        # [1] The last read may have held a CR or half a character back
        events = self._split(self._text(b'', final=True))

        # [2] A final event without its blank line still counts
        tail = ''.join(self._pending).strip('\n')
        self._pending, self._size, self._last = [], 0, ''
        data_text = _data_of(tail)
        return events if data_text is None else [*events, data_text]
