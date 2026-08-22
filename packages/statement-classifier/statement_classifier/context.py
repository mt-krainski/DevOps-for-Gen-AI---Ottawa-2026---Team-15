"""The window: the sentences around a statement, sliced from the source."""

from collections.abc import Sequence

import syntok.segmenter as segmenter

SENTENCES_EITHER_SIDE = 5

_Span = tuple[int, int]


def windows_for(text: str, statements: Sequence[str]) -> list[str]:
    """Give each statement the sentences around it, sliced from the source.

    The statements arrive in reading order, so each one is located by scanning
    forward from where the last one ended. A repeated clause therefore matches
    its own occurrence rather than the first one in the text.

    Args:
        text: The source the statements were split out of.
        statements: The statements, in the order they appear in the source.

    Returns:
        One window per statement, in the same order. Each window is a verbatim
        slice of `text`, so the source's own spacing and punctuation survive.
    """
    if not statements:
        return []

    spans = _sentence_spans(text)
    if not spans:
        return [text for _ in statements]

    windows: list[str] = []
    cursor = 0
    for statement in statements:
        match = text.find(statement, cursor)
        if match == -1:
            # The scan cannot match wording the model changed — a paraphrase, or
            # whitespace or quotes it reflowed. Reading order puts the statement
            # near the cursor, so window there and step the cursor on by what
            # this statement consumed. Stepping a whole sentence instead would
            # outrun text split into clauses, and stepping only by the length
            # would lag behind text split into sentences; a run of misses drifts
            # away from its own position either way.
            anchor = cursor
            cursor = _step_over(text, spans, cursor, len(statement))
        else:
            anchor = match
            cursor = match + len(statement)
        windows.append(_window(text, spans, anchor))
    return windows


def _sentence_spans(text: str) -> list[_Span]:
    """Locate every sentence in the text as a (start, end) character span."""
    spans: list[_Span] = []
    for paragraph in segmenter.analyze(text):
        for sentence in paragraph:
            last = sentence[-1]
            spans.append((sentence[0].offset, last.offset + len(last.value)))
    return spans


def _window(text: str, spans: Sequence[_Span], anchor: int) -> str:
    """Slice the sentence holding `anchor`, plus the ones either side of it."""
    index = _sentence_holding(spans, anchor)
    first = max(0, index - SENTENCES_EITHER_SIDE)
    last = min(len(spans) - 1, index + SENTENCES_EITHER_SIDE)
    return text[spans[first][0] : spans[last][1]]


def _sentence_holding(spans: Sequence[_Span], anchor: int) -> int:
    """The index of the sentence the anchor falls in, or the one after a gap."""
    for index, (_, end) in enumerate(spans):
        if anchor < end:
            return index
    return len(spans) - 1


def _next_sentence_start(text: str, spans: Sequence[_Span], position: int) -> int:
    """The start of the first sentence beginning after `position`.

    Past the last sentence there is nowhere further to walk, so this returns the
    end of the text. The result never moves backwards, which keeps it usable as
    the floor of the next search.
    """
    for start, _ in spans:
        if start > position:
            return start
    return max(position, len(text))


def _step_over(text: str, spans: Sequence[_Span], position: int, length: int) -> int:
    """Move `position` past a statement of `length` that could not be located.

    A statement that reaches the end of the sentence it started in leaves that
    sentence, so the next one starts at the next sentence. A shorter one is a
    clause, and the next statement starts where this one ended.
    """
    index = _sentence_holding(spans, position)
    if position + length >= spans[index][1]:
        return _next_sentence_start(text, spans, position)
    return position + length
