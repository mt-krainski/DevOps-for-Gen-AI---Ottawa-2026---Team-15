"""Tests for the surrounding-context window in `statement_classifier.context`."""

from statement_classifier.context import windows_for


def _numbered(count: int) -> str:
    """A text of `count` one-clause sentences, each naming its own position."""
    return " ".join(f"Sentence {i} stands here." for i in range(1, count + 1))


def test_window_holds_the_statement_and_five_sentences_either_side() -> None:
    """The published span: the statement's sentence, and five on each side."""
    text = _numbered(21)

    window = windows_for(text, ["Sentence 11 stands here."])[0]

    assert window == " ".join(f"Sentence {i} stands here." for i in range(6, 17))


def test_window_is_a_verbatim_slice_of_the_source() -> None:
    """Original spacing and punctuation survive, so the slice is the source."""
    text = "One.  Two.\n\nThree. Four. Five. Six. Seven. Eight. Nine. Ten. Eleven."

    window = windows_for(text, ["Three."])[0]

    assert window in text
    assert window != text
    assert window.startswith("One.  Two.\n\nThree.")


def test_window_at_the_start_takes_what_exists() -> None:
    """Near the start there are fewer than five sentences before it."""
    text = _numbered(21)

    window = windows_for(text, ["Sentence 2 stands here."])[0]

    assert window.startswith("Sentence 1 stands here.")
    assert window.endswith("Sentence 7 stands here.")


def test_window_at_the_end_takes_what_exists() -> None:
    """Near the end there are fewer than five sentences after it."""
    text = _numbered(21)

    window = windows_for(text, ["Sentence 20 stands here."])[0]

    assert window.startswith("Sentence 15 stands here.")
    assert window.endswith("Sentence 21 stands here.")


def test_six_sentences_carry_whole_from_either_edge() -> None:
    """Six is the width a window reaches from any position, so nothing is trimmed."""
    text = _numbered(6)

    assert windows_for(text, ["Sentence 1 stands here."]) == [text]
    assert windows_for(text, ["Sentence 6 stands here."]) == [text]


def test_a_seventh_sentence_puts_the_far_edge_out_of_reach() -> None:
    """Past six, a statement at one edge no longer reaches the other."""
    text = _numbered(7)

    assert windows_for(text, ["Sentence 1 stands here."])[0] != text


def test_a_repeated_clause_matches_the_next_one_not_the_first() -> None:
    """Statements arrive in reading order, so the scan runs forward."""
    text = _numbered(30).replace("Sentence 25 stands here.", "Sentence 3 stands here.")

    first, second = windows_for(
        text, ["Sentence 3 stands here.", "Sentence 3 stands here."]
    )

    assert "Sentence 1 stands here." in first
    assert "Sentence 24 stands here." in second
    assert "Sentence 1 stands here." not in second


def test_an_unfindable_statement_anchors_on_the_reading_cursor() -> None:
    """A paraphrased statement still gets a local window, never the whole text."""
    text = _numbered(40)

    windows = windows_for(
        text, ["Sentence 20 stands here.", "a paraphrase the source never held"]
    )

    assert "Sentence 21 stands here." in windows[1]
    assert windows[1] != text
    assert len(windows[1]) < len(text)


def test_one_window_per_statement_in_order() -> None:
    """Every statement gets its own window, and sits inside it."""
    text = _numbered(40)
    statements = [f"Sentence {i} stands here." for i in (5, 20, 35)]

    windows = windows_for(text, statements)

    assert len(windows) == 3
    for statement, window in zip(statements, windows, strict=True):
        assert statement in window


def test_no_statements_yields_no_windows() -> None:
    """Nothing to place, so nothing is returned."""
    assert windows_for(_numbered(12), []) == []


def test_consecutive_misses_walk_forward_instead_of_stacking() -> None:
    """A model that breaks verbatim wording throughout still gets local windows."""
    source = " ".join(f"He said \u201cclaim {i}\u201d was true." for i in range(1, 41))
    segments = [f'He said "claim {i}" was true.' for i in range(1, 41)]

    windows = windows_for(source, segments)

    assert len(set(windows)) > 1
    assert "claim 40" in windows[-1]
    assert "claim 1" not in windows[-1]


def test_clause_splits_do_not_outrun_the_text_on_a_miss_run() -> None:
    """Two statements to a sentence must not walk the cursor at twice reading pace."""
    source = " ".join(
        f"He said “claim {i}” was true, although he doubted “claim {i}”."
        for i in range(1, 21)
    )
    segments: list[str] = []
    for i in range(1, 21):
        segments.append(f'He said "claim {i}" was true,')
        segments.append(f'although he doubted "claim {i}".')

    windows = windows_for(source, segments)

    kept = sum(
        1
        for index, window in enumerate(windows)
        if f"\u201cclaim {index // 2 + 1}\u201d" in window
    )
    assert kept >= len(segments) - 1
