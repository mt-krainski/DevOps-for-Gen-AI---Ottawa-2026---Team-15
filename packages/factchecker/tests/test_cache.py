"""Tests for the run-scoped store of fetched material in `factchecker.cache`."""

from factchecker.cache import RunCache

QUERY = "does water boil at 100 C"
RESULT = "Water boils at 100 C at one atmosphere."
URL = "https://example.test/Boiling"
PAGE = "# Boiling\n\nWater boils at 100 C at one atmosphere."


def test_an_unrecorded_search_is_a_miss() -> None:
    """A fresh cache serves nothing, so the caller performs the call."""
    assert RunCache().search(QUERY) is None


def test_a_recorded_search_is_served_back() -> None:
    """The stored text comes back whole, so a second asker pays nothing."""
    cache = RunCache()

    cache.record_search(QUERY, RESULT)

    assert cache.search(QUERY) == RESULT


def test_case_and_whitespace_differences_hit_the_same_search_key() -> None:
    """Two agents phrasing one question alike are one search, not two."""
    cache = RunCache()

    cache.record_search(f"  {QUERY.upper()}\n", RESULT)

    assert cache.search(QUERY) == RESULT


def test_an_unrecorded_page_is_a_miss() -> None:
    """A fresh cache serves no page either."""
    assert RunCache().page(URL) is None


def test_a_recorded_page_is_served_back() -> None:
    """The stored markdown comes back whole."""
    cache = RunCache()

    cache.record_page(URL, PAGE)

    assert cache.page(URL) == PAGE


def test_a_page_key_is_the_url_exactly_as_given() -> None:
    """A URL's path is case-sensitive, so folding it would serve the wrong page."""
    cache = RunCache()

    cache.record_page(URL, PAGE)

    assert cache.page(URL.lower()) is None


def test_the_two_stores_do_not_reach_into_one_another() -> None:
    """A query that reads like a URL is still a query."""
    cache = RunCache()

    cache.record_search(URL, RESULT)

    assert cache.page(URL) is None
    assert cache.search(URL) == RESULT
