"""What one run has already fetched from the web, so it fetches it once."""


class RunCache:
    """One run's fetched material, shared by every statement the run checks.

    The cache holds what the web returned and nothing a model produced. Two
    statements that read one page still reach independent verdicts, because no
    ruling is stored here.

    Nothing in this class awaits, so a read and the write that follows it cannot be
    interleaved by another statement's task. That is what lets one instance serve
    every statement of a run at once without a lock.
    """

    def __init__(self) -> None:
        """Start empty, with the two stores kept apart."""
        self._searches: dict[str, str] = {}
        self._pages: dict[str, str] = {}

    def search(self, query: str) -> str | None:
        """Return what an earlier identical search returned.

        Args:
            query: The question asked of the search engine.

        Returns:
            The stored result, or `None` where this run has not run the search.
        """
        return self._searches.get(_search_key(query))

    def record_search(self, query: str, result: str) -> None:
        """Store what a search returned.

        Args:
            query: The question asked of the search engine.
            result: What the search engine returned, as text.
        """
        self._searches[_search_key(query)] = result

    def page(self, url: str) -> str | None:
        """Return what an earlier fetch of this URL returned.

        Args:
            url: The page's address.

        Returns:
            The stored markdown, or `None` where this run has not fetched the page.
        """
        return self._pages.get(url)

    def record_page(self, url: str, markdown: str) -> None:
        """Store a fetched page as fetched, before any ceiling is applied.

        Args:
            url: The page's address.
            markdown: The page as markdown, whole.
        """
        self._pages[url] = markdown


def _search_key(query: str) -> str:
    """Fold a query to the key two askers of one question share."""
    return query.strip().casefold()
