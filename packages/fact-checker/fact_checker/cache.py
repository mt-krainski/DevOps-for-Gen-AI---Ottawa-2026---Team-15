"""One run's shared fetched material, fetched once however many statements want it."""

import asyncio
from collections.abc import Awaitable, Callable
from functools import partial


class RunCache:
    """What one run has fetched, keyed by search arguments and by scraped URL.

    Statements drawn from one document search for overlapping things, so the
    cache spares the second and third statement a call the first already paid
    for. It stores what was fetched, never a failure and never any reasoning.
    """

    def __init__(self) -> None:
        """Start empty, with both counters at zero."""
        self.hits = 0
        self.misses = 0
        self._calls: dict[str, asyncio.Task[str]] = {}

    async def get_or_call(self, key: str, produce: Callable[[], Awaitable[str]]) -> str:
        """Return the material for `key`, calling `produce` only where it is new.

        Args:
            key: What names the material: the search arguments, or the URL.
            produce: How to fetch it, called at most once for a resolved key.

        Returns:
            The fetched string, whether this call fetched it or another did.

        Raises:
            Exception: Whatever `produce` raised, to this caller and to every
                other caller waiting on the same flight.
        """
        flight = self._calls.get(key)
        if flight is None:
            self.misses += 1
            flight = asyncio.create_task(produce())
            self._calls[key] = flight
            flight.add_done_callback(partial(self._release_unless_fetched, key))
        else:
            self.hits += 1
        # A statement that times out cancels its own work. Shielding keeps that
        # cancellation off the shared flight, which the other statements are
        # still waiting on.
        return await asyncio.shield(flight)

    def _release_unless_fetched(self, key: str, flight: asyncio.Task[str]) -> None:
        if flight.cancelled() or flight.exception() is not None:
            self._calls.pop(key, None)
