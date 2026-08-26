"""One run's shared fetched material, and the single flight that fills it."""

import asyncio
from collections.abc import Awaitable, Callable

import pytest

from fact_checker.cache import RunCache


async def test_a_miss_calls_produce_and_a_hit_does_not() -> None:
    """The second caller on a resolved key gets the stored string."""
    cache = RunCache()
    calls = 0

    async def produce() -> str:
        nonlocal calls
        calls += 1
        return "page"

    assert await cache.get_or_call("scrape:https://example.test", produce) == "page"
    assert await cache.get_or_call("scrape:https://example.test", produce) == "page"
    assert calls == 1


async def test_a_different_key_calls_produce_again() -> None:
    """Nothing is shared between keys."""
    cache = RunCache()
    produced: list[str] = []

    def producer(value: str) -> Callable[[], Awaitable[str]]:
        async def produce() -> str:
            produced.append(value)
            return value

        return produce

    assert await cache.get_or_call("search:one", producer("one")) == "one"
    assert await cache.get_or_call("search:two", producer("two")) == "two"
    assert produced == ["one", "two"]


async def test_two_concurrent_callers_share_one_call() -> None:
    """The second caller awaits the flight already out, rather than starting one."""
    cache = RunCache()
    started = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def produce() -> str:
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()
        return "page"

    first = asyncio.create_task(cache.get_or_call("scrape:x", produce))
    await started.wait()
    second = asyncio.create_task(cache.get_or_call("scrape:x", produce))
    await asyncio.sleep(0)
    release.set()

    assert await asyncio.gather(first, second) == ["page", "page"]
    assert calls == 1


async def test_a_failing_call_reaches_every_waiter() -> None:
    """One failure, and both callers see it rather than one seeing nothing."""
    cache = RunCache()
    started = asyncio.Event()
    release = asyncio.Event()

    async def produce() -> str:
        started.set()
        await release.wait()
        raise RuntimeError("the server said no")

    first = asyncio.create_task(cache.get_or_call("scrape:x", produce))
    await started.wait()
    second = asyncio.create_task(cache.get_or_call("scrape:x", produce))
    await asyncio.sleep(0)
    release.set()

    outcomes = await asyncio.gather(first, second, return_exceptions=True)

    assert [str(outcome) for outcome in outcomes] == ["the server said no"] * 2


async def test_a_failing_call_leaves_the_key_free() -> None:
    """The cache stores what was fetched, so a failure blocks nobody later."""
    cache = RunCache()

    async def fail() -> str:
        raise RuntimeError("the server said no")

    async def succeed() -> str:
        return "page"

    with pytest.raises(RuntimeError):
        await cache.get_or_call("scrape:x", fail)

    assert await cache.get_or_call("scrape:x", succeed) == "page"


async def test_the_counters_report_what_happened() -> None:
    """Two misses on two keys, then two hits on the first of them."""
    cache = RunCache()

    async def produce() -> str:
        return "page"

    await cache.get_or_call("search:one", produce)
    await cache.get_or_call("search:two", produce)
    await cache.get_or_call("search:one", produce)
    await cache.get_or_call("search:one", produce)

    assert (cache.misses, cache.hits) == (2, 2)


async def test_a_failed_call_counts_as_a_miss_only_once() -> None:
    """A released key is a miss again, never a hit on something never stored."""
    cache = RunCache()

    async def fail() -> str:
        raise RuntimeError("the server said no")

    for _ in range(2):
        with pytest.raises(RuntimeError):
            await cache.get_or_call("scrape:x", fail)

    assert (cache.misses, cache.hits) == (2, 0)
