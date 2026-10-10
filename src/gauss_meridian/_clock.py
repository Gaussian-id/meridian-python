"""The SDK's only clock. Tests replace these two functions; nothing else reads time for waiting."""

import asyncio
import time


def sleep(seconds: float) -> None:
    time.sleep(seconds)


async def asleep(seconds: float) -> None:
    await asyncio.sleep(seconds)
