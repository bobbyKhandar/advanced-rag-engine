import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class QueryItem:
    user_id: int
    chat_id: int
    text: str
    message_id: int
    response_future: asyncio.Future = field(compare=False, default=None)

    def __post_init__(self):
        if self.response_future is None:
            loop = asyncio.get_running_loop()
            self.response_future = loop.create_future()


class Processor(ABC):

    @abstractmethod
    async def process(self, text: str) -> str:
        ...


class QueryQueue:

    def __init__(self, processor: Optional[Processor] = None, max_concurrency: int = 1):
        self._queue: asyncio.Queue[QueryItem] = asyncio.Queue()
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._processor = processor
        self._worker_task: Optional[asyncio.Task] = None
        self._running = False

    @property
    def processor(self) -> Optional[Processor]:
        return self._processor

    @processor.setter
    def processor(self, value: Processor) -> None:
        self._processor = value

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()

    async def enqueue(self, user_id: int, chat_id: int, text: str, message_id: int) -> str:
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        item = QueryItem(
            user_id=user_id,
            chat_id=chat_id,
            text=text,
            message_id=message_id,
            response_future=future,
        )
        await self._queue.put(item)
        return await future

    async def _worker(self) -> None:
        while self._running:
            item = await self._queue.get()
            async with self._semaphore:
                try:
                    if self._processor is None:
                        raise RuntimeError("No processor set on QueryQueue")
                    logger.debug("Processing query from user %s: %.50s", item.user_id, item.text)
                    result = await self._processor.process(item.text)
                    if not item.response_future.done():
                        item.response_future.set_result(result)
                except Exception as exc:
                    logger.exception("Query processing failed for user %s", item.user_id)
                    if not item.response_future.done():
                        item.response_future.set_exception(exc)
                finally:
                    self._queue.task_done()

    def start(self) -> None:
        if self._running:
            logger.warning("QueryQueue worker already running")
            return
        self._running = True
        self._worker_task = asyncio.create_task(self._worker())
        logger.info("QueryQueue worker started")

    async def stop(self) -> None:
        self._running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        logger.info("QueryQueue worker stopped")