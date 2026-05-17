import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

from telegram import Bot

from src.rag.ingestor import Ingestor, IngestResult

logger = logging.getLogger(__name__)


@dataclass
class IngestItem:
    file_path: str
    original_name: str
    chat_id: int
    user_id: int


class IngestQueue:

    def __init__(self, bot: Bot, max_concurrency: int = 1) -> None:
        self._bot = bot
        self._queue: asyncio.Queue[IngestItem] = asyncio.Queue()
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._ingestor = Ingestor()
        self._worker_task: Optional[asyncio.Task] = None
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()

    async def enqueue(self, item: IngestItem) -> None:
        await self._queue.put(item)

    async def _worker(self) -> None:
        while self._running:
            item = await self._queue.get()
            async with self._semaphore:
                try:
                    logger.info("Processing ingest: %s", item.original_name)
                    result = self._ingestor.process_document(
                        item.file_path, item.original_name
                    )
                    if result.success:
                        msg = (
                            f"Done: {item.original_name} ingested "
                            f"({result.chunks} chunks)"
                        )
                    else:
                        msg = f"Failed: {item.original_name} - {result.error}"
                    await self._bot.send_message(
                        chat_id=item.chat_id, text=msg
                    )
                except Exception as exc:
                    logger.exception(
                        "Ingest failed for %s", item.original_name
                    )
                    try:
                        await self._bot.send_message(
                            chat_id=item.chat_id,
                            text=f"Failed: {item.original_name} - {exc}",
                        )
                    except Exception:
                        logger.exception(
                            "Failed to send error message for %s",
                            item.original_name,
                        )
                finally:
                    self._queue.task_done()

    def start(self) -> None:
        if self._running:
            logger.warning("IngestQueue worker already running")
            return
        self._running = True
        self._worker_task = asyncio.create_task(self._worker())
        logger.info("IngestQueue worker started")

    async def stop(self) -> None:
        self._running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        logger.info("IngestQueue worker stopped")
