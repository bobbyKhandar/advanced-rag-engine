"""Bot unit tests"""

import asyncio
from abc import ABC
from unittest.mock import AsyncMock, MagicMock

import pytest
from telegram import Bot, Update
from telegram.ext import ContextTypes

from src.bot.base import BotHandler
from src.bot.query_queue import Processor, QueryQueue, QueryItem
from src.rag.ingestor import Ingestor


# ── Mock Processors ──────────────────────────────────────────────────


class EchoProcessor(Processor):
    async def process(self, text: str) -> str:
        return f"Processed: {text}"


class DelayedProcessor(Processor):
    def __init__(self, delay: float = 0.1):
        self.delay = delay

    async def process(self, text: str) -> str:
        await asyncio.sleep(self.delay)
        return f"Result: {text}"


# ── Queue Lifecycle ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_queue_lifecycle():
    queue = QueryQueue(processor=EchoProcessor())
    assert not queue.is_running
    assert queue.queue_size == 0

    queue.start()
    assert queue.is_running

    await queue.stop()
    assert not queue.is_running


# ── Basic Enqueue / Dequeue ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_enqueue_and_process():
    queue = QueryQueue(processor=EchoProcessor())
    queue.start()

    result = await queue.enqueue(user_id=1, chat_id=10, text="hello", message_id=100)
    assert result == "Processed: hello"

    await queue.stop()


# ── Multiple Items in Order ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_multiple_items_order():
    queue = QueryQueue(processor=DelayedProcessor(delay=0.05))
    queue.start()

    r1 = await queue.enqueue(user_id=1, chat_id=10, text="first", message_id=100)
    r2 = await queue.enqueue(user_id=2, chat_id=20, text="second", message_id=101)

    assert r1 == "Result: first"
    assert r2 == "Result: second"

    await queue.stop()


# ── Concurrent Users ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_concurrent_users():
    queue = QueryQueue(processor=DelayedProcessor(delay=0.1), max_concurrency=3)
    queue.start()

    async def user_query(uid: int, text: str) -> str:
        return await queue.enqueue(
            user_id=uid, chat_id=uid * 10, text=text, message_id=uid
        )

    results = await asyncio.gather(
        user_query(1, "A"),
        user_query(2, "B"),
        user_query(3, "C"),
    )

    assert results == ["Result: A", "Result: B", "Result: C"]
    await queue.stop()


# ── Queue Size Tracking ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_queue_size():
    processor = DelayedProcessor(delay=0.3)
    queue = QueryQueue(processor=processor, max_concurrency=1)
    queue.start()

    task1 = asyncio.create_task(
        queue.enqueue(user_id=1, chat_id=10, text="first", message_id=100)
    )
    await asyncio.sleep(0.05)

    task2 = asyncio.create_task(
        queue.enqueue(user_id=2, chat_id=20, text="second", message_id=101)
    )
    await asyncio.sleep(0.05)

    assert queue.queue_size == 1

    await task1
    await task2
    await queue.stop()


# ── No Processor Error ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_no_processor_error():
    queue = QueryQueue()
    queue.start()

    with pytest.raises(RuntimeError, match="No processor set"):
        await queue.enqueue(user_id=1, chat_id=10, text="test", message_id=100)

    await queue.stop()


# ── Processor ABC Cannot Be Instantiated ─────────────────────────────


def test_processor_is_abstract():
    assert issubclass(Processor, ABC)
    with pytest.raises(TypeError, match="abstract"):
        Processor()


# ── QueryItem Future Lifecycle ───────────────────────────────────────


@pytest.mark.asyncio
async def test_query_item_future():
    loop = asyncio.get_running_loop()
    future = loop.create_future()

    item = QueryItem(
        user_id=1,
        chat_id=10,
        text="test",
        message_id=100,
        response_future=future,
    )

    assert item.user_id == 1
    assert item.chat_id == 10
    assert item.text == "test"
    assert item.message_id == 100
    assert item.response_future is future
    assert not item.response_future.done()

    item.response_future.set_result("done")
    assert item.response_future.done()
    assert await item.response_future == "done"


# ── BotHandler Error Boundary ────────────────────────────────────────


@pytest.mark.asyncio
async def test_handler_error_boundary():
    mock_update = MagicMock(spec=Update)
    mock_update.effective_chat.send_message = AsyncMock()
    mock_context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)

    class FailingHandler(BotHandler):
        async def handle(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            raise ValueError("something broke")

    handler = FailingHandler()
    await handler(mock_update, mock_context)

    mock_update.effective_chat.send_message.assert_awaited_once_with(
        "An unexpected error occurred. Please try again."
    )


# ── IngestItem dataclass ──────────────────────────────────────────────


class TestIngestItem:

    def test_fields(self):
        from src.bot.ingest_queue import IngestItem

        item = IngestItem(
            file_path="/tmp/doc.pdf",
            original_name="doc.pdf",
            chat_id=123,
            user_id=456,
        )
        assert item.file_path == "/tmp/doc.pdf"
        assert item.original_name == "doc.pdf"
        assert item.chat_id == 123
        assert item.user_id == 456


# ── IngestQueue ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_ingest_queue_lifecycle():
    from src.bot.ingest_queue import IngestQueue

    bot = MagicMock(spec=Bot)
    bot.send_message = AsyncMock()

    queue = IngestQueue(bot=bot)
    assert not queue.is_running
    assert queue.queue_size == 0

    queue.start()
    assert queue.is_running

    await queue.stop()
    assert not queue.is_running


@pytest.mark.asyncio
async def test_ingest_queue_enqueue():
    from src.bot.ingest_queue import IngestItem, IngestQueue

    bot = MagicMock(spec=Bot)
    bot.send_message = AsyncMock()

    queue = IngestQueue(bot=bot)
    queue.start()

    item = IngestItem(
        file_path="/tmp/test.pdf",
        original_name="test.pdf",
        chat_id=100,
        user_id=1,
    )
    await queue.enqueue(item)

    # Give the worker a moment to process
    await asyncio.sleep(0.1)

    await queue.stop()

    bot.send_message.assert_awaited_once()
    _args, kwargs = bot.send_message.call_args
    assert kwargs["chat_id"] == 100
    assert "test.pdf" in kwargs["text"]


@pytest.mark.asyncio
async def test_ingest_queue_enqueue_failure():
    from src.bot.ingest_queue import IngestItem, IngestQueue

    bot = MagicMock(spec=Bot)
    bot.send_message = AsyncMock()

    queue = IngestQueue(bot=bot)
    queue.start()

    # Monkey-patch the ingestor to raise
    from src.rag.ingestor import Ingestor
    original = Ingestor.process_document

    def _fail(*args, **kwargs):
        raise RuntimeError("oops")

    Ingestor.process_document = _fail

    item = IngestItem(
        file_path="/tmp/bad.pdf",
        original_name="bad.pdf",
        chat_id=200,
        user_id=2,
    )
    await queue.enqueue(item)
    await asyncio.sleep(0.1)

    Ingestor.process_document = original
    await queue.stop()

    # Should have sent an error message
    bot.send_message.assert_awaited_once()
    _args, kwargs = bot.send_message.call_args
    assert kwargs["chat_id"] == 200
    assert "bad.pdf" in kwargs["text"]
    assert "oops" in kwargs["text"]


@pytest.mark.asyncio
async def test_ingest_queue_size():
    from src.bot.ingest_queue import IngestItem, IngestQueue

    bot = MagicMock(spec=Bot)
    bot.send_message = AsyncMock()

    queue = IngestQueue(bot=bot, max_concurrency=1)

    # Enqueue before starting — worker hasn't consumed anything yet
    item1 = IngestItem(file_path="/a/1.pdf", original_name="1.pdf", chat_id=1, user_id=1)
    item2 = IngestItem(file_path="/a/2.pdf", original_name="2.pdf", chat_id=2, user_id=2)

    await queue.enqueue(item1)
    await queue.enqueue(item2)

    assert queue.queue_size == 2

    # Start the worker and let it drain
    queue.start()
    await asyncio.sleep(0.1)

    assert queue.queue_size == 0
    await queue.stop()