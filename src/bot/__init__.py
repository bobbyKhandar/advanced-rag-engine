import logging
from typing import Optional

from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
)

from src.bot.base import BotHandler, KeyboardBuilder
from src.bot.handlers import HelpHandler, QueryHandler, StartHandler
from src.bot.keyboards import ConfirmKeyboard, MainMenuKeyboard
from src.bot.query_queue import Processor, QueryItem, QueryQueue
from src.bot.states import ConversationState

logger = logging.getLogger(__name__)


def configure_logging(debug: bool = False) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s",
        level=level,
    )


async def _post_init(app: Application) -> None:
    queue: QueryQueue = app.bot_data["query_queue"]
    queue.start()
    logger.info("Application initialized")


async def _post_stop(app: Application) -> None:
    queue: QueryQueue = app.bot_data["query_queue"]
    await queue.stop()
    logger.info("Application stopped")


def create_application(token: str, processor: Optional[Processor] = None) -> Application:
    app = (
        Application.builder()
        .token(token)
        .post_init(_post_init)
        .post_stop(_post_stop)
        .build()
    )

    app.bot_data["query_queue"] = QueryQueue(processor=processor)

    app.add_handler(CommandHandler("start", StartHandler()))
    app.add_handler(CommandHandler("help", HelpHandler()))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, QueryHandler()))

    logger.info("Application created with %d handlers", 3)
    return app


__all__ = [
    "configure_logging",
    "create_application",
    "BotHandler",
    "KeyboardBuilder",
    "ConversationState",
    "Processor",
    "QueryItem",
    "QueryQueue",
    "MainMenuKeyboard",
    "ConfirmKeyboard",
]