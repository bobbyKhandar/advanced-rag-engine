import logging
from abc import ABC, abstractmethod
from typing import Any

from telegram import Update
from telegram.ext import ContextTypes

logger = logging.getLogger(__name__)


class BotHandler(ABC):

    async def __call__(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        try:
            await self.handle(update, context)
        except Exception:
            logger.exception("Unhandled error in %s", self.__class__.__name__)
            if update and update.effective_chat:
                await update.effective_chat.send_message(
                    "An unexpected error occurred. Please try again."
                )

    @abstractmethod
    async def handle(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        ...


class KeyboardBuilder(ABC):

    @abstractmethod
    def build(self) -> Any:
        ...