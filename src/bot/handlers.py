import logging
import os

from telegram import Update
from telegram.ext import ContextTypes

from src.bot.base import BotHandler
from src.bot.ingest_queue import IngestItem
from src.bot.keyboards import MainMenuKeyboard
from src.bot.query_queue import QueryQueue

logger = logging.getLogger(__name__)


class StartHandler(BotHandler):

    async def handle(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        keyboard = MainMenuKeyboard().build()
        await update.message.reply_text(
            "Welcome to the RAG Engine Bot!\n\n"
            "I can answer questions based on bike manuals. "
            "Send me a question or use /help to learn more.",
            reply_markup=keyboard,
        )


class HelpHandler(BotHandler):

    async def handle(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await update.message.reply_text(
            "Available commands:\n"
            "/start - Show welcome message\n"
            "/help - Show this help message\n\n"
            "Just send me any question and I'll search the manuals for an answer."
        )


class QueryHandler(BotHandler):

    async def handle(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        queue: QueryQueue = context.application.bot_data["query_queue"]
        user = update.effective_user

        await update.message.reply_text("Processing your question...")

        try:
            response = await queue.enqueue(
                user_id=user.id,
                chat_id=update.effective_chat.id,
                text=update.message.text,
                message_id=update.message.message_id,
            )
            await update.message.reply_text(response)
        except NotImplementedError:
            await update.message.reply_text(
                "The engine isn't connected yet. This feature will be available soon."
            )
        except Exception:
            logger.exception("Failed to process query for user %s", user.id)
            await update.message.reply_text(
                "Sorry, I couldn't process your question right now. Please try again later."
            )


class DocumentUploadHandler(BotHandler):
    UPLOAD_DIR = os.path.join("data", "uploads")

    async def handle(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        document = update.effective_message.document
        os.makedirs(self.UPLOAD_DIR, exist_ok=True)

        file = await context.bot.get_file(document.file_id)
        dest = os.path.join(self.UPLOAD_DIR, f"{document.file_id}.pdf")
        await file.download_to_drive(dest)

        queue = context.application.bot_data["ingest_queue"]
        item = IngestItem(
            file_path=dest,
            original_name=document.file_name or "document.pdf",
            chat_id=update.effective_chat.id,
            user_id=update.effective_user.id,
        )
        await queue.enqueue(item)

        await update.message.reply_text("Got it, processing...")