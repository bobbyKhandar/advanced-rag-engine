from typing import Any

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup

from src.bot.base import KeyboardBuilder


class MainMenuKeyboard(KeyboardBuilder):

    def build(self) -> Any:
        return ReplyKeyboardMarkup(
            keyboard=[["Ask a question", "Help"]],
            resize_keyboard=True,
        )


class ConfirmKeyboard(KeyboardBuilder):

    def build(self) -> Any:
        return InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(text="Yes", callback_data="confirm_yes"),
                    InlineKeyboardButton(text="No", callback_data="confirm_no"),
                ]
            ]
        )