"""Application entry point"""

from src.bot import configure_logging, create_application
from src.config.settings import settings
from src.rag.rag_processor import RagProcessor


def main() -> None:
    configure_logging(debug=settings.DEBUG)

    processor = RagProcessor()
    app = create_application(token=settings.TELEGRAM_TOKEN, processor=processor)
    app.run_polling()


if __name__ == "__main__":
    main()