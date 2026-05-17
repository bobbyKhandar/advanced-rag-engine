"""Application entry point"""

import asyncio

from src.bot import configure_logging, create_application
from src.config.settings import settings


async def main() -> None:
    configure_logging(debug=settings.DEBUG)
    app = create_application(token=settings.TELEGRAM_TOKEN)
    await app.run_polling()


if __name__ == "__main__":
    asyncio.run(main())