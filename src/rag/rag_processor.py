import asyncio
import logging
import os

import yaml

from src.bot.query_queue import Processor
from src.rag.vector_store import VectorStore

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "config.yaml"
)


class RagProcessor(Processor):

    def __init__(self, config_path: str = _DEFAULT_CONFIG_PATH) -> None:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        rag_cfg = config.get("rag", {})
        vs_cfg = config.get("vector_store", {})
        embedding_cfg = rag_cfg.get("embedding", {})

        self.top_k: int = rag_cfg.get("top_k", 5)

        import dotenv

        dotenv.load_dotenv()

        self._vector_store = VectorStore(
            collection=vs_cfg.get("collection", "documents"),
            url=self._resolve(vs_cfg.get("url", "")),
            api_key=self._resolve(vs_cfg.get("api_key", "")),
            model_name=embedding_cfg.get("model", "all-MiniLM-L6-v2"),
            recreate=False,
        )

    @staticmethod
    def _resolve(value: str) -> str:
        import re

        return re.sub(r"\$\{(\w+)\}", lambda m: os.environ.get(m.group(1), ""), value)

    async def process(self, text: str) -> str:
        docs = await asyncio.to_thread(
            self._vector_store.similarity_search, text, k=self.top_k
        )
        if not docs:
            return "I couldn't find any relevant information in the bike manuals."

        parts: list[str] = []
        for i, doc in enumerate(docs, 1):
            score = doc.metadata.get("score", 0.0)
            source = doc.metadata.get("source", "unknown")
            content = doc.page_content[:300].strip()
            parts.append(f"{i}. [{score:.2f}] ({source})\n{content}\n")

        return "\n".join(parts)
