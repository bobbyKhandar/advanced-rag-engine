import asyncio
import logging
import os

import yaml

from src.bot.query_queue import Processor
from src.rag.generator import Generator
from src.rag.retriever import SmallToBigRetriever
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
        llm_cfg = rag_cfg.get("llm", {})

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
        self._retriever = SmallToBigRetriever(self._vector_store)

        self._generator = Generator(
            api_key=os.environ.get("GROQ_API_KEY", ""),
            model=llm_cfg.get("model", "llama-3.3-70b-versatile"),
            base_url=llm_cfg.get("base_url", "https://api.groq.com/openai/v1"),
            temperature=llm_cfg.get("temperature", 0.3),
        )

    @staticmethod
    def _resolve(value: str) -> str:
        import re

        return re.sub(r"\$\{(\w+)\}", lambda m: os.environ.get(m.group(1), ""), value)

    async def process(self, text: str) -> str:
        docs = await asyncio.to_thread(
            self._retriever.retrieve, text, k=self.top_k
        )
        if not docs:
            return "I couldn't find any relevant information in the bike manuals."

        context_parts: list[str] = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "unknown")
            chapter = doc.metadata.get("chapter", "")
            section = doc.metadata.get("section", "")
            heading = f"{chapter} / {section}".strip(" /")
            context_parts.append(
                f"[Source {i}: {source} {heading}]\n{doc.page_content}\n"
            )

        context = "\n".join(context_parts)
        answer = await asyncio.to_thread(
            self._generator.generate, text, context
        )
        return answer
