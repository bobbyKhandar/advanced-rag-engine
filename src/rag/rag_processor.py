import asyncio
import logging
import os

import yaml

from src.bot.query_queue import Processor
from src.config.prompts import RAG_PROMPT_TEMPLATE, RAG_SYSTEM_PROMPT
from src.rag.bike_meta import decompose_query
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
        groq_api_key = os.environ.get("GROQ_API_KEY", "")

        self._vector_store = VectorStore(
            collection=vs_cfg.get("collection", "documents"),
            url=self._resolve(vs_cfg.get("url", "")),
            api_key=self._resolve(vs_cfg.get("api_key", "")),
            model_name=embedding_cfg.get("model", "all-MiniLM-L6-v2"),
            recreate=False,
        )
        self._retriever = SmallToBigRetriever(self._vector_store)

        self._groq_api_key = groq_api_key
        self._llm_model = llm_cfg.get("model", "llama-3.3-70b-versatile")
        self._llm_base_url = llm_cfg.get(
            "base_url", "https://api.groq.com/openai/v1"
        )
        self._llm_temperature = llm_cfg.get("temperature", 0.3)

        self._generator = Generator(
            api_key=groq_api_key,
            model=self._llm_model,
            base_url=self._llm_base_url,
            temperature=self._llm_temperature,
        )

    @staticmethod
    def _resolve(value: str) -> str:
        import re

        return re.sub(r"\$\{(\w+)\}", lambda m: os.environ.get(m.group(1), ""), value)

    async def process(self, text: str) -> str:
        # Step 1: Decompose query — extract bike model + clean question
        decomposed = await asyncio.to_thread(
            decompose_query,
            text,
            api_key=self._groq_api_key,
            base_url=self._llm_base_url,
            model=self._llm_model,
        )
        bike_model = decomposed.get("model") or None
        clean_question = decomposed.get("question", text)

        # Step 2: Retrieve — try with bike model filter, fall back to unfiltered
        docs = await asyncio.to_thread(
            self._retriever.retrieve, clean_question, k=self.top_k, bike_model=bike_model
        )
        if not docs and bike_model:
            logger.info(
                "No results with bike_model='%s', retrying unfiltered", bike_model
            )
            docs = await asyncio.to_thread(
                self._retriever.retrieve, clean_question, k=self.top_k
            )

        if not docs:
            return "I couldn't find any relevant information in the bike manuals."

        # Step 3: Build context with bike metadata
        context_parts: list[str] = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "unknown")
            chapter = doc.metadata.get("chapter", "")
            section = doc.metadata.get("section", "")
            doc_bike = doc.metadata.get("bike_full_name", "")
            heading = f"{chapter} / {section}".strip(" /")
            bike_tag = f" [{doc_bike}]" if doc_bike else ""
            context_parts.append(
                f"[Source {i}: {source}{bike_tag} {heading}]\n{doc.page_content}\n"
            )

        context = "\n".join(context_parts)

        # Step 4: Generate answer with updated system prompt
        user_bike = bike_model or "unknown model"
        context_bikes = ", ".join(
            sorted(
                {
                    d.metadata.get("bike_full_name", "unknown")
                    for d in docs
                    if d.metadata.get("bike_full_name")
                }
            )
        ) or "unknown manual"

        answer = await asyncio.to_thread(
            self._generator.generate,
            question=clean_question,
            context=context,
            system_prompt=(
                "You are answering questions about motorcycle repair manuals. "
                "The user asked about **{user_bike}**. "
                "The retrieved context is from: **{context_bikes}**. "
                "Use your judgment:\n"
                "- Allow partial/fuzzy model name matches "
                "(e.g., 'ktm 125' may refer to 'KTM RC 125').\n"
                "- If a procedure is shared across platforms, apply it.\n"
                "- If the context is for a clearly different model and "
                "the procedure doesn't apply, note the difference.\n"
                "- Do not add unnecessary disclaimers about missing information.\n"
                "- Answer concisely using the provided context."
            ).format(user_bike=user_bike, context_bikes=context_bikes),
        )
        return answer
