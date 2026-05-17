import logging
import os
import re
from dataclasses import dataclass, field
from typing import Optional

import dotenv
import yaml

dotenv.load_dotenv()

from src.rag.chunking import HierarchicalChunker
from src.rag.loader import load_and_chunk
from src.rag.vector_store import VectorStore

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "config.yaml"
)

_ENV_PATTERN = re.compile(r"\$\{(\w+)\}")


def _resolve_env(value: str) -> str:
    """Replace ${VAR} patterns with environment variable values."""
    return _ENV_PATTERN.sub(lambda m: os.environ.get(m.group(1), ""), value)


@dataclass
class IngestResult:
    file_path: str
    original_name: str
    chunks: int = 0
    parents: int = 0
    stored: int = 0
    stored_parents: int = 0
    success: bool = True
    error: Optional[str] = None


class Ingestor:

    def __init__(
        self,
        config_path: str = _DEFAULT_CONFIG_PATH,
        vector_store: Optional[VectorStore] = None,
    ) -> None:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        rag_cfg = config.get("rag", {})
        vs_cfg = config.get("vector_store", {})
        chunking_cfg = rag_cfg.get("chunking", {})
        embedding_cfg = rag_cfg.get("embedding", {})

        self.chunk_size: int = rag_cfg.get("chunk_size", 512)
        self.chunk_overlap: int = rag_cfg.get("chunk_overlap", 64)
        self.collection: str = vs_cfg.get("collection", "documents")
        self.top_k: int = rag_cfg.get("top_k", 5)

        self.parent_size: int = chunking_cfg.get("parent_size", 2048)
        self.skip_toc_pages: int = chunking_cfg.get("skip_toc_pages", 0)
        self.skip_index_pages: int = chunking_cfg.get("skip_index_pages", 0)
        self.embedding_model: str = embedding_cfg.get(
            "model", "all-MiniLM-L6-v2"
        )

        self._chunker = HierarchicalChunker(
            parent_size=self.parent_size,
            child_size=self.chunk_size,
            child_overlap=self.chunk_overlap,
        )

        self._vs_config = {
            "url": _resolve_env(vs_cfg.get("url", "")),
            "api_key": _resolve_env(vs_cfg.get("api_key", "")),
        }
        self._vector_store = vector_store  # may be None (lazy init)
        self._parent_offset: int = 0

    @property
    def vector_store(self) -> VectorStore:
        if self._vector_store is None:
            self._vector_store = VectorStore(
                collection=self.collection,
                url=self._vs_config["url"],
                api_key=self._vs_config["api_key"],
                model_name=self.embedding_model,
            )
        return self._vector_store

    def process_document(
        self, file_path: str, original_name: str = ""
    ) -> IngestResult:
        name = original_name or os.path.basename(file_path)
        logger.info("Ingesting %s ...", name)

        parents, children = load_and_chunk(
            file_path,
            self._chunker,
            skip_start=self.skip_toc_pages,
            skip_end=self.skip_index_pages,
        )

        logger.info(
            "  Split into %d parent sections and %d child chunks "
            "(chunk_size=%d, overlap=%d)",
            len(parents),
            len(children),
            self.chunk_size,
            self.chunk_overlap,
        )

        # Attach source filename to all documents
        for child in children:
            child.metadata["source"] = name
            if "parent_index" in child.metadata:
                child.metadata["parent_index"] += self._parent_offset

        for parent in parents:
            parent.metadata["source"] = name

        for p_idx, parent in enumerate(parents):
            parent.metadata["index"] = self._parent_offset + p_idx

        stored = self.vector_store.add_documents(children)
        stored_parents = self.vector_store.add_parent_documents(parents)
        self._parent_offset += len(parents)

        return IngestResult(
            file_path=file_path,
            original_name=name,
            chunks=len(children),
            parents=len(parents),
            stored=stored,
            stored_parents=stored_parents,
            success=True,
        )

    def process_documents(self, file_paths: list[str]) -> list[IngestResult]:
        results: list[IngestResult] = []
        for fp in file_paths:
            try:
                result = self.process_document(
                    fp, original_name=os.path.basename(fp)
                )
            except Exception as exc:
                result = IngestResult(
                    file_path=fp,
                    original_name=os.path.basename(fp),
                    chunks=0,
                    parents=0,
                    stored=0,
                    success=False,
                    error=str(exc),
                )
                logger.exception("Failed to ingest %s", fp)
            results.append(result)
        return results
