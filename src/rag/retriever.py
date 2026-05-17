"""Document retrieval logic — Small-to-Big strategy."""

import logging

from langchain_core.documents import Document

from src.rag.vector_store import VectorStore

logger = logging.getLogger(__name__)


class SmallToBigRetriever:
    """Small-to-Big retrieval: search small (child) chunks, return big (parent) chunks."""

    def __init__(self, vector_store: VectorStore) -> None:
        self._vector_store = vector_store

    def retrieve(self, query: str, k: int = 5) -> list[Document]:
        child_docs = self._vector_store.similarity_search(query, k=k)
        if not child_docs:
            return []

        parent_indices = sorted(
            {
                doc.metadata["parent_index"]
                for doc in child_docs
                if doc.metadata.get("parent_index") is not None
            }
        )
        if not parent_indices:
            return []

        parent_docs = self._vector_store.get_parents_by_indices(parent_indices)

        best_scores: dict[int, float] = {}
        for child in child_docs:
            p_idx = child.metadata.get("parent_index")
            score = child.metadata.get("score", 0.0)
            if p_idx is not None and (
                p_idx not in best_scores or score > best_scores[p_idx]
            ):
                best_scores[p_idx] = score

        for doc in parent_docs:
            p_idx = doc.metadata.get("index")
            if p_idx is not None and p_idx in best_scores:
                doc.metadata["score"] = best_scores[p_idx]

        logger.debug(
            "Small-to-Big: %d child hits → %d unique parents",
            len(child_docs),
            len(parent_docs),
        )
        return parent_docs
