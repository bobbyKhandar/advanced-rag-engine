import logging
from typing import Optional

from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    PointStruct,
    VectorParams,
)

import uuid

PARENT_ID_OFFSET = 10_000_000

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None  # type: ignore[assignment]


class VectorStore:

    def __init__(
        self,
        collection: str,
        url: str,
        api_key: str,
        model_name: str = "all-MiniLM-L6-v2",
        recreate: bool = True,
    ) -> None:
        self._next_child_id = 0
        self._next_parent_id = 0
        logger.info(
            "Initializing VectorStore: collection=%s model=%s",
            collection,
            model_name,
        )
        self.collection = collection

        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers is required. Install it with: pip install sentence-transformers"
            )
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_embedding_dimension()

        self.client = QdrantClient(url=url, api_key=api_key)

        if recreate:
            self.client.recreate_collection(
                collection_name=collection,
                vectors_config=VectorParams(
                    size=self.dimension,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(
                "Collection '%s' created (dim=%d, distance=COSINE)",
                collection,
                self.dimension,
            )
            self._ensure_payload_indexes()
        elif not self.client.collection_exists(collection):
            self.client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(
                    size=self.dimension,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(
                "Collection '%s' created on demand (dim=%d, distance=COSINE)",
                collection,
                self.dimension,
            )
            self._ensure_payload_indexes()
        else:
            self._ensure_payload_indexes()

    def _ensure_payload_indexes(self, ) -> None:
        for field_name, field_type in [("type", "keyword"), ("index", "integer")]:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection,
                    field_name=field_name,
                    field_type=field_type,
                )
                logger.info("Created payload index on '%s' (%s)", field_name, field_type)
            except Exception:
                pass

    def add_documents(self, documents: list[Document]) -> int:
        if not documents:
            return 0

        texts = [doc.page_content for doc in documents]
        embeddings = self.model.encode(texts, show_progress_bar=False)

        points = []
        for idx, (doc, emb) in enumerate(zip(documents, embeddings)):
            points.append(
                PointStruct(
                    id=self._next_child_id + idx,
                    vector=emb.tolist(),
                    payload={
                        "type": "child",
                        "text": doc.page_content,
                        "source": doc.metadata.get("source", ""),
                        "parent_index": doc.metadata.get("parent_index"),
                        "chunk_index": doc.metadata.get("chunk_index"),
                        "chapter": doc.metadata.get("Chapter", ""),
                        "section": doc.metadata.get("Section", ""),
                        "subsection": doc.metadata.get("Subsection", ""),
                    },
                )
            )

        self._next_child_id += len(points)
        self.client.upsert(collection_name=self.collection, points=points)
        logger.info("Stored %d vectors in '%s'", len(points), self.collection)
        return len(points)

    def add_parent_documents(self, parents: list[Document]) -> int:
        if not parents:
            return 0

        texts = [doc.page_content for doc in parents]
        embeddings = self.model.encode(texts, show_progress_bar=False)

        points = []
        for idx, (doc, emb) in enumerate(zip(parents, embeddings)):
            parent_index = doc.metadata.get("index", self._next_parent_id + idx)
            points.append(
                PointStruct(
                    id=PARENT_ID_OFFSET + self._next_parent_id + idx,
                    vector=emb.tolist(),
                    payload={
                        "type": "parent",
                        "text": doc.page_content,
                        "source": doc.metadata.get("source", ""),
                        "index": parent_index,
                        "chapter": doc.metadata.get("Chapter", ""),
                        "section": doc.metadata.get("Section", ""),
                        "subsection": doc.metadata.get("Subsection", ""),
                    },
                )
            )

        self._next_parent_id += len(points)
        self.client.upsert(collection_name=self.collection, points=points)
        logger.info("Stored %d parent vectors in '%s'", len(points), self.collection)
        return len(points)

    def get_parents_by_indices(self, indices: list[int]) -> list[Document]:
        if not indices:
            return []

        records, _ = self.client.scroll(
            collection_name=self.collection,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="type", match=MatchValue(value="parent")),
                    FieldCondition(key="index", match=MatchAny(any=indices)),
                ]
            ),
            limit=len(indices),
            with_payload=True,
            with_vectors=False,
        )

        docs: list[Document] = []
        for record in records:
            payload = record.payload or {}
            metadata = {
                "source": payload.get("source", ""),
                "index": payload.get("index"),
                "chapter": payload.get("chapter", ""),
                "section": payload.get("section", ""),
                "subsection": payload.get("subsection", ""),
            }
            docs.append(
                Document(page_content=payload.get("text", ""), metadata=metadata)
            )
        return docs

    def similarity_search(
        self, query: str, k: int = 5, doc_type: str = "child"
    ) -> list[Document]:
        query_vector = self.model.encode(query).tolist()
        query_filter: Optional[Filter] = None
        if doc_type:
            query_filter = Filter(
                must=[FieldCondition(key="type", match=MatchValue(value=doc_type))]
            )
        response = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            query_filter=query_filter,
            limit=k,
        )
        docs: list[Document] = []
        for hit in response.points:
            payload = hit.payload or {}
            metadata = {
                "type": payload.get("type", ""),
                "source": payload.get("source", ""),
                "parent_index": payload.get("parent_index"),
                "chunk_index": payload.get("chunk_index"),
                "chapter": payload.get("chapter", ""),
                "section": payload.get("section", ""),
                "subsection": payload.get("subsection", ""),
                "score": hit.score,
            }
            docs.append(
                Document(
                    page_content=payload.get("text", ""), metadata=metadata
                )
            )
        return docs

    def __len__(self) -> int:
        result = self.client.count(collection_name=self.collection)
        return result.count
