import logging
from typing import Optional

from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

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

    def add_documents(self, documents: list[Document]) -> int:
        if not documents:
            return 0

        texts = [doc.page_content for doc in documents]
        embeddings = self.model.encode(texts, show_progress_bar=False)

        points = []
        for idx, (doc, emb) in enumerate(zip(documents, embeddings)):
            points.append(
                PointStruct(
                    id=idx,
                    vector=emb.tolist(),
                    payload={
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

        self.client.upsert(collection_name=self.collection, points=points)
        logger.info("Stored %d vectors in '%s'", len(points), self.collection)
        return len(points)

    def similarity_search(self, query: str, k: int = 5) -> list[Document]:
        query_vector = self.model.encode(query).tolist()
        response = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            limit=k,
        )
        docs: list[Document] = []
        for hit in response.points:
            payload = hit.payload or {}
            metadata = {
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
