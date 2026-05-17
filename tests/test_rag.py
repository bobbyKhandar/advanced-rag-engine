"""Tests for the RAG ingestion pipeline."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

import fitz
import pytest
from langchain_core.documents import Document

from scripts.bulk_ingest import collect_pdfs
from src.rag.chunking import HierarchicalChunker
from src.rag.ingestor import IngestResult, Ingestor, _resolve_env
from src.rag.loader import get_page_count, load_pdf_to_markdown


# ── Helpers ───────────────────────────────────────────────────────────


def _make_pdf(tmpdir: Path, text: str = "Test document", filename: str = "test.pdf") -> Path:
    path = tmpdir / filename
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 700), text)
    doc.save(str(path))
    doc.close()
    return path


def _ingestor_no_skip() -> Ingestor:
    """Return an Ingestor with page skipping disabled and no real vector store."""
    cfg = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    cfg.write(
        "rag:\n"
        "  chunking:\n"
        "    skip_toc_pages: 0\n"
        "    skip_index_pages: 0\n"
        "vector_store:\n"
        "  collection: test\n"
    )
    cfg.close()
    mock_store = MagicMock()
    mock_store.add_documents.return_value = 0
    mock_store.add_parent_documents.return_value = 0
    ing = Ingestor(config_path=cfg.name, vector_store=mock_store)
    return ing


MOCK_EMBEDDING = np.array([[0.1] * 384])


# ── _resolve_env ──────────────────────────────────────────────────────


class TestResolveEnv:

    def test_resolves_simple_var(self):
        os.environ["_TEST_VAR"] = "hello"
        assert _resolve_env("${_TEST_VAR}") == "hello"

    def test_unset_var_returns_empty(self):
        assert _resolve_env("${_NONEXISTENT}") == ""

    def test_no_var_returns_original(self):
        assert _resolve_env("plain-string") == "plain-string"

    def test_mixed_text_with_var(self):
        os.environ["_TEST_URL"] = "https://qdrant.dev"
        assert _resolve_env("url=${_TEST_URL}/path") == "url=https://qdrant.dev/path"


# ── IngestResult dataclass ────────────────────────────────────────────


class TestIngestResult:

    def test_defaults(self):
        result = IngestResult(file_path="/a/b.pdf", original_name="b.pdf")
        assert result.file_path == "/a/b.pdf"
        assert result.original_name == "b.pdf"
        assert result.chunks == 0
        assert result.parents == 0
        assert result.stored == 0
        assert result.success is True
        assert result.error is None

    def test_all_fields(self):
        result = IngestResult(
            file_path="/x/y.pdf",
            original_name="y.pdf",
            chunks=42,
            parents=5,
            stored=40,
            success=False,
            error="something went wrong",
        )
        assert result.chunks == 42
        assert result.parents == 5
        assert result.stored == 40
        assert result.success is False
        assert result.error == "something went wrong"


# ── HierarchicalChunker ───────────────────────────────────────────────


class TestHierarchicalChunker:

    def test_init_defaults(self):
        chunker = HierarchicalChunker()
        assert chunker.child_splitter._chunk_size == 512
        assert chunker.child_splitter._chunk_overlap == 64

    def test_init_custom(self):
        chunker = HierarchicalChunker(parent_size=1000, child_size=100, child_overlap=10)
        assert chunker.child_splitter._chunk_size == 100
        assert chunker.child_splitter._chunk_overlap == 10

    def test_split_text_with_headers(self):
        md = (
            "# Chapter 1\n\n"
            "Introduction text here.\n\n"
            "## Section 1.1\n\n"
            "Detailed content for section one.\n\n"
            "### Subsection 1.1.1\n\n"
            "Very specific details.\n\n"
            "# Chapter 2\n\n"
            "Second chapter content."
        )
        chunker = HierarchicalChunker(parent_size=2000, child_size=500, child_overlap=0)
        parents, children = chunker.split_text(md)

        assert len(parents) == 4
        assert parents[0].metadata.get("Chapter") == "Chapter 1"
        assert parents[1].metadata.get("Section") == "Section 1.1"
        assert parents[2].metadata.get("Subsection") == "Subsection 1.1.1"
        assert parents[3].metadata.get("Chapter") == "Chapter 2"

        assert len(children) >= 4
        for c in children:
            assert "parent_index" in c.metadata
            assert "chunk_index" in c.metadata

    def test_split_text_no_headers(self):
        md = (
            "Just a plain paragraph without any markdown headers.\n\n"
            "Another paragraph with more text content.\n\n"
            "And a third one for good measure."
        )
        chunker = HierarchicalChunker(parent_size=2000, child_size=100, child_overlap=0)
        parents, children = chunker.split_text(md)

        assert len(parents) == 1
        assert parents[0].metadata == {}
        assert len(children) > 0

    def test_split_text_empty(self):
        chunker = HierarchicalChunker()
        parents, children = chunker.split_text("")
        assert len(parents) == 0
        assert len(children) == 0

    def test_child_metadata_links_to_parent(self):
        md = (
            "# Intro\n\n"
            "First section content here. " * 20 + "\n\n"
            "# Details\n\n"
            "Second section content. " * 20
        )
        chunker = HierarchicalChunker(parent_size=2000, child_size=50, child_overlap=0)
        parents, children = chunker.split_text(md)

        parent_indices = {c.metadata["parent_index"] for c in children}
        assert parent_indices == {0, 1}

        for p_idx in range(len(parents)):
            kids = [c for c in children if c.metadata["parent_index"] == p_idx]
            assert len(kids) > 0

    def test_split_documents_merges_and_splits(self):
        docs = [
            Document(page_content="# Header A\n\nText A"),
            Document(page_content="# Header B\n\nText B"),
        ]
        chunker = HierarchicalChunker()
        parents, children = chunker.split_documents(docs)
        assert len(parents) == 2
        assert len(children) >= 2

    def test_tables_in_markdown_are_preserved(self):
        md = (
            "# Table Section\n\n"
            "| Name  | Value |\n"
            "|-------|-------|\n"
            "| Alpha | 100   |\n"
            "| Beta  | 200   |\n\n"
            "Some text after the table."
        )
        chunker = HierarchicalChunker(parent_size=2000, child_size=500, child_overlap=0)
        parents, children = chunker.split_text(md)

        assert len(parents) == 1
        combined = " ".join(c.page_content for c in children)
        assert "|" in combined
        assert "Alpha" in combined
        assert "Beta" in combined


# ── PDF loading ───────────────────────────────────────────────────────


class TestLoadPdfToMarkdown:

    def test_loads_single_page(self, tmp_path: Path):
        pdf = _make_pdf(tmp_path, "## Heading\n\nHello world")
        md = load_pdf_to_markdown(str(pdf))
        assert isinstance(md, str)
        assert len(md) > 0
        assert "Hello" in md

    def test_get_page_count(self, tmp_path: Path):
        pdf = _make_pdf(tmp_path, "Page 1")
        assert get_page_count(str(pdf)) == 1

    def test_page_skip_removes_all_pages(self, tmp_path: Path):
        pdf = _make_pdf(tmp_path, "## Intro\n\nContent", filename="multi.pdf")
        md_trimmed = load_pdf_to_markdown(str(pdf), skip_start=1, skip_end=0)
        assert md_trimmed == ""


# ── Ingestor ──────────────────────────────────────────────────────────


class TestIngestorInit:

    def test_loads_config_from_default_path(self):
        ingestor = Ingestor(vector_store=MagicMock())
        assert isinstance(ingestor.chunk_size, int)
        assert isinstance(ingestor.chunk_overlap, int)
        assert isinstance(ingestor.collection, str)
        assert ingestor.chunk_size > 0
        assert ingestor.collection != ""

    def test_loads_config_from_explicit_path(self):
        config_path = Path(__file__).resolve().parent.parent / "config.yaml"
        ingestor = Ingestor(config_path=str(config_path), vector_store=MagicMock())
        assert ingestor.chunk_size == 512
        assert ingestor.chunk_overlap == 64
        assert ingestor.collection == "bike_manuals"
        assert ingestor.parent_size == 2048
        assert ingestor.skip_toc_pages == 4
        assert ingestor.skip_index_pages == 3
        assert ingestor.embedding_model == "all-MiniLM-L6-v2"

    def test_missing_config_falls_back_to_defaults(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("rag: {}\nvector_store: {}\n")
            tmp = f.name
        try:
            ingestor = Ingestor(config_path=tmp, vector_store=MagicMock())
            assert ingestor.chunk_size == 512
            assert ingestor.chunk_overlap == 64
            assert ingestor.collection == "documents"
            assert ingestor.parent_size == 2048
            assert ingestor.skip_toc_pages == 0
            assert ingestor.embedding_model == "all-MiniLM-L6-v2"
        finally:
            os.unlink(tmp)


class TestProcessDocument:

    def test_ingests_real_pdf(self, tmp_path: Path):
        pdf = _make_pdf(tmp_path, "# Chapter A\n\nLots of content here.\n\n## Section B\n\nMore data.")
        ingestor = _ingestor_no_skip()
        result = ingestor.process_document(str(pdf), original_name="test.pdf")

        assert isinstance(result, IngestResult)
        assert result.file_path == str(pdf)
        assert result.original_name == "test.pdf"
        assert result.success is True
        assert result.chunks > 0
        assert result.parents > 0
        assert result.stored == 0

    def test_falls_back_to_basename(self, tmp_path: Path):
        pdf = _make_pdf(tmp_path, "Plain content", filename="mydoc.pdf")
        ingestor = _ingestor_no_skip()
        result = ingestor.process_document(str(pdf))
        assert result.original_name == "mydoc.pdf"

    def test_empty_original_name_uses_basename(self, tmp_path: Path):
        pdf = _make_pdf(tmp_path, "Something", filename="fallback.pdf")
        ingestor = _ingestor_no_skip()
        result = ingestor.process_document(str(pdf), original_name="")
        assert result.original_name == "fallback.pdf"


class TestProcessDocuments:

    def test_batch_returns_list_of_results(self, tmp_path: Path):
        p1 = _make_pdf(tmp_path, "# Doc 1\n\nContent", filename="a.pdf")
        p2 = _make_pdf(tmp_path, "# Doc 2\n\nContent", filename="b.pdf")
        ingestor = _ingestor_no_skip()
        results = ingestor.process_documents([str(p1), str(p2)])

        assert len(results) == 2
        assert all(r.success for r in results)
        assert all(r.chunks > 0 for r in results)

    def test_continues_on_error(self, tmp_path: Path):
        good = _make_pdf(tmp_path, "# Good\n\nContent", filename="good.pdf")
        bad = "/nonexistent/path.pdf"
        ingestor = _ingestor_no_skip()
        results = ingestor.process_documents([str(good), bad])

        assert len(results) == 2
        assert results[0].success is True
        assert results[0].chunks > 0
        assert results[1].success is False
        assert results[1].error is not None

    def test_empty_list_returns_empty(self):
        ingestor = Ingestor(vector_store=MagicMock())
        results = ingestor.process_documents([])
        assert results == []

    def test_stores_chunks_to_vector_db(self, tmp_path: Path):
        pdf = _make_pdf(tmp_path, "# Topic\n\nContent here.\n\n## Subtopic\n\nMore details.")
        mock_store = MagicMock()
        mock_store.add_documents.return_value = 42
        mock_store.add_parent_documents.return_value = 5
        ingestor = Ingestor(
            config_path=str(Path(__file__).resolve().parent.parent / "config.yaml"),
            vector_store=mock_store,
        )
        ingestor.skip_toc_pages = 0
        ingestor.skip_index_pages = 0
        result = ingestor.process_document(str(pdf))

        assert result.success is True
        assert result.stored == 42
        assert result.stored_parents == 5
        mock_store.add_documents.assert_called_once()
        mock_store.add_parent_documents.assert_called_once()


# ── CLI helper ────────────────────────────────────────────────────────


class TestCollectPdfs:

    def test_finds_pdfs_in_directory(self, tmp_path: Path):
        (tmp_path / "a.pdf").write_text("")
        (tmp_path / "b.PDF").write_text("")
        (tmp_path / "readme.txt").write_text("")
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "c.pdf").write_text("")

        pdfs = collect_pdfs(str(tmp_path))
        assert len(pdfs) == 3
        assert all(p.lower().endswith(".pdf") for p in pdfs)

    def test_empty_directory_returns_empty(self, tmp_path: Path):
        assert collect_pdfs(str(tmp_path)) == []

    def test_non_pdf_ignored(self, tmp_path: Path):
        (tmp_path / "data.txt").write_text("")
        (tmp_path / "notes.md").write_text("")
        assert collect_pdfs(str(tmp_path)) == []


# ── VectorStore ───────────────────────────────────────────────────────


class TestVectorStore:

    @patch("src.rag.vector_store.SentenceTransformer")
    @patch("src.rag.vector_store.QdrantClient")
    def test_add_documents(self, MockClient, MockTransformer):
        mock_model = MagicMock()
        mock_model.get_embedding_dimension.return_value = 384
        mock_model.encode.return_value = np.array([[0.1] * 384, [0.2] * 384])
        MockTransformer.return_value = mock_model

        mock_client = MagicMock()
        MockClient.return_value = mock_client

        from src.rag.vector_store import VectorStore

        store = VectorStore(collection="test", url="http://fake", api_key="key")

        docs = [
            Document(page_content="First chunk", metadata={"source": "a.pdf"}),
            Document(page_content="Second chunk", metadata={"source": "a.pdf"}),
        ]
        n = store.add_documents(docs)
        assert n == 2
        mock_client.upsert.assert_called_once()

    @patch("src.rag.vector_store.SentenceTransformer")
    @patch("src.rag.vector_store.QdrantClient")
    def test_add_empty_returns_zero(self, MockClient, MockTransformer):
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        MockTransformer.return_value = mock_model

        MockClient.return_value = MagicMock()

        from src.rag.vector_store import VectorStore

        store = VectorStore(collection="test", url="http://fake", api_key="key")
        n = store.add_documents([])
        assert n == 0

    @patch("src.rag.vector_store.SentenceTransformer")
    @patch("src.rag.vector_store.QdrantClient")
    def test_similarity_search(self, MockClient, MockTransformer):
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.encode.return_value = np.array([0.1] * 384)
        MockTransformer.return_value = mock_model

        from qdrant_client.models import ScoredPoint

        mock_hit = MagicMock(spec=ScoredPoint)
        mock_hit.score = 0.95
        mock_hit.payload = {
            "type": "child",
            "text": "Found content",
            "source": "doc.pdf",
            "parent_index": 0,
            "chunk_index": 1,
        }

        from qdrant_client.http.models.models import QueryResponse

        mock_response = MagicMock(spec=QueryResponse)
        mock_response.points = [mock_hit]
        mock_client = MagicMock()
        mock_client.query_points.return_value = mock_response
        MockClient.return_value = mock_client

        from src.rag.vector_store import VectorStore

        store = VectorStore(collection="test", url="http://fake", api_key="key")
        results = store.similarity_search("query", k=3)

        assert len(results) == 1
        assert results[0].page_content == "Found content"
        assert results[0].metadata["score"] == 0.95
        mock_client.query_points.assert_called_once()

    @patch("src.rag.vector_store.SentenceTransformer")
    @patch("src.rag.vector_store.QdrantClient")
    def test_add_parent_documents(self, MockClient, MockTransformer):
        mock_model = MagicMock()
        mock_model.get_embedding_dimension.return_value = 384
        mock_model.encode.return_value = np.array([[0.3] * 384, [0.4] * 384])
        MockTransformer.return_value = mock_model

        mock_client = MagicMock()
        MockClient.return_value = mock_client

        from src.rag.vector_store import PARENT_ID_OFFSET, VectorStore

        store = VectorStore(collection="test", url="http://fake", api_key="key")

        parents = [
            Document(
                page_content="## Big Chapter\n\nLots of context here.",
                metadata={"source": "a.pdf", "Chapter": "Ch1"},
            ),
            Document(
                page_content="## Another Chapter\n\nMore context.",
                metadata={"source": "a.pdf", "Chapter": "Ch2"},
            ),
        ]
        n = store.add_parent_documents(parents)
        assert n == 2

        call_args = mock_client.upsert.call_args
        points = call_args[1]["points"]
        assert len(points) == 2
        assert points[0].id == PARENT_ID_OFFSET + 0
        assert points[1].id == PARENT_ID_OFFSET + 1
        assert points[0].payload["type"] == "parent"
        assert points[0].payload["index"] == 0
        assert points[1].payload["index"] == 1

    @patch("src.rag.vector_store.SentenceTransformer")
    @patch("src.rag.vector_store.QdrantClient")
    def test_add_parent_empty_returns_zero(self, MockClient, MockTransformer):
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        MockTransformer.return_value = mock_model
        MockClient.return_value = MagicMock()

        from src.rag.vector_store import VectorStore

        store = VectorStore(collection="test", url="http://fake", api_key="key")
        n = store.add_parent_documents([])
        assert n == 0

    @patch("src.rag.vector_store.SentenceTransformer")
    @patch("src.rag.vector_store.QdrantClient")
    def test_get_parents_by_indices(self, MockClient, MockTransformer):
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.encode.return_value = np.array([0.1] * 384)
        MockTransformer.return_value = mock_model

        from qdrant_client.models import Record

        mock_record = MagicMock(spec=Record)
        mock_record.payload = {
            "type": "parent",
            "text": "## Parent content\n\nFull context here.",
            "source": "doc.pdf",
            "index": 0,
            "chapter": "Ch1",
            "section": "Sec1",
        }

        mock_client = MagicMock()
        mock_client.scroll.return_value = ([mock_record], None)
        MockClient.return_value = mock_client

        from src.rag.vector_store import VectorStore

        store = VectorStore(collection="test", url="http://fake", api_key="key")
        results = store.get_parents_by_indices([0, 1])

        assert len(results) == 1
        assert results[0].page_content == "## Parent content\n\nFull context here."
        assert results[0].metadata["index"] == 0
        assert results[0].metadata["chapter"] == "Ch1"

        mock_client.scroll.assert_called_once()

    @patch("src.rag.vector_store.SentenceTransformer")
    @patch("src.rag.vector_store.QdrantClient")
    def test_get_parents_by_indices_empty(self, MockClient, MockTransformer):
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        MockTransformer.return_value = mock_model
        MockClient.return_value = MagicMock()

        from src.rag.vector_store import VectorStore

        store = VectorStore(collection="test", url="http://fake", api_key="key")
        results = store.get_parents_by_indices([])
        assert results == []

    @patch("src.rag.vector_store.SentenceTransformer")
    @patch("src.rag.vector_store.QdrantClient")
    def test_similarity_search_filters_by_type(self, MockClient, MockTransformer):
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.encode.return_value = np.array([0.1] * 384)
        MockTransformer.return_value = mock_model

        from qdrant_client.models import ScoredPoint
        from qdrant_client.http.models.models import QueryResponse

        mock_hit = MagicMock(spec=ScoredPoint)
        mock_hit.score = 0.95
        mock_hit.payload = {
            "type": "child",
            "text": "Child chunk",
            "source": "doc.pdf",
            "parent_index": 0,
        }

        mock_response = MagicMock(spec=QueryResponse)
        mock_response.points = [mock_hit]
        mock_client = MagicMock()
        mock_client.query_points.return_value = mock_response
        MockClient.return_value = mock_client

        from src.rag.vector_store import VectorStore

        store = VectorStore(collection="test", url="http://fake", api_key="key")
        results = store.similarity_search("query", k=3)

        assert len(results) == 1
        call_kwargs = mock_client.query_points.call_args[1]
        assert "query_filter" in call_kwargs
        assert call_kwargs["query_filter"] is not None

    @patch("src.rag.vector_store.SentenceTransformer")
    @patch("src.rag.vector_store.QdrantClient")
    def test_len(self, MockClient, MockTransformer):
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        MockTransformer.return_value = mock_model

        from qdrant_client.models import CountResult

        mock_client = MagicMock()
        mock_client.count.return_value = CountResult(count=7)
        MockClient.return_value = mock_client

        from src.rag.vector_store import VectorStore

        store = VectorStore(collection="test", url="http://fake", api_key="key")
        assert len(store) == 7


# ── RagProcessor ──────────────────────────────────────────────────────


class TestRagProcessor:

    def _make_child_doc(
        self, content: str, score: float, source: str = "test.pdf", parent_index: int = 0
    ):
        doc = MagicMock(spec=Document)
        doc.page_content = content
        doc.metadata = {
            "score": score,
            "source": source,
            "parent_index": parent_index,
            "type": "child",
        }
        return doc

    def _make_parent_doc(
        self, content: str, source: str = "test.pdf", index: int = 0,
        chapter: str = "", section: str = "",
    ):
        return Document(
            page_content=content,
            metadata={
                "source": source,
                "index": index,
                "chapter": chapter,
                "section": section,
            },
        )

    @patch("src.rag.rag_processor.Generator")
    @patch("src.rag.rag_processor.VectorStore")
    @patch("dotenv.load_dotenv")
    def test_process_formats_results(self, mock_dotenv, MockVectorStore, MockGenerator):
        mock_vs = MagicMock()
        mock_vs.similarity_search.return_value = [
            self._make_child_doc("Brake lever torque", 0.95, "manual.pdf", parent_index=0),
            self._make_child_doc("Oil change interval", 0.82, "manual.pdf", parent_index=1),
        ]
        mock_vs.get_parents_by_indices.return_value = [
            self._make_parent_doc(
                "## Brakes\n\nBrake lever torque specification is 10 Nm.",
                "manual.pdf", index=0, chapter="Ch3", section="Brakes",
            ),
            self._make_parent_doc(
                "## Oil\n\nOil change interval every 6 months.",
                "manual.pdf", index=1, chapter="Ch5", section="Oil",
            ),
        ]
        MockVectorStore.return_value = mock_vs

        mock_gen = MagicMock()
        mock_gen.generate.return_value = (
            "The brake lever torque specification is 10 Nm, "
            "and the oil change interval is every 6 months."
        )
        MockGenerator.return_value = mock_gen

        from src.rag.rag_processor import RagProcessor

        proc = RagProcessor()
        import asyncio
        result = asyncio.run(proc.process("brake"))

        assert "brake lever torque" in result.lower()
        assert "oil change" in result.lower()
        # Verify context was passed to generator
        call_context = mock_gen.generate.call_args[0][1]
        assert "Brake lever torque" in call_context
        assert "Oil change interval" in call_context

    @patch("src.rag.rag_processor.Generator")
    @patch("src.rag.rag_processor.VectorStore")
    @patch("dotenv.load_dotenv")
    def test_process_empty_results(self, mock_dotenv, MockVectorStore, MockGenerator):
        mock_vs = MagicMock()
        mock_vs.similarity_search.return_value = []
        MockVectorStore.return_value = mock_vs

        from src.rag.rag_processor import RagProcessor

        proc = RagProcessor()
        import asyncio
        result = asyncio.run(proc.process("nonexistent topic"))

        assert "couldn't find" in result.lower()
        MockGenerator.return_value.generate.assert_not_called()


# ── SmallToBigRetriever ─────────────────────────────────────────────────


class TestSmallToBigRetriever:

    def _make_child(
        self, content: str, parent_index: int, score: float = 0.9, source: str = "doc.pdf"
    ) -> Document:
        doc = MagicMock(spec=Document)
        doc.page_content = content
        doc.metadata = {
            "score": score,
            "source": source,
            "parent_index": parent_index,
            "type": "child",
        }
        return doc

    def _make_parent(self, content: str, index: int, source: str = "doc.pdf") -> Document:
        return Document(
            page_content=content,
            metadata={"source": source, "index": index},
        )

    def test_retrieve_expands_to_parents(self):
        from src.rag.retriever import SmallToBigRetriever

        mock_store = MagicMock()
        mock_store.similarity_search.return_value = [
            self._make_child("Child A", parent_index=0, score=0.95),
            self._make_child("Child B", parent_index=1, score=0.82),
        ]
        mock_store.get_parents_by_indices.return_value = [
            self._make_parent("## Parent A content", index=0),
            self._make_parent("## Parent B content", index=1),
        ]

        retriever = SmallToBigRetriever(mock_store)
        results = retriever.retrieve("query", k=5)

        assert len(results) == 2
        assert results[0].page_content == "## Parent A content"
        assert results[1].page_content == "## Parent B content"
        mock_store.get_parents_by_indices.assert_called_once_with([0, 1])

    def test_retrieve_empty_results(self):
        from src.rag.retriever import SmallToBigRetriever

        mock_store = MagicMock()
        mock_store.similarity_search.return_value = []

        retriever = SmallToBigRetriever(mock_store)
        results = retriever.retrieve("query", k=5)

        assert results == []

    def test_retrieve_attaches_best_score(self):
        from src.rag.retriever import SmallToBigRetriever

        mock_store = MagicMock()
        mock_store.similarity_search.return_value = [
            self._make_child("Child A1", parent_index=0, score=0.70),
            self._make_child("Child A2", parent_index=0, score=0.95),
            self._make_child("Child B1", parent_index=1, score=0.82),
        ]
        mock_store.get_parents_by_indices.return_value = [
            self._make_parent("## Parent A", index=0),
            self._make_parent("## Parent B", index=1),
        ]

        retriever = SmallToBigRetriever(mock_store)
        results = retriever.retrieve("query", k=5)

        assert len(results) == 2
        # Parent 0 should get the best score from its children (0.95)
        parent0 = [d for d in results if d.metadata["index"] == 0][0]
        assert parent0.metadata["score"] == 0.95
        # Parent 1 should get score 0.82
        parent1 = [d for d in results if d.metadata["index"] == 1][0]
        assert parent1.metadata["score"] == 0.82

    def test_retrieve_deduplicates_parents(self):
        from src.rag.retriever import SmallToBigRetriever

        mock_store = MagicMock()
        mock_store.similarity_search.return_value = [
            self._make_child("Child A1", parent_index=0, score=0.90),
            self._make_child("Child A2", parent_index=0, score=0.85),
            self._make_child("Child A3", parent_index=0, score=0.80),
        ]
        mock_store.get_parents_by_indices.return_value = [
            self._make_parent("## Parent A", index=0),
        ]

        retriever = SmallToBigRetriever(mock_store)
        results = retriever.retrieve("query", k=5)

        assert len(results) == 1
        assert results[0].metadata["index"] == 0

    def test_retrieve_parents_not_found(self):
        from src.rag.retriever import SmallToBigRetriever

        mock_store = MagicMock()
        mock_store.similarity_search.return_value = []
        mock_store.get_parents_by_indices.return_value = []

        retriever = SmallToBigRetriever(mock_store)
        results = retriever.retrieve("query", k=5)
        assert results == []


# ── Generator ─────────────────────────────────────────────────────────


class TestGenerator:

    @patch("src.rag.generator.OpenAI")
    def test_generate_returns_answer(self, MockOpenAI):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "The torque specification is 10 Nm."
        mock_response.choices = [mock_choice]
        mock_response.usage = MagicMock(prompt_tokens=50, completion_tokens=10)
        mock_client.chat.completions.create.return_value = mock_response
        MockOpenAI.return_value = mock_client

        from src.rag.generator import Generator

        gen = Generator(api_key="test-key")
        result = gen.generate("What is the torque?", "## Brakes\n\nTorque: 10 Nm")

        assert result == "The torque specification is 10 Nm."
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "llama-3.3-70b-versatile"
        assert call_kwargs["temperature"] == 0.3
        assert len(call_kwargs["messages"]) == 2
        assert call_kwargs["messages"][1]["role"] == "user"
        assert "Torque: 10 Nm" in call_kwargs["messages"][1]["content"]
        assert "What is the torque?" in call_kwargs["messages"][1]["content"]

    @patch("src.rag.generator.OpenAI")
    def test_generate_custom_params(self, MockOpenAI):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Answer."
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response
        MockOpenAI.return_value = mock_client

        from src.rag.generator import Generator

        gen = Generator(
            api_key="test-key",
            model="mixtral-8x7b-32768",
            base_url="https://api.groq.com/openai/v1",
            temperature=0.7,
        )
        result = gen.generate("Question?", "Context.", system_prompt="Be brief.")

        assert result == "Answer."
        call_kwargs = mock_client.chat.completions.create.call_args[1]
        assert call_kwargs["model"] == "mixtral-8x7b-32768"
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["messages"][0]["content"] == "Be brief."

    @patch("src.rag.generator.OpenAI")
    def test_generate_empty_response(self, MockOpenAI):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = ""
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response
        MockOpenAI.return_value = mock_client

        from src.rag.generator import Generator

        gen = Generator(api_key="test-key")
        result = gen.generate("Question?", "Context.")
        assert result == ""


# ── Integration ───────────────────────────────────────────────────────


class TestFullPipeline:

    def test_pdf_to_markdown_to_chunker(self, tmp_path: Path):
        md_content = (
            "# Engine\n\n"
            "The engine is a 4-stroke unit.\n\n"
            "## Oil Change\n\n"
            "Change oil every 6 months.\n\n"
            "# Brakes\n\n"
            "Disc brakes front and rear."
        )
        pdf = _make_pdf(tmp_path, md_content)

        ingestor = _ingestor_no_skip()
        result = ingestor.process_document(str(pdf))

        assert result.success is True
        assert result.chunks > 0
        assert result.parents > 0
        assert result.stored == 0
