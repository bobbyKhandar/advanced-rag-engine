import logging
from typing import Optional

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

logger = logging.getLogger(__name__)

HEADERS_TO_SPLIT_ON = [
    ("#", "Chapter"),
    ("##", "Section"),
    ("###", "Subsection"),
]


class HierarchicalChunker:

    def __init__(
        self,
        parent_size: int = 2048,
        child_size: int = 512,
        child_overlap: int = 64,
    ) -> None:
        self.parent_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=HEADERS_TO_SPLIT_ON,
            strip_headers=False,
        )
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_size,
            chunk_overlap=child_overlap,
            separators=["\n\n", "\n", ".", " ", ""],
        )

    def split_text(
        self, markdown: str
    ) -> tuple[list[Document], list[Document]]:
        parents = self.parent_splitter.split_text(markdown)
        if not parents and markdown.strip():
            parents = [Document(page_content=markdown, metadata={})]

        children: list[Document] = []
        for p_idx, parent in enumerate(parents):
            chunks = self.child_splitter.split_documents([parent])
            for c_idx, chunk in enumerate(chunks):
                chunk.metadata["parent_index"] = p_idx
                chunk.metadata["chunk_index"] = c_idx
            children.extend(chunks)

        logger.debug(
            "Hierarchical split: %d parents, %d children",
            len(parents),
            len(children),
        )
        return parents, children

    def split_documents(
        self, documents: list[Document]
    ) -> tuple[list[Document], list[Document]]:
        text = "\n\n".join(doc.page_content for doc in documents)
        return self.split_text(text)
