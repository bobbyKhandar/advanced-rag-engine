#!/usr/bin/env python3
"""Bulk ingest PDFs into the RAG vector store."""

import argparse
import logging
import os
import sys
from pathlib import Path

# Ensure the project root is on sys.path so src.* imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag.ingestor import Ingestor

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def collect_pdfs(input_dir: str) -> list[str]:
    pdfs: list[str] = []
    for root, _dirs, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith(".pdf"):
                pdfs.append(os.path.join(root, f))
    return sorted(pdfs)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bulk ingest PDFs into the RAG vector store"
    )
    parser.add_argument(
        "--input-dir",
        default="data/manuals",
        help="Directory containing PDF files (default: data/manuals)",
    )
    parser.add_argument(
        "--collection",
        default=None,
        help="Qdrant collection name (overrides config.yaml)",
    )
    parser.add_argument(
        "--skip-start",
        type=int,
        default=None,
        help="Override skip_toc_pages (config default: 4)",
    )
    parser.add_argument(
        "--skip-end",
        type=int,
        default=None,
        help="Override skip_index_pages (config default: 3)",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        logger.error("Input directory not found: %s", args.input_dir)
        sys.exit(1)

    pdfs = collect_pdfs(args.input_dir)
    if not pdfs:
        logger.warning("No PDF files found in %s", args.input_dir)
        sys.exit(0)

    logger.info("Found %d PDF(s) in %s", len(pdfs), args.input_dir)

    ingestor = Ingestor()
    if args.collection:
        ingestor.collection = args.collection
    if args.skip_start is not None:
        ingestor.skip_toc_pages = args.skip_start
    if args.skip_end is not None:
        ingestor.skip_index_pages = args.skip_end

    results = ingestor.process_documents(pdfs)

    ok = sum(1 for r in results if r.success)
    err = sum(1 for r in results if not r.success)
    total_chunks = sum(r.chunks for r in results)
    total_stored = sum(r.stored for r in results)

    print()
    for r in results:
        status = "OK" if r.success else "FAIL"
        detail = f"{r.chunks} chunks, {r.stored} stored" if r.success else r.error
        print(f"  [{status}] {r.original_name} - {detail}")
    print("-" * 50)
    print(f"  Total: {len(results)} files, {ok} ok, {err} failed, {total_chunks} chunks, {total_stored} stored")


if __name__ == "__main__":
    main()
