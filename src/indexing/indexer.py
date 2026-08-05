"""Simple indexer: parse PDFs → split → store in Chroma + build BM25."""

from pathlib import Path

import os

from src.ingestion.parser import PDFParser
from src.ingestion.splitter import PaperSentenceSplitter
from src.indexing.vector_store import VectorStore
from src.indexing.bm25_index import BM25Index, _doc_signature
from src.retrieval.hybrid_retriever import HybridRetriever

BM25_CACHE_PATH = os.path.join("data", "bm25_index.pkl")


def build_index(
    data_dir: str = "data/raw",
    collection_name: str = "paper_knowledge",
):
    """Parse all PDFs, build Chroma index + BM25 index, return HybridRetriever."""
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"[!] data dir not found: {data_dir}, creating it...")
        data_path.mkdir(parents=True, exist_ok=True)
        print(f"[!] Put your PDF files in {data_path.resolve()}/ then re-run.")
        return None

    pdf_files = list(data_path.glob("*.pdf"))
    if not pdf_files:
        print(f"[!] No PDF files found in {data_path.resolve()}/")
        print("[!] Add PDF files and re-run to build the index.")
        return None

    print(f"[*] Found {len(pdf_files)} PDF(s)")

    parser = PDFParser()
    splitter = PaperSentenceSplitter(chunk_size=500, chunk_overlap=80)
    store = VectorStore()
    collection = store.get_or_create_collection(collection_name)

    all_chunks = []
    total_chunks = 0
    for pdf_file in pdf_files:
        print(f"  -> parsing: {pdf_file.name}")
        text = parser.parse(str(pdf_file))
        chunks = splitter.split(text)
        print(f"     {len(chunks)} chunks")

        store.add_documents(collection, chunks, source=pdf_file.name)
        all_chunks.extend(chunks)
        total_chunks += len(chunks)

    print(f"\n[OK] Chroma index: {total_chunks} chunks")

    # ── build BM25 from the FULL collection ──
    # (covers seed data / imported QA + the PDFs just parsed, so the
    #  keyword index stays consistent with everything actually in Chroma)
    print("[*] Building BM25 index ...")
    result = collection.get()
    full_docs = result.get("documents", [])
    bm25 = BM25Index()
    bm25.build(full_docs)
    print(f"[OK] BM25 index: {bm25.document_count} documents")
    bm25.save(BM25_CACHE_PATH)
    print(f"[*] BM25 index saved to {BM25_CACHE_PATH}")

    return HybridRetriever(collection, bm25, alpha=0.3)


def build_hybrid_from_collection(collection, alpha: float = 0.3) -> HybridRetriever:
    """Build a HybridRetriever from an existing Chroma collection.

    Tries to load a persisted BM25 index first (fast); rebuilds and saves
    if the cache is missing or stale (doesn't match the current collection).
    """
    bm25 = BM25Index()

    # ── try loading from disk ──
    if os.path.exists(BM25_CACHE_PATH):
        if bm25.load(BM25_CACHE_PATH):
            # verify freshness: cache must match the current collection
            try:
                result = collection.get()
                docs = result.get("documents", [])
            except Exception:
                docs = []
            if _doc_signature(docs) == bm25.signature:
                print(f"[*] BM25 index loaded from {BM25_CACHE_PATH} "
                      f"({bm25.document_count} docs)")
                return HybridRetriever(collection, bm25, alpha=alpha)
            print("[!] BM25 cache is stale — rebuilding from current collection")
        docs = []

    # ── rebuild from Chroma ──
    if not docs:
        try:
            result = collection.get()
            docs = result.get("documents", [])
        except Exception:
            docs = []

    if not docs:
        print("[!] Collection is empty, BM25 index will be empty.")
        bm25.build([])
        return HybridRetriever(collection, bm25, alpha=alpha)

    print(f"[*] Building BM25 index from {len(docs)} chunks (one-time) ...")
    bm25.build(docs)
    print(f"[OK] BM25 index: {bm25.document_count} documents")

    # ── persist to disk ──
    bm25.save(BM25_CACHE_PATH)
    print(f"[*] BM25 index saved to {BM25_CACHE_PATH}")

    return HybridRetriever(collection, bm25, alpha=alpha)
