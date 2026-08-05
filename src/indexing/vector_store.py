"""Chroma vector store with BGE Chinese embeddings."""

import os
from typing import List, Optional

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()


class VectorStore:
    """Thin wrapper around Chroma for paper knowledge storage."""

    def __init__(
        self,
        persist_path: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        self.persist_path = persist_path or os.getenv(
            "CHROMA_PERSIST_PATH", "./chroma_db"
        )
        self.model_name = model_name or os.getenv(
            "EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5"
        )

        self.client = chromadb.PersistentClient(path=self.persist_path)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.model_name
        )

    def get_or_create_collection(self, name: str = "paper_knowledge"):
        # 获取/创建集合
        """Get an existing collection or create one."""
        return self.client.get_or_create_collection(
            name=name,
            embedding_function=self.embedding_fn,
        )

    def add_documents(
        self,
        collection,
        chunks: List[str],
        metadatas: Optional[List[dict]] = None,
        source: str = "",
    ):
        # 写入文档
        """Insert chunks into a collection.  Replaces existing docs from the same source."""
        ids = [f"{source}_chunk_{i}" for i in range(len(chunks))]
        if metadatas is None:
            metadatas = [{"source": source}] * len(chunks)
        else:
            for m in metadatas:
                m.setdefault("source", source)

        # delete old entries from this source before adding
        if source:
            try:
                existing = collection.get(where={"source": source})
                if existing["ids"]:
                    collection.delete(ids=existing["ids"])
            except Exception:
                pass

        if chunks:
            collection.add(ids=ids, documents=chunks, metadatas=metadatas)

    def query(self, collection, query: str, n_results: int = 5) -> dict:
        """Semantic search for the top-N chunks."""
        return collection.query(query_texts=[query], n_results=n_results)
