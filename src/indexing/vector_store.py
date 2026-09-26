"""Chroma vector store with BGE Chinese embeddings."""

import os
from typing import List, Optional

import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

from src.domain import DEFAULT_COLLECTION

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

    def get_or_create_collection(self, name: str = DEFAULT_COLLECTION):
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


def fetch_all(collection, batch: int = 5000) -> dict:
    """Fetch every record in a collection, paginated.

    ``collection.get()`` 不带 limit 时会试图一次取回全部记录，Chroma 的
    SQLite 后端因此撞上 ``SQLITE_MAX_VARIABLE_NUMBER``（约 32,766）：
    实测 27,672 条可以通过，54,618 条报 "too many SQL variables"。

    也就是说一次性取数的做法把索引规模卡在了 3 万条左右 —— 而本项目
    重建后的规模是这个数的近两倍。分批取没有这个问题。

    Returns ``{"ids": [...], "documents": [...], "metadatas": [...]}``.
    """
    ids: List[str] = []
    documents: List[str] = []
    metadatas: List[dict] = []
    offset = 0
    while True:
        page = collection.get(limit=batch, offset=offset)
        page_ids = page.get("ids") or []
        if not page_ids:
            break
        ids.extend(page_ids)
        documents.extend(page.get("documents") or [])
        metadatas.extend(page.get("metadatas") or [])
        offset += len(page_ids)
        if len(page_ids) < batch:
            break
    return {"ids": ids, "documents": documents, "metadatas": metadatas}
