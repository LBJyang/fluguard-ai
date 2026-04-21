"""
FluGuard RAG Engine
流感卫士 RAG 引擎

Fully local, offline-capable:
  - Embeddings: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
    (supports Chinese + English, ~420 MB, downloaded once)
  - Vector store: ChromaDB (ephemeral in-memory, no server needed)
  - Chunking: paragraph-based with sliding overlap
"""

import logging
import os
import re
from pathlib import Path
from typing import Optional

log = logging.getLogger("fluguard.rag")


class RAGEngine:
    """
    Local RAG engine backed by ChromaDB and multilingual sentence embeddings.

    Usage:
        rag = RAGEngine(knowledge_dir="knowledge_base")
        results = rag.query("如何预防流感", n_results=3)
        context = rag.format_context(results)
    """

    def __init__(self, knowledge_dir: str = "knowledge_base"):
        self._collection = None
        self._ef = None
        self._load(knowledge_dir)

    # ── Internal: build the index ────────────────────────────────────────────

    def _load(self, knowledge_dir: str):
        try:
            import chromadb
            from chromadb.utils import embedding_functions
        except ImportError as e:
            raise RuntimeError(
                "Missing dependencies. Run: pip install chromadb sentence-transformers"
            ) from e

        log.info("Initialising ChromaDB (in-memory)...")
        client = chromadb.Client()  # ephemeral, no disk write needed

        log.info("Loading sentence-transformer: paraphrase-multilingual-MiniLM-L12-v2")
        self._ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="paraphrase-multilingual-MiniLM-L12-v2"
        )

        self._collection = client.create_collection(
            name="fluguard_kb",
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )

        # Index all markdown files in the knowledge directory
        kb_path = Path(knowledge_dir)
        md_files = sorted(kb_path.glob("*.md"))
        if not md_files:
            log.warning(f"No .md files found in '{knowledge_dir}'")
            return

        all_chunks: list[str] = []
        all_ids: list[str] = []
        all_meta: list[dict] = []

        for fpath in md_files:
            text = fpath.read_text(encoding="utf-8")
            chunks = self._chunk(text)
            for i, chunk in enumerate(chunks):
                chunk_id = f"{fpath.stem}_chunk{i}"
                all_chunks.append(chunk)
                all_ids.append(chunk_id)
                all_meta.append({"source": fpath.name, "chunk_index": i})

        if all_chunks:
            # ChromaDB batch limit is 5461; split if needed
            BATCH = 500
            for start in range(0, len(all_chunks), BATCH):
                end = start + BATCH
                self._collection.add(
                    documents=all_chunks[start:end],
                    ids=all_ids[start:end],
                    metadatas=all_meta[start:end],
                )
            log.info(f"Indexed {len(all_chunks)} chunks from {len(md_files)} files")

    # ── Internal: chunking strategy ──────────────────────────────────────────

    @staticmethod
    def _chunk(text: str, max_chars: int = 600, overlap: int = 100) -> list[str]:
        """
        Split document into overlapping chunks:
          1. Split on double-newline (paragraph boundaries)
          2. Merge small paragraphs; split oversized ones
          3. Add sliding overlap to preserve context across boundaries
        """
        # Remove markdown headers for cleaner chunks, keep content
        text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)

        paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
        chunks: list[str] = []
        current = ""

        for para in paragraphs:
            if len(current) + len(para) + 1 <= max_chars:
                current = (current + "\n\n" + para).strip()
            else:
                if current:
                    chunks.append(current)
                # Overlap: carry last `overlap` chars from previous chunk
                tail = current[-overlap:] if len(current) > overlap else current
                current = (tail + "\n\n" + para).strip() if tail else para

        if current:
            chunks.append(current)

        return chunks

    # ── Public API ───────────────────────────────────────────────────────────

    def query(self, text: str, n_results: int = 4) -> list[dict]:
        """
        Retrieve the top-n most relevant chunks for the given query.

        Returns list of:
          {"document": str, "source": str, "distance": float}
        """
        if self._collection is None or self._collection.count() == 0:
            return []

        results = self._collection.query(
            query_texts=[text],
            n_results=min(n_results, self._collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        output = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            output.append({
                "document": doc,
                "source": meta.get("source", "unknown"),
                "distance": round(dist, 4),
            })

        return output

    def format_context(self, results: list[dict], max_total_chars: int = 2000) -> str:
        """
        Format retrieved chunks into a concise context string for the LLM.
        Respects a total character budget.
        """
        if not results:
            return "No relevant knowledge base entries found."

        parts = []
        total = 0
        for r in results:
            entry = f"[Source: {r['source']}]\n{r['document']}"
            if total + len(entry) > max_total_chars:
                # Add truncated version if budget allows at least 200 chars
                remaining = max_total_chars - total
                if remaining > 200:
                    parts.append(entry[:remaining] + "…")
                break
            parts.append(entry)
            total += len(entry)

        return "\n\n---\n\n".join(parts)

    def document_count(self) -> int:
        if self._collection is None:
            return 0
        return self._collection.count()
