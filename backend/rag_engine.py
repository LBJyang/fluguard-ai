"""
FluGuard RAG Engine
流感卫士 RAG 引擎

Cloud-optimised implementation using scikit-learn TF-IDF (no PyTorch / no ChromaDB).
Keeps the same public interface as the local ChromaDB+sentence-transformers version
so main.py requires zero changes.

Why TF-IDF for cloud:
  - sentence-transformers pulls PyTorch (~2 GB) → exceeds Railway free tier image limit
  - scikit-learn TF-IDF is ~50 MB, works well on a focused domain knowledge base
  - For a 5-file / 39-chunk medical knowledge base, TF-IDF retrieval quality is
    indistinguishable from dense embeddings for the queries we send
"""

import logging
import re
from pathlib import Path

import numpy as np

log = logging.getLogger("fluguard.rag")


class RAGEngine:
    """
    Lightweight RAG engine backed by scikit-learn TF-IDF + cosine similarity.

    Public interface is identical to the ChromaDB version:
        rag = RAGEngine(knowledge_dir="knowledge_base")
        results = rag.query("如何预防流感", n_results=3)
        context = rag.format_context(results)
    """

    def __init__(self, knowledge_dir: str = "knowledge_base"):
        self._chunks: list[str] = []
        self._sources: list[str] = []
        self._vectorizer = None
        self._matrix = None
        self._load(knowledge_dir)

    # ── Internal: build index ────────────────────────────────────────────────

    def _load(self, knowledge_dir: str):
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
        except ImportError as e:
            raise RuntimeError("Run: pip install scikit-learn") from e

        kb_path = Path(knowledge_dir)
        md_files = sorted(kb_path.glob("*.md"))
        if not md_files:
            log.warning(f"No .md files found in '{knowledge_dir}'")
            return

        for fpath in md_files:
            text = fpath.read_text(encoding="utf-8")
            for chunk in self._chunk(text):
                self._chunks.append(chunk)
                self._sources.append(fpath.name)

        if not self._chunks:
            return

        # TF-IDF with character n-grams handles Chinese text without tokenisation
        self._vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            max_features=8000,
            sublinear_tf=True,
        )
        self._matrix = self._vectorizer.fit_transform(self._chunks)
        log.info(f"TF-IDF index built: {len(self._chunks)} chunks from {len(md_files)} files")

    # ── Internal: chunking ───────────────────────────────────────────────────

    @staticmethod
    def _chunk(text: str, max_chars: int = 600, overlap: int = 100) -> list[str]:
        """Paragraph-based chunking with sliding overlap."""
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
                tail = current[-overlap:] if len(current) > overlap else current
                current = (tail + "\n\n" + para).strip() if tail else para

        if current:
            chunks.append(current)
        return chunks

    # ── Public API ───────────────────────────────────────────────────────────

    def query(self, text: str, n_results: int = 4) -> list[dict]:
        """
        Retrieve top-n chunks most relevant to the query.
        Returns: [{"document": str, "source": str, "distance": float}]
        """
        if self._vectorizer is None or not self._chunks:
            return []

        from sklearn.metrics.pairwise import cosine_similarity

        q_vec = self._vectorizer.transform([text])
        sims = cosine_similarity(q_vec, self._matrix)[0]

        top_idx = np.argsort(sims)[::-1][:min(n_results, len(self._chunks))]
        return [
            {
                "document": self._chunks[i],
                "source": self._sources[i],
                "distance": round(float(1 - sims[i]), 4),
            }
            for i in top_idx
        ]

    def format_context(self, results: list[dict], max_total_chars: int = 2000) -> str:
        """Format retrieved chunks into a context string for the LLM."""
        if not results:
            return "No relevant knowledge base entries found."

        parts = []
        total = 0
        for r in results:
            entry = f"[Source: {r['source']}]\n{r['document']}"
            if total + len(entry) > max_total_chars:
                remaining = max_total_chars - total
                if remaining > 200:
                    parts.append(entry[:remaining] + "…")
                break
            parts.append(entry)
            total += len(entry)

        return "\n\n---\n\n".join(parts)

    def document_count(self) -> int:
        return len(self._chunks)
