"""
RAG Pipeline Core
-----------------
Handles document ingestion, chunking, embedding, vector storage, retrieval, and generation.
Uses:  sentence-transformers (embeddings) + ChromaDB (vector store) + Anthropic Claude (LLM)
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anthropic
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ── Data Models ────────────────────────────────────────────────────────────────
@dataclass
class Chunk:
    text: str
    source: str
    chunk_index: int
    doc_id: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def chunk_id(self) -> str:
        raw = f"{self.doc_id}::{self.chunk_index}"
        return hashlib.md5(raw.encode()).hexdigest()


@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float          # cosine similarity 0-1


@dataclass
class RAGResponse:
    answer: str
    sources: list[RetrievalResult]
    query: str
    latency_ms: float


# ── Document Preprocessor ──────────────────────────────────────────────────────
class DocumentPreprocessor:
    """Cleans and chunks raw text documents."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # ── public ──
    def ingest(self, text: str, source: str) -> list[Chunk]:
        cleaned = self._clean(text)
        if not cleaned:
            raise ValueError(f"Document '{source}' is empty after cleaning.")
        sentences = self._split_sentences(cleaned)
        raw_chunks = self._chunk_sentences(sentences)
        doc_id = hashlib.md5(source.encode()).hexdigest()
        return [
            Chunk(text=c, source=source, chunk_index=i, doc_id=doc_id)
            for i, c in enumerate(raw_chunks)
        ]

    def ingest_file(self, path: str | Path) -> list[Chunk]:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            text = self._read_pdf(path)
        elif suffix in {".txt", ".md", ".rst"}:
            text = path.read_text(encoding="utf-8", errors="replace")
        else:
            raise ValueError(f"Unsupported file type: {suffix}")
        return self.ingest(text, source=str(path.name))

    # ── private ──
    @staticmethod
    def _clean(text: str) -> str:
        text = re.sub(r"\s+", " ", text)          # collapse whitespace
        text = re.sub(r"[^\x00-\x7F]+", " ", text)  # strip non-ASCII
        return text.strip()

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        # Simple but robust sentence splitter
        parts = re.split(r"(?<=[.!?])\s+", text)
        return [p.strip() for p in parts if p.strip()]

    def _chunk_sentences(self, sentences: list[str]) -> list[str]:
        chunks, current, current_len = [], [], 0
        for sent in sentences:
            sent_len = len(sent)
            if current_len + sent_len > self.chunk_size and current:
                chunks.append(" ".join(current))
                # overlap: keep last sentences that fit in overlap window
                overlap_buf, overlap_len = [], 0
                for s in reversed(current):
                    if overlap_len + len(s) <= self.chunk_overlap:
                        overlap_buf.insert(0, s)
                        overlap_len += len(s)
                    else:
                        break
                current, current_len = overlap_buf, overlap_len
            current.append(sent)
            current_len += sent_len
        if current:
            chunks.append(" ".join(current))
        return chunks

    @staticmethod
    def _read_pdf(path: Path) -> str:
        try:
            import pdfplumber
            with pdfplumber.open(path) as pdf:
                return "\n".join(
                    page.extract_text() or "" for page in pdf.pages
                )
        except ImportError:
            raise ImportError("Install pdfplumber to ingest PDF files: pip install pdfplumber")


# ── Embedding Model ────────────────────────────────────────────────────────────
class EmbeddingModel:
    """Wraps sentence-transformers for local, free embeddings."""

    MODEL_NAME = "all-MiniLM-L6-v2"   # fast, 384-dim, great for RAG

    def __init__(self):
        log.info("Loading embedding model '%s' …", self.MODEL_NAME)
        self._model = SentenceTransformer(self.MODEL_NAME)
        log.info("Embedding model ready.")

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        return vectors.tolist()

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


# ── Vector Store ───────────────────────────────────────────────────────────────
class VectorStore:
    """ChromaDB-backed persistent vector store."""

    COLLECTION = "rag_documents"

    def __init__(self, persist_dir: str = "./chroma_db"):
        self._client = chromadb.Client(
            Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=persist_dir,
                anonymized_telemetry=False,
            )
        )
        self._col = self._client.get_or_create_collection(
            name=self.COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        log.info("VectorStore ready. Collection '%s' has %d docs.", self.COLLECTION, self._col.count())

    # ── public ──
    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> int:
        """Upsert chunks (skips duplicates by chunk_id)."""
        ids = [c.chunk_id for c in chunks]
        docs = [c.text for c in chunks]
        metas = [{"source": c.source, "chunk_index": c.chunk_index, "doc_id": c.doc_id} for c in chunks]
        self._col.upsert(ids=ids, embeddings=embeddings, documents=docs, metadatas=metas)
        self._client.persist()
        return len(ids)

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        where: dict | None = None,
    ) -> list[RetrievalResult]:
        kwargs: dict[str, Any] = dict(
            query_embeddings=[query_embedding],
            n_results=min(top_k, max(1, self._col.count())),
            include=["documents", "metadatas", "distances"],
        )
        if where:
            kwargs["where"] = where

        res = self._col.query(**kwargs)
        results = []
        for doc, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            chunk = Chunk(
                text=doc,
                source=meta["source"],
                chunk_index=meta["chunk_index"],
                doc_id=meta["doc_id"],
            )
            # ChromaDB cosine distance → similarity
            score = 1.0 - dist
            results.append(RetrievalResult(chunk=chunk, score=score))
        return results

    def delete_source(self, source: str) -> int:
        results = self._col.get(where={"source": source})
        ids = results.get("ids", [])
        if ids:
            self._col.delete(ids=ids)
            self._client.persist()
        return len(ids)

    def list_sources(self) -> list[str]:
        all_meta = self._col.get(include=["metadatas"])["metadatas"]
        return sorted({m["source"] for m in all_meta})

    @property
    def count(self) -> int:
        return self._col.count()


# ── LLM Generator ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are a helpful, precise AI assistant.
Answer the user's question using ONLY the context provided below.
If the context does not contain enough information, say so clearly — do not hallucinate.
Cite the source document name(s) when relevant.
"""


class LLMGenerator:
    """Calls Claude via the Anthropic SDK."""

    MODEL = "claude-sonnet-4-20250514"

    def __init__(self, api_key: str | None = None):
        self._client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    def generate(self, query: str, context_chunks: list[RetrievalResult], max_tokens: int = 1024) -> str:
        if not context_chunks:
            return "I could not find any relevant context in the knowledge base to answer your question."

        context_block = self._build_context(context_chunks)
        user_message = f"Context:\n{context_block}\n\nQuestion: {query}"

        response = self._client.messages.create(
            model=self.MODEL,
            max_tokens=max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        return response.content[0].text

    @staticmethod
    def _build_context(chunks: list[RetrievalResult]) -> str:
        parts = []
        for i, r in enumerate(chunks, 1):
            parts.append(
                f"[{i}] Source: {r.chunk.source} (relevance: {r.score:.2f})\n{r.chunk.text}"
            )
        return "\n\n".join(parts)


# ── RAG Pipeline (Facade) ──────────────────────────────────────────────────────
class RAGPipeline:
    """
    Orchestrates the full RAG workflow:
      ingest  →  embed  →  store
      query   →  retrieve  →  generate
    """

    def __init__(
        self,
        persist_dir: str = "./chroma_db",
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        top_k: int = 5,
        min_score: float = 0.25,
        api_key: str | None = None,
    ):
        self.top_k = top_k
        self.min_score = min_score
        self.preprocessor = DocumentPreprocessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.embedder = EmbeddingModel()
        self.vector_store = VectorStore(persist_dir=persist_dir)
        self.generator = LLMGenerator(api_key=api_key)

    # ── ingestion ──
    def ingest_text(self, text: str, source: str) -> dict:
        chunks = self.preprocessor.ingest(text, source)
        return self._store_chunks(chunks)

    def ingest_file(self, path: str | Path) -> dict:
        chunks = self.preprocessor.ingest_file(path)
        return self._store_chunks(chunks)

    def _store_chunks(self, chunks: list[Chunk]) -> dict:
        embeddings = self.embedder.embed([c.text for c in chunks])
        n = self.vector_store.add_chunks(chunks, embeddings)
        log.info("Ingested %d chunks from '%s'.", n, chunks[0].source)
        return {"source": chunks[0].source, "chunks_stored": n}

    # ── query ──
    def query(self, question: str) -> RAGResponse:
        if not question.strip():
            raise ValueError("Query must not be empty.")
        t0 = time.perf_counter()

        q_emb = self.embedder.embed_one(question)
        raw = self.vector_store.query(q_emb, top_k=self.top_k)
        retrieved = [r for r in raw if r.score >= self.min_score]

        answer = self.generator.generate(question, retrieved)
        latency = (time.perf_counter() - t0) * 1000

        log.info("Query answered in %.0f ms  |  %d chunks used.", latency, len(retrieved))
        return RAGResponse(answer=answer, sources=retrieved, query=question, latency_ms=latency)

    # ── management ──
    def list_sources(self) -> list[str]:
        return self.vector_store.list_sources()

    def delete_source(self, source: str) -> int:
        return self.vector_store.delete_source(source)

    @property
    def document_count(self) -> int:
        return self.vector_store.count
