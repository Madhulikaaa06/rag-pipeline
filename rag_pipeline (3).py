"""
RAG Pipeline Core
-----------------
End-to-end Retrieval-Augmented Generation pipeline.

Components:
  - DocumentPreprocessor : cleans and chunks raw text
  - EmbeddingModel       : generates vectors using sentence-transformers
  - VectorStore          : stores and retrieves chunks using ChromaDB
  - LLMGenerator         : generates answers using Groq / Llama 3.1
  - RAGPipeline          : orchestrates all components end-to-end

LLM Used: Llama 3.1 8B via Groq API (free tier)
Embeddings: all-MiniLM-L6-v2 (local, no API needed)
Vector DB: ChromaDB (persistent, cosine similarity)
"""

import hashlib
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import chromadb
from groq import Groq
from sentence_transformers import SentenceTransformer

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ── Data Models ──────────────────────────────────────────────────────────────
@dataclass
class Chunk:
    """Represents a single text chunk from a document."""
    text: str
    source: str
    chunk_index: int
    doc_id: str
    metadata: dict = field(default_factory=dict)

    @property
    def chunk_id(self) -> str:
        """Unique ID based on document and chunk index."""
        raw = self.doc_id + "::" + str(self.chunk_index)
        return hashlib.md5(raw.encode()).hexdigest()


@dataclass
class RetrievalResult:
    """A retrieved chunk with its similarity score."""
    chunk: Chunk
    score: float  # cosine similarity 0-1


@dataclass
class RAGResponse:
    """Final response from the RAG pipeline."""
    answer: str
    sources: list
    query: str
    latency_ms: float


# ── Document Preprocessor ────────────────────────────────────────────────────
class DocumentPreprocessor:
    """
    Cleans and chunks raw text documents.

    Steps:
      1. Remove extra whitespace and non-ASCII characters
      2. Split into sentences
      3. Group sentences into chunks with configurable size
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def ingest(self, text: str, source: str) -> list:
        """Convert raw text into a list of Chunk objects."""
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

    def ingest_file(self, path) -> list:
        """Ingest a text or PDF file."""
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

    @staticmethod
    def _clean(text: str) -> str:
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^\x00-\x7F]+", " ", text)
        return text.strip()

    @staticmethod
    def _split_sentences(text: str) -> list:
        parts = re.split(r"(?<=[.!?])\s+", text)
        return [p.strip() for p in parts if p.strip()]

    def _chunk_sentences(self, sentences: list) -> list:
        chunks, current, current_len = [], [], 0
        for sent in sentences:
            if current_len + len(sent) > self.chunk_size and current:
                chunks.append(" ".join(current))
                overlap_buf, overlap_len = [], 0
                for s in reversed(current):
                    if overlap_len + len(s) <= self.chunk_overlap:
                        overlap_buf.insert(0, s)
                        overlap_len += len(s)
                    else:
                        break
                current, current_len = overlap_buf, overlap_len
            current.append(sent)
            current_len += len(sent)
        if current:
            chunks.append(" ".join(current))
        return chunks

    @staticmethod
    def _read_pdf(path) -> str:
        try:
            import pdfplumber
            with pdfplumber.open(path) as pdf:
                return "\n".join(page.extract_text() or "" for page in pdf.pages)
        except ImportError:
            raise ImportError("Install pdfplumber: pip install pdfplumber")


# ── Embedding Model ──────────────────────────────────────────────────────────
class EmbeddingModel:
    """
    Local embedding model using sentence-transformers.
    Model: all-MiniLM-L6-v2 (384 dimensions, fast, free)
    """

    MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self):
        log.info("Loading embedding model '%s' ...", self.MODEL_NAME)
        self._model = SentenceTransformer(self.MODEL_NAME)
        log.info("Embedding model ready.")

    def embed(self, texts: list) -> list:
        """Embed a list of texts into vectors."""
        if not texts:
            return []
        vectors = self._model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        return vectors.tolist()

    def embed_one(self, text: str) -> list:
        """Embed a single text."""
        return self.embed([text])[0]


# ── Vector Store ─────────────────────────────────────────────────────────────
class VectorStore:
    """
    ChromaDB-backed persistent vector store.
    Uses cosine similarity for retrieval.
    """

    COLLECTION = "rag_documents"

    def __init__(self, persist_dir: str = "./chroma_db"):
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._col = self._client.get_or_create_collection(
            name=self.COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
        log.info("VectorStore ready. Collection has %d chunks.", self._col.count())

    def add_chunks(self, chunks: list, embeddings: list) -> int:
        """Store chunks and their embeddings. Skips duplicates."""
        ids = [c.chunk_id for c in chunks]
        docs = [c.text for c in chunks]
        metas = [{"source": c.source, "chunk_index": c.chunk_index, "doc_id": c.doc_id} for c in chunks]
        self._col.upsert(ids=ids, embeddings=embeddings, documents=docs, metadatas=metas)
        return len(ids)

    def query(self, query_embedding: list, top_k: int = 5) -> list:
        """Find the top-K most similar chunks."""
        count = self._col.count()
        if count == 0:
            return []
        results = self._col.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, count),
            include=["documents", "metadatas", "distances"],
        )
        output = []
        for doc, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            chunk = Chunk(
                text=doc,
                source=meta["source"],
                chunk_index=meta["chunk_index"],
                doc_id=meta["doc_id"],
            )
            output.append(RetrievalResult(chunk=chunk, score=round(1.0 - dist, 4)))
        return output

    def delete_source(self, source: str) -> int:
        """Delete all chunks from a specific source."""
        results = self._col.get(where={"source": source})
        ids = results.get("ids", [])
        if ids:
            self._col.delete(ids=ids)
        return len(ids)

    def list_sources(self) -> list:
        """List all ingested document sources."""
        all_meta = self._col.get(include=["metadatas"])["metadatas"]
        return sorted(set(m["source"] for m in all_meta))

    @property
    def count(self) -> int:
        return self._col.count()


# ── LLM Generator ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a helpful, precise AI assistant.
Answer the user's question using ONLY the context provided below.
If the context does not contain enough information, say so clearly — do not make up answers.
Cite the source document name when relevant."""


class LLMGenerator:
    """
    Generates answers using Groq API (Llama 3.1 8B).
    Free tier, fast inference.
    """

    MODEL = "llama-3.1-8b-instant"

    def __init__(self, api_key: str = None):
        self._client = Groq(api_key=api_key or os.environ.get("GROQ_API_KEY"))

    def generate(self, query: str, context_chunks: list, max_tokens: int = 1024) -> str:
        """Generate a grounded answer from retrieved context."""
        if not context_chunks:
            return "I could not find any relevant context in the knowledge base to answer your question."

        context = self._build_context(context_chunks)
        prompt = (
            "Answer using ONLY the context below. "
            "If not enough info, say so clearly.\n\n"
            "Context:\n" + context + "\nQuestion: " + query
        )

        response = self._client.chat.completions.create(
            model=self.MODEL,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content

    @staticmethod
    def _build_context(chunks: list) -> str:
        parts = []
        for i, r in enumerate(chunks, 1):
            parts.append(
                "[" + str(i) + "] Source: " + r.chunk.source +
                " (relevance: " + str(r.score) + ")\n" + r.chunk.text
            )
        return "\n\n".join(parts)


# ── RAG Pipeline (Main Class) ─────────────────────────────────────────────────
class RAGPipeline:
    """
    End-to-end RAG Pipeline.

    INGEST:  text/file → preprocess → embed → store in ChromaDB
    QUERY:   question  → embed → retrieve top-K → generate answer with LLM
    """

    def __init__(
        self,
        persist_dir: str = "./chroma_db",
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        top_k: int = 5,
        min_score: float = 0.25,
        api_key: str = None,
    ):
        self.top_k = top_k
        self.min_score = min_score
        self.preprocessor = DocumentPreprocessor(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.embedder = EmbeddingModel()
        self.vector_store = VectorStore(persist_dir=persist_dir)
        self.generator = LLMGenerator(api_key=api_key)

    def ingest_text(self, text: str, source: str) -> dict:
        """Ingest raw text into the pipeline."""
        chunks = self.preprocessor.ingest(text, source)
        embeddings = self.embedder.embed([c.text for c in chunks])
        n = self.vector_store.add_chunks(chunks, embeddings)
        log.info("Ingested %d chunks from '%s'.", n, source)
        return {"source": source, "chunks_stored": n}

    def ingest_file(self, path) -> dict:
        """Ingest a file (PDF, TXT, MD) into the pipeline."""
        chunks = self.preprocessor.ingest_file(path)
        embeddings = self.embedder.embed([c.text for c in chunks])
        n = self.vector_store.add_chunks(chunks, embeddings)
        log.info("Ingested %d chunks from '%s'.", n, chunks[0].source)
        return {"source": chunks[0].source, "chunks_stored": n}

    def query(self, question: str) -> RAGResponse:
        """Ask a question and get a grounded answer."""
        if not question.strip():
            raise ValueError("Question cannot be empty.")

        t0 = time.perf_counter()
        q_emb = self.embedder.embed_one(question)
        raw = self.vector_store.query(q_emb, top_k=self.top_k)
        retrieved = [r for r in raw if r.score >= self.min_score]
        answer = self.generator.generate(question, retrieved)
        latency = (time.perf_counter() - t0) * 1000

        log.info("Query answered in %.0f ms | %d chunks used.", latency, len(retrieved))
        return RAGResponse(answer=answer, sources=retrieved, query=question, latency_ms=latency)

    def list_sources(self) -> list:
        """List all ingested document sources."""
        return self.vector_store.list_sources()

    def delete_source(self, source: str) -> int:
        """Delete all chunks from a source."""
        return self.vector_store.delete_source(source)

    @property
    def document_count(self) -> int:
        """Total number of chunks in the vector store."""
        return self.vector_store.count
