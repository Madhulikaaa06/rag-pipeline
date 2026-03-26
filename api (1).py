"""
FastAPI Backend for RAG Pipeline
----------------------------------
Provides REST API endpoints for the RAG pipeline.

Endpoints:
  POST   /documents/text       - Ingest raw text
  POST   /documents/file       - Ingest uploaded file (PDF/TXT/MD)
  GET    /documents            - List all ingested sources
  DELETE /documents/{source}   - Delete a source
  POST   /query                - Ask a question
  GET    /health               - System health check

LLM: Groq / Llama 3.1 8B (free tier)
"""

import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from rag_pipeline import RAGPipeline

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="RAG Pipeline API",
    description="End-to-end Retrieval-Augmented Generation system using Groq + ChromaDB",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Singleton pipeline ────────────────────────────────────────────────────────
_pipeline = None


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline(
            persist_dir=os.getenv("CHROMA_DIR", "./chroma_db"),
            chunk_size=int(os.getenv("CHUNK_SIZE", "512")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "64")),
            top_k=int(os.getenv("TOP_K", "5")),
            min_score=float(os.getenv("MIN_SCORE", "0.25")),
            api_key=os.getenv("GROQ_API_KEY"),
        )
    return _pipeline


# ── Request / Response Schemas ────────────────────────────────────────────────
class IngestTextRequest(BaseModel):
    text: str = Field(..., min_length=10, description="Document text content")
    source: str = Field(..., min_length=1, description="Document name or source")


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, description="Natural language question")
    top_k: int = Field(default=5, ge=1, le=20)
    min_score: float = Field(default=0.25, ge=0.0, le=1.0)


class IngestResponse(BaseModel):
    source: str
    chunks_stored: int
    message: str


class SourceResult(BaseModel):
    chunk_index: int
    source: str
    score: float
    excerpt: str


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: list
    latency_ms: float
    chunks_used: int


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    """Check system health and list ingested sources."""
    pipe = get_pipeline()
    return {
        "status": "ok",
        "chunks_in_store": pipe.document_count,
        "sources": pipe.list_sources(),
        "llm": "Groq / llama-3.1-8b-instant",
        "embeddings": "all-MiniLM-L6-v2",
    }


@app.post("/documents/text", response_model=IngestResponse, status_code=201)
def ingest_text(req: IngestTextRequest):
    """Ingest raw text into the pipeline."""
    try:
        result = get_pipeline().ingest_text(req.text, req.source)
        return IngestResponse(**result, message="Document ingested successfully.")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")


@app.post("/documents/file", response_model=IngestResponse, status_code=201)
async def ingest_file(file: UploadFile = File(...)):
    """Ingest an uploaded file (PDF, TXT, MD)."""
    allowed = {".pdf", ".txt", ".md", ".rst"}
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in allowed:
        raise HTTPException(status_code=415, detail=f"Unsupported file type '{suffix}'. Allowed: {allowed}")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(await file.read())
            tmp_path = Path(tmp.name)

        result = get_pipeline().ingest_file(tmp_path)
        result["source"] = file.filename
        return IngestResponse(**result, message="File ingested successfully.")
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File ingestion failed: {e}")
    finally:
        if tmp_path:
            tmp_path.unlink(missing_ok=True)


@app.get("/documents")
def list_documents():
    """List all ingested document sources."""
    return get_pipeline().list_sources()


@app.delete("/documents/{source}")
def delete_document(source: str):
    """Delete all chunks from a specific source."""
    n = get_pipeline().delete_source(source)
    if n == 0:
        raise HTTPException(status_code=404, detail=f"Source '{source}' not found.")
    return {"deleted_chunks": n, "source": source}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    """Ask a question and get a grounded answer."""
    pipe = get_pipeline()
    pipe.top_k = req.top_k
    pipe.min_score = req.min_score

    try:
        rag = pipe.query(req.question)
        sources = [
            {
                "source": r.chunk.source,
                "chunk_index": r.chunk.chunk_index,
                "score": r.score,
                "excerpt": r.chunk.text[:300] + ("..." if len(r.chunk.text) > 300 else ""),
            }
            for r in rag.sources
        ]
        return QueryResponse(
            query=rag.query,
            answer=rag.answer,
            sources=sources,
            latency_ms=round(rag.latency_ms, 1),
            chunks_used=len(sources),
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")


# ── Error handler ─────────────────────────────────────────────────────────────
@app.exception_handler(Exception)
async def generic_handler(request, exc):
    return JSONResponse(status_code=500, content={"detail": str(exc)})
