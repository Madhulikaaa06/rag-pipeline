# RAG Pipeline

End-to-end Retrieval-Augmented Generation system with a FastAPI backend and browser-based UI.

```
┌─────────────────────────────────────────────────────────────────┐
│                     RAG PIPELINE ARCHITECTURE                   │
│                                                                 │
│  INGEST                                                         │
│  ───────                                                        │
│  Document → Preprocessor → Chunks → EmbeddingModel → ChromaDB  │
│             (clean, split,          (MiniLM L6 v2,   (persist  │
│              overlap)               384-dim)          cosine)   │
│                                                                 │
│  QUERY                                                          │
│  ──────                                                         │
│  Question → EmbeddingModel → ChromaDB (ANN) → Top-K Chunks     │
│                                         ↓                       │
│                               LLM (Claude) + Context           │
│                                         ↓                       │
│                               Grounded Answer + Sources         │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your Anthropic API key

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Run the CLI demo (no server required)

```bash
python demo.py
```

### 4. Start the FastAPI server

```bash
cd app
uvicorn api:app --reload --port 8000
```

### 5. Open the UI

Open `ui/index.html` in your browser — it connects to `http://localhost:8000` by default.

Or visit the interactive API docs at `http://localhost:8000/docs`.

---

## Project Structure

```
rag_pipeline/
├── app/
│   ├── rag_pipeline.py   # Core pipeline (preprocessing, embeddings, vector store, LLM)
│   └── api.py            # FastAPI application
├── ui/
│   └── index.html        # Single-file browser UI
├── tests/
│   └── test_rag.py       # pytest test suite
├── demo.py               # Standalone CLI demo
└── requirements.txt
```

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/documents/text` | Ingest raw text |
| `POST` | `/documents/file` | Ingest uploaded file (PDF/TXT/MD) |
| `GET`  | `/documents`      | List all ingested sources |
| `DELETE` | `/documents/{source}` | Remove a source |
| `POST` | `/query`          | Ask a question |
| `GET`  | `/health`         | System health + stats |

### Ingest text
```bash
curl -X POST http://localhost:8000/documents/text \
  -H "Content-Type: application/json" \
  -d '{"text": "Your document content here.", "source": "my-doc.txt"}'
```

### Ingest file
```bash
curl -X POST http://localhost:8000/documents/file \
  -F "file=@/path/to/document.pdf"
```

### Query
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is RAG?", "top_k": 5, "min_score": 0.25}'
```

## Running Tests

```bash
pytest tests/ -v
```

## Configuration

Set via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | *(required)* | Anthropic API key |
| `CHROMA_DIR` | `./chroma_db` | ChromaDB persistence directory |
| `CHUNK_SIZE` | `512` | Max characters per chunk |
| `CHUNK_OVERLAP` | `64` | Overlap between adjacent chunks |
| `TOP_K` | `5` | Number of chunks to retrieve |
| `MIN_SCORE` | `0.25` | Minimum cosine similarity threshold |

## Design Decisions

**Embeddings**: `all-MiniLM-L6-v2` runs locally (no API cost), produces 384-dim normalized vectors, and achieves strong retrieval performance across general domains.

**Vector store**: ChromaDB with DuckDB+Parquet backend gives persistent, zero-infrastructure storage suitable for development and moderate production use.

**Chunking**: Sentence-aware splitting with configurable overlap prevents context loss at chunk boundaries. Overlapping windows ensure no sentence is orphaned.

**Score filtering**: A `min_score` threshold prevents low-relevance chunks from polluting the LLM context, reducing hallucination risk.

**LLM grounding**: The system prompt explicitly instructs Claude to answer only from provided context and cite sources, making the system auditable.
