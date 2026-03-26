# RAG Pipeline

End-to-end **Retrieval-Augmented Generation (RAG)** system built in Python.

---

## What is RAG?

RAG is a technique that makes AI answer questions using **your own documents** instead of just its training data. It works in two phases:

1. **Ingest** — Documents are cleaned, chunked, and stored as vectors in a database
2. **Query** — Questions are matched to relevant chunks and an LLM generates a grounded answer

---

## Architecture

```
INGEST PHASE:
Document → Clean & Chunk → Embed (MiniLM) → Store (ChromaDB)

QUERY PHASE:
Question → Embed → Retrieve Top-K Chunks → Generate Answer (Groq/Llama)
```

---

## Technologies Used

| Component | Technology |
|-----------|-----------|
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector Database | ChromaDB (cosine similarity) |
| LLM | Llama 3.1 8B via Groq API (free) |
| API | FastAPI |
| Language | Python 3.9+ |

---

## Project Structure

```
rag-pipeline/
├── rag_pipeline.py      # Core pipeline (preprocessing, embeddings, vector store, LLM)
├── api.py               # FastAPI REST API
├── ui/
│   └── index.html       # Browser-based UI
├── RAG_Pipeline.ipynb   # Google Colab notebook demo
├── requirements.txt     # Python dependencies
└── README.md
```

---

## Quick Start

### Option 1 — Google Colab (easiest)
Open `RAG_Pipeline.ipynb` in Google Colab and run all cells.

### Option 2 — Run locally

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Set your Groq API key**
```bash
export GROQ_API_KEY=gsk_your_key_here
```
Get a free key at: https://console.groq.com

**3. Start the API server**
```bash
uvicorn api:app --reload --port 8000
```

**4. Open the UI**
Open `ui/index.html` in your browser.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/documents/text` | Ingest raw text |
| `POST` | `/documents/file` | Ingest PDF/TXT/MD file |
| `GET` | `/documents` | List all sources |
| `DELETE` | `/documents/{source}` | Delete a source |
| `POST` | `/query` | Ask a question |
| `GET` | `/health` | System health |

---

## Key Design Decisions

- **Local Embeddings** — runs locally with no API cost
- **ChromaDB** — persistent vector storage, data survives restarts
- **Groq / Llama 3.1** — free API, LLM answers ONLY from provided context
- **Score Filtering** — filters irrelevant chunks before sending to LLM
- **Sentence-aware Chunking** — splits at sentence boundaries with overlap

---

## Getting a Free Groq API Key

1. Go to https://console.groq.com
2. Sign up with Google
3. Click API Keys → Create API Key
4. Copy the key (starts with gsk_...)
