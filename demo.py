"""
demo.py — Standalone CLI demo of the RAG pipeline (no server needed)
Usage:  ANTHROPIC_API_KEY=sk-... python demo.py
"""

import os, sys
sys.path.insert(0, "app")

from rag_pipeline import RAGPipeline

SAMPLE_DOCS = [
    (
        "transformers.txt",
        """
        Transformers are a type of deep learning model introduced in the paper
        "Attention Is All You Need" by Vaswani et al. (2017). They rely entirely
        on self-attention mechanisms, dispensing with recurrence and convolutions.
        The architecture consists of an encoder and decoder, each made of stacked
        layers containing multi-head self-attention and feed-forward sub-layers.
        Positional encodings are added to token embeddings to preserve sequence order.
        Transformers power most modern large language models including GPT, BERT, and Claude.
        """,
    ),
    (
        "rag_overview.txt",
        """
        Retrieval-Augmented Generation (RAG) is a technique that enhances large language
        models by grounding their responses in retrieved documents. The pipeline works in
        two phases: first, documents are chunked and embedded into a vector database;
        second, at query time, the question is embedded and the most similar document
        chunks are retrieved and provided as context to the LLM. This reduces hallucination
        and allows the model to answer questions about private or recent information
        that was not part of its training data. RAG is widely used in enterprise AI systems.
        """,
    ),
    (
        "vector_db.txt",
        """
        A vector database stores high-dimensional numerical vectors representing semantic
        content. Similarity search is performed using approximate nearest neighbour (ANN)
        algorithms such as HNSW (Hierarchical Navigable Small World graphs). ChromaDB,
        Pinecone, Weaviate, and Qdrant are popular vector database solutions. The distance
        metric used determines retrieval quality: cosine similarity is preferred for
        text embeddings as it measures directional similarity independent of magnitude.
        Vector databases enable sub-millisecond retrieval across millions of documents.
        """,
    ),
]

QUERIES = [
    "What is a transformer model and who invented it?",
    "How does RAG reduce hallucination in LLMs?",
    "What distance metric is best for text embeddings and why?",
    "What happens during the query phase of a RAG system?",
]


def main():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("⚠  Set ANTHROPIC_API_KEY environment variable first.")
        sys.exit(1)

    print("\n" + "═"*60)
    print("  RAG PIPELINE DEMO")
    print("═"*60)

    pipe = RAGPipeline(persist_dir="./demo_chroma", api_key=api_key)

    # ── 1. Ingest ──────────────────────────────────────────────────────────
    print("\n[1/3] INGESTING SAMPLE DOCUMENTS …\n")
    for source, text in SAMPLE_DOCS:
        result = pipe.ingest_text(text.strip(), source)
        print(f"  ✓  {result['source']:25s}  →  {result['chunks_stored']} chunks")

    print(f"\n  Total chunks in store: {pipe.document_count}")
    print(f"  Sources: {pipe.list_sources()}")

    # ── 2. Query ───────────────────────────────────────────────────────────
    print("\n[2/3] RUNNING QUERIES …\n")
    for q in QUERIES:
        print(f"  Q: {q}")
        response = pipe.query(q)
        print(f"  A: {response.answer[:280]}{'…' if len(response.answer)>280 else ''}")
        print(f"     [{len(response.sources)} chunks used  |  {response.latency_ms:.0f}ms]\n")

    # ── 3. Cleanup ─────────────────────────────────────────────────────────
    print("[3/3] CLEANUP — deleting demo sources …")
    for source, _ in SAMPLE_DOCS:
        pipe.delete_source(source)
    print("  Done.\n")


if __name__ == "__main__":
    main()
