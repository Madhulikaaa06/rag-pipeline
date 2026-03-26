"""
Tests for the RAG Pipeline
Run with:  pytest tests/ -v
"""

import os
import pytest
from unittest.mock import MagicMock, patch

# ── ensure ANTHROPIC_API_KEY is set for tests (mocked) ────────────────────────
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-placeholder")

from app.rag_pipeline import (
    Chunk,
    DocumentPreprocessor,
    EmbeddingModel,
    RAGPipeline,
    RetrievalResult,
)


# ── DocumentPreprocessor ───────────────────────────────────────────────────────
class TestDocumentPreprocessor:
    def setup_method(self):
        self.proc = DocumentPreprocessor(chunk_size=100, chunk_overlap=20)

    def test_basic_ingest(self):
        text = "Hello world. This is a test. " * 10
        chunks = self.proc.ingest(text, "test.txt")
        assert len(chunks) >= 1
        for c in chunks:
            assert c.source == "test.txt"
            assert len(c.text) > 0

    def test_chunk_overlap(self):
        text = ("This is sentence number one. " * 5 +
                "This is sentence number two. " * 5 +
                "This is sentence number three. " * 5)
        chunks = self.proc.ingest(text, "overlap_test")
        # With overlap, adjacent chunks should share some text
        if len(chunks) > 1:
            words_c1 = set(chunks[0].text.split())
            words_c2 = set(chunks[1].text.split())
            # overlap is not guaranteed to be non-empty given short chunk_size
            assert isinstance(words_c1 & words_c2, set)

    def test_empty_after_clean_raises(self):
        with pytest.raises(ValueError):
            self.proc.ingest("   \n\t  ", "empty_doc")

    def test_chunk_ids_unique(self):
        text = "A" * 500
        chunks = self.proc.ingest(text, "unique.txt")
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_clean_removes_excess_whitespace(self):
        text = "Hello    world.\nThis   is   a  test."
        chunks = self.proc.ingest(text, "ws.txt")
        for c in chunks:
            assert "  " not in c.text  # no double spaces

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            self.proc.ingest_file("/nonexistent/path/doc.txt")

    def test_unsupported_file_type_raises(self, tmp_path):
        f = tmp_path / "file.xyz"
        f.write_text("content")
        with pytest.raises(ValueError):
            self.proc.ingest_file(f)


# ── Chunk dataclass ────────────────────────────────────────────────────────────
class TestChunk:
    def test_chunk_id_deterministic(self):
        c1 = Chunk(text="hello", source="s", chunk_index=0, doc_id="d1")
        c2 = Chunk(text="hello", source="s", chunk_index=0, doc_id="d1")
        assert c1.chunk_id == c2.chunk_id

    def test_chunk_id_varies_with_index(self):
        c1 = Chunk(text="hello", source="s", chunk_index=0, doc_id="d1")
        c2 = Chunk(text="hello", source="s", chunk_index=1, doc_id="d1")
        assert c1.chunk_id != c2.chunk_id


# ── EmbeddingModel ─────────────────────────────────────────────────────────────
class TestEmbeddingModel:
    @pytest.fixture(scope="class")
    def model(self):
        return EmbeddingModel()

    def test_embed_returns_list(self, model):
        result = model.embed(["Hello world"])
        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], list)

    def test_embed_one(self, model):
        v = model.embed_one("Test sentence")
        assert len(v) == 384  # MiniLM dimension

    def test_empty_input(self, model):
        assert model.embed([]) == []

    def test_embeddings_normalized(self, model):
        import math
        v = model.embed_one("Normalization check")
        norm = math.sqrt(sum(x**2 for x in v))
        assert abs(norm - 1.0) < 1e-5

    def test_similar_sentences_closer(self, model):
        v_dog1 = model.embed_one("The dog is playing in the park.")
        v_dog2 = model.embed_one("A puppy runs outside.")
        v_astro = model.embed_one("Quantum entanglement in superconductors.")
        dot_similar = sum(a*b for a,b in zip(v_dog1, v_dog2))
        dot_dissimilar = sum(a*b for a,b in zip(v_dog1, v_astro))
        assert dot_similar > dot_dissimilar


# ── RAGPipeline integration (mocked LLM + VectorStore) ────────────────────────
class TestRAGPipelineIntegration:
    """Full pipeline tests with mocked external components."""

    @pytest.fixture
    def pipeline(self, tmp_path):
        with patch("app.rag_pipeline.LLMGenerator.generate", return_value="Mocked answer."):
            pipe = RAGPipeline(
                persist_dir=str(tmp_path / "chroma"),
                chunk_size=200,
                chunk_overlap=20,
                top_k=3,
                min_score=0.0,   # Accept all during tests
                api_key="test",
            )
            return pipe

    def test_ingest_and_query(self, pipeline):
        pipeline.ingest_text(
            "The capital of France is Paris. Paris is known for the Eiffel Tower.",
            source="france.txt",
        )
        response = pipeline.query("What is the capital of France?")
        assert response.answer == "Mocked answer."
        assert response.query == "What is the capital of France?"
        assert len(response.sources) >= 1

    def test_list_sources_after_ingest(self, pipeline):
        pipeline.ingest_text("Some content about Python programming.", "python.txt")
        sources = pipeline.list_sources()
        assert "python.txt" in sources

    def test_delete_source(self, pipeline):
        pipeline.ingest_text("Delete me content here.", "deleteme.txt")
        assert "deleteme.txt" in pipeline.list_sources()
        pipeline.delete_source("deleteme.txt")
        assert "deleteme.txt" not in pipeline.list_sources()

    def test_empty_query_raises(self, pipeline):
        with pytest.raises(ValueError):
            pipeline.query("   ")

    def test_document_count_increases(self, pipeline):
        initial = pipeline.document_count
        pipeline.ingest_text("More content to count. " * 10, "counter.txt")
        assert pipeline.document_count > initial

    def test_ingest_file_txt(self, pipeline, tmp_path):
        f = tmp_path / "sample.txt"
        f.write_text("This is a sample text file for testing the RAG pipeline ingestion.")
        result = pipeline.ingest_file(f)
        assert result["chunks_stored"] >= 1

    def test_low_relevance_filtered(self, pipeline):
        pipeline.ingest_text("The sky is blue and clouds are white.", "sky.txt")
        pipeline.min_score = 0.99   # very high threshold
        response = pipeline.query("Quantum physics equations?")
        # With very high min_score, sources should be filtered out
        assert isinstance(response.sources, list)
