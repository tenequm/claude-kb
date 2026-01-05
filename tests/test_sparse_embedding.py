"""Tests for PyTorch SPLADE sparse embedding model."""

import numpy as np
import pytest


class TestSparseEmbeddingModel:
    """Test SparseEmbeddingModel produces compatible output format."""

    @pytest.fixture
    def model(self):
        """Create a sparse embedding model instance."""
        from claude_kb.db import SparseEmbeddingModel

        return SparseEmbeddingModel()

    def test_output_format(self, model):
        """Verify output has .indices and .values as numpy arrays."""
        results = model.encode(["test query for sparse embedding"])

        assert len(results) == 1
        sparse = results[0]

        # Check attributes exist
        assert hasattr(sparse, "indices")
        assert hasattr(sparse, "values")

        # Check types
        assert isinstance(sparse.indices, np.ndarray)
        assert isinstance(sparse.values, np.ndarray)

        # Check dtypes match expected format
        assert sparse.indices.dtype == np.int32
        assert sparse.values.dtype == np.float32

        # Check shapes match
        assert sparse.indices.shape == sparse.values.shape

        # Check we have non-zero activations
        assert len(sparse.indices) > 0

    def test_batch_encoding(self, model):
        """Verify batch encoding works correctly."""
        texts = [
            "first document about machine learning",
            "second document about natural language processing",
            "third document about computer vision",
        ]
        results = model.encode(texts)

        assert len(results) == 3
        for sparse in results:
            assert isinstance(sparse.indices, np.ndarray)
            assert isinstance(sparse.values, np.ndarray)
            assert len(sparse.indices) > 0

    def test_tolist_compatibility(self, model):
        """Verify .tolist() works for Qdrant compatibility."""
        results = model.encode(["test"])
        sparse = results[0]

        # This is how it's used in cli.py and import_claude.py
        indices_list = sparse.indices.tolist()
        values_list = sparse.values.tolist()

        assert isinstance(indices_list, list)
        assert isinstance(values_list, list)
        assert all(isinstance(i, int) for i in indices_list)
        assert all(isinstance(v, float) for v in values_list)

    def test_empty_input(self, model):
        """Verify empty input is handled."""
        results = model.encode([])
        assert results == []

    def test_single_word(self, model):
        """Verify single word input works."""
        results = model.encode(["hello"])
        assert len(results) == 1
        assert len(results[0].indices) > 0
