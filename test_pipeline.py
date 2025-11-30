"""Test suite for Chemical Space Explorer pipeline."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import load_config, validate_smiles
from src.featurization import compute_fingerprints, compute_properties
from src.embedding import compute_umap_embedding
from src.clustering import perform_clustering


@pytest.fixture
def config():
    """Load configuration."""
    return load_config("config.yaml")


@pytest.fixture
def sample_data():
    """Create sample molecular data."""
    smiles_list = [
        "CCO", "CC(C)O", "CCCO", "c1ccccc1", "c1ccccc1O",
        "c1ccccc1C", "CC(=O)O", "CCC(=O)O", "CCCC(=O)O", "CC(C)C"
    ]
    return pd.DataFrame({
        'smiles': smiles_list,
        'name': [f'mol_{i}' for i in range(len(smiles_list))]
    })


class TestValidation:
    """Test SMILES validation."""

    def test_valid_smiles(self):
        is_valid, canonical = validate_smiles("CCO")
        assert is_valid
        assert canonical is not None

    def test_invalid_smiles(self):
        is_valid, canonical = validate_smiles("INVALID")
        assert not is_valid
        assert canonical is None


class TestFeaturization:
    """Test molecular featurization."""

    def test_fingerprint_computation(self, config, sample_data):
        fingerprints = compute_fingerprints(sample_data, config)
        assert fingerprints.shape[0] == len(sample_data)
        assert fingerprints.shape[1] == config['fingerprint']['n_bits']
        assert not np.isnan(fingerprints).any()
        assert np.all((fingerprints == 0) | (fingerprints == 1))

    def test_property_computation(self, sample_data):
        df = compute_properties(sample_data)
        required_props = ['LogP', 'Molecular Weight', 'TPSA']
        for prop in required_props:
            assert prop in df.columns


class TestEmbedding:
    """Test UMAP embedding."""

    def test_umap_embedding(self, config, sample_data):
        fingerprints = compute_fingerprints(sample_data, config)
        embedding = compute_umap_embedding(fingerprints, config)
        assert embedding.shape[0] == len(sample_data)
        assert embedding.shape[1] == 2
        assert not np.isnan(embedding).any()

    def test_umap_deterministic(self, config, sample_data):
        fingerprints = compute_fingerprints(sample_data, config)
        embedding1 = compute_umap_embedding(fingerprints, config)
        embedding2 = compute_umap_embedding(fingerprints, config)
        np.testing.assert_array_almost_equal(embedding1, embedding2)


class TestClustering:
    """Test K-Means clustering."""

    def test_clustering(self, config, sample_data):
        fingerprints = compute_fingerprints(sample_data, config)
        embedding = compute_umap_embedding(fingerprints, config)
        labels = perform_clustering(embedding, config, n_clusters=3)
        assert len(labels) == len(sample_data)
        assert labels.min() >= 0
        assert labels.max() < 3


class TestPipeline:
    """Test full pipeline integration."""

    def test_full_pipeline(self, config, sample_data):
        df = compute_properties(sample_data)
        fingerprints = compute_fingerprints(df, config)
        embedding = compute_umap_embedding(fingerprints, config)
        labels = perform_clustering(embedding, config)

        assert len(df) == len(sample_data)
        assert fingerprints.shape[0] == len(df)
        assert embedding.shape[0] == len(df)
        assert len(labels) == len(df)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
