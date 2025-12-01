"""Dimensionality reduction using UMAP."""

import numpy as np
from typing import Dict
from umap import UMAP


def compute_umap_embedding(
    fingerprints: np.ndarray, 
    config: Dict,
    n_neighbors: int = None,
    min_dist: float = None
) -> np.ndarray:
    """Compute UMAP embedding of molecular fingerprints."""
    umap_config = config['umap']

    n_neighbors = n_neighbors if n_neighbors is not None else umap_config['n_neighbors']
    min_dist = min_dist if min_dist is not None else umap_config['min_dist']

    reducer = UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=umap_config['n_components'],
        metric=umap_config['metric'],
        random_state=umap_config['random_state'],
        n_jobs=1
    )

    embedding = reducer.fit_transform(fingerprints)
    return embedding
