"""Clustering and cluster analysis."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances


def perform_clustering(
    embedding: np.ndarray,
    config: Dict,
    n_clusters: int = None
) -> np.ndarray:
    """Perform K-Means clustering on UMAP embedding."""
    kmeans_config = config['kmeans']
    n_clusters = n_clusters if n_clusters is not None else kmeans_config['default_n_clusters']
    n_clusters = max(kmeans_config['min_clusters'], 
                     min(n_clusters, kmeans_config['max_clusters']))
    n_clusters = min(n_clusters, len(embedding))

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=kmeans_config['random_state'],
        n_init=kmeans_config['n_init']
    )
    return kmeans.fit_predict(embedding)


def get_cluster_representatives(
    cluster_id: int,
    df: pd.DataFrame,
    fingerprints: np.ndarray,
    embedding: np.ndarray,
    labels: np.ndarray,
    config: Dict
) -> Tuple[List[int], int]:
    """Get representative molecules for a cluster."""
    n_reps = config['cluster']['n_representatives']
    cluster_mask = labels == cluster_id
    cluster_indices = np.where(cluster_mask)[0]

    if len(cluster_indices) == 0:
        return [], -1

    cluster_fps = fingerprints[cluster_mask]

    if len(cluster_indices) == 1:
        medoid_local_idx = 0
    else:
        distances = pairwise_distances(cluster_fps, metric='jaccard')
        medoid_local_idx = np.argmin(distances.sum(axis=1))

    medoid_global_idx = cluster_indices[medoid_local_idx]

    if len(cluster_indices) <= n_reps:
        representatives = cluster_indices.tolist()
    else:
        medoid_fp = cluster_fps[medoid_local_idx].reshape(1, -1)
        distances_to_medoid = pairwise_distances(
            cluster_fps, medoid_fp, metric='jaccard'
        ).flatten()
        closest_local = np.argsort(distances_to_medoid)[:n_reps]
        representatives = cluster_indices[closest_local].tolist()

    return representatives, medoid_global_idx


def get_cluster_statistics(
    cluster_id: int,
    df: pd.DataFrame,
    labels: np.ndarray,
    label_column: str = None
) -> Dict:
    """Get statistics for a cluster."""
    cluster_mask = labels == cluster_id
    cluster_df = df[cluster_mask]

    stats = {
        'size': int(cluster_mask.sum()),
        'mean_mw': float(cluster_df['Molecular Weight'].mean()),
        'mean_logp': float(cluster_df['LogP'].mean()),
        'mean_tpsa': float(cluster_df['TPSA'].mean())
    }

    if label_column and label_column in df.columns:
        label_counts = cluster_df[label_column].value_counts().to_dict()
        stats['label_distribution'] = label_counts

    return stats
