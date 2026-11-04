"""
Classical Distance Baselines
============================
Implements point-to-point multivariate Euclidean distance and Sakoe-Chiba constrained
Classical Dependent Dynamic Time Warping (DTW_D).
"""

import numpy as np
import torch
from scipy.spatial.distance import cdist


def multivariate_euclidean(seq1: torch.Tensor, seq2: torch.Tensor) -> float:
    """Computes the standard Frobenius/Euclidean norm between two multivariate time series."""
    return torch.norm(seq1 - seq2).item()


def multivariate_classical_dtw(seq1: torch.Tensor, seq2: torch.Tensor, window: int = 15) -> float:
    """
    Optimized Dependent Classical DTW (DTW_D).
    Precomputes the pointwise Euclidean distance matrix using vectorized C routines
    and performs sequential dynamic programming restricted to the Sakoe-Chiba band.
    """
    s1 = seq1.detach().cpu().numpy()
    s2 = seq2.detach().cpu().numpy()

    N, _ = s1.shape
    M, _ = s2.shape

    dist_matrix = cdist(s1, s2, metric="euclidean")

    dtw_matrix = np.full((N + 1, M + 1), np.inf)
    dtw_matrix[0, 0] = 0.0

    for i in range(1, N + 1):
        start_j = max(1, i - window)
        end_j = min(M + 1, i + window + 1)

        for j in range(start_j, end_j):
            cost = dist_matrix[i - 1, j - 1]
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],      # insertion
                dtw_matrix[i, j - 1],      # deletion
                dtw_matrix[i - 1, j - 1],  # match
            )

    return float(dtw_matrix[N, M])
