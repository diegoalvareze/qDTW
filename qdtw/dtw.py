"""
Sakoe-Chiba Quantum Dynamic Time Warping (qDTW) Engine
======================================================
Implements batched pointwise quantum distance evaluation and dual-path dynamic
programming (NumPy fast path for inference, Python list of PyTorch scalar tensors
for differentiable autograd training).
"""

import numpy as np
import torch


def hybrid_multivariate_qdtw(
    seq1: torch.Tensor,
    seq2: torch.Tensor,
    weights: torch.Tensor,
    quantum_circuit,
    window: int = 15,
    device: torch.device = torch.device("cpu"),
):
    """
    Computes the Quantum Dynamic Time Warping (qDTW) alignment cost between two sequences.
    - Pre-batches all valid (i, j) pairs inside the Sakoe-Chiba band into a single PQC pass.
    - Uses a Dual-Path execution strategy:
        * Inference Mode (no_grad): detaches to NumPy arrays for rapid C/Python scalar min().
        * Training Mode (grad enabled): populates a 2D Python list of scalar PyTorch tensors
          to preserve the Autograd computation graph without inplace tensor errors.
    """
    N, _ = seq1.shape
    M, _ = seq2.shape

    idx_i, idx_j = [], []
    for i in range(1, N + 1):
        start_j = max(1, i - window)
        end_j = min(M + 1, i + window + 1)
        for j in range(start_j, end_j):
            idx_i.append(i - 1)
            idx_j.append(j - 1)

    batch_vec1 = seq1[idx_i]
    batch_vec2 = seq2[idx_j]

    q_vals = quantum_circuit(batch_vec1, batch_vec2, weights)
    distances = 1.0 - q_vals

    # FAST PATH: Evaluation / Inference Mode
    if not torch.is_grad_enabled():
        dist_np = distances.detach().cpu().numpy()
        dtw_matrix = np.full((N + 1, M + 1), np.inf)
        dtw_matrix[0, 0] = 0.0

        pair_index = 0
        for i in range(1, N + 1):
            start_j = max(1, i - window)
            end_j = min(M + 1, i + window + 1)
            for j in range(start_j, end_j):
                cost = dist_np[pair_index]
                pair_index += 1
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i - 1, j],
                    dtw_matrix[i, j - 1],
                    dtw_matrix[i - 1, j - 1],
                )
        return float(dtw_matrix[N, M])

    # AUTODIFF PATH: Training Mode (Autograd Graph Active)
    dtw_matrix = [
        [torch.tensor(float("inf"), device=device) for _ in range(M + 1)]
        for _ in range(N + 1)
    ]
    dtw_matrix[0][0] = torch.tensor(0.0, device=device)

    pair_index = 0
    for i in range(1, N + 1):
        start_j = max(1, i - window)
        end_j = min(M + 1, i + window + 1)
        for j in range(start_j, end_j):
            dist = distances[pair_index]
            pair_index += 1
            min_prev = torch.min(dtw_matrix[i - 1][j], dtw_matrix[i][j - 1])
            min_prev = torch.min(min_prev, dtw_matrix[i - 1][j - 1])
            dtw_matrix[i][j] = dist + min_prev

    return dtw_matrix[N][M]
