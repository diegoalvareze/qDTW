"""
Quantum Dynamic Time Warping (qDTW) Package
===========================================
Modular PyTorch and PennyLane implementation for multivariate time series classification.
"""

from .classical import multivariate_classical_dtw, multivariate_euclidean
from .quantum_circuits import get_quantum_circuit_and_weights
from .dtw import hybrid_multivariate_qdtw
from .data import load_public_dataset, prepare_custom_dataset, check_quantum_scale
from .training import train_quantum_metric
from .utils import (
    set_seed,
    print_model_specs,
    print_trivial_baseline,
    precompute_distance_matrix,
    predict_knn,
    evaluate_cached,
)

__all__ = [
    "multivariate_euclidean",
    "multivariate_classical_dtw",
    "get_quantum_circuit_and_weights",
    "hybrid_multivariate_qdtw",
    "load_public_dataset",
    "prepare_custom_dataset",
    "check_quantum_scale",
    "train_quantum_metric",
    "set_seed",
    "print_model_specs",
    "print_trivial_baseline",
    "precompute_distance_matrix",
    "predict_knn",
    "evaluate_cached",
]
