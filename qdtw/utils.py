"""
Evaluation, Caching, and Diagnostic Utilities
=============================================
Provides reproducibility seeding, quantum circuit resource inspection, symmetric/asymmetric
distance matrix caching, and k-Nearest Neighbors (k-NN) classification evaluators.
"""

import time
from collections import Counter
import numpy as np
import pennylane as qml
import torch


def set_seed(seed: int = 42):
    """Sets random seeds for PyTorch and NumPy to ensure reproducible runs."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def print_model_specs(model_name: str, is_quantum: bool = False, quantum_circuit=None, weights=None, device=None):
    """Prints gate depth, total gate count, and trainable parameter specifications."""
    print("\n" + "=" * 50)
    print(f"[{model_name} Specifications]")
    if not is_quantum:
        print("Architecture: Classical Baseline (0 quantum gates, 0 trainable weights)")
    else:
        n_q = weights.shape[1] if len(weights.shape) == 3 else 1
        dummy_x1 = torch.zeros((1, n_q), device=device)
        dummy_x2 = torch.zeros((1, n_q), device=device)
        specs = qml.specs(quantum_circuit, level="device")(dummy_x1, dummy_x2, weights)
        resources = specs["resources"]
        total_gates = getattr(resources, "num_gates", sum(resources.gate_types.values()))
        print(f"Quantum Circuit Depth : {resources.depth} gates")
        print(f"Total Gate Count      : {total_gates}")
        print(f"Trainable Parameters  : {weights.numel()}")
    print("=" * 50)


def precompute_distance_matrix(eval_data: torch.Tensor, train_data: torch.Tensor, distance_fn, time_est_steps: int = 3):
    """Precomputes the pairwise distance matrix between evaluation and training sequences."""
    N_eval, N_train = len(eval_data), len(train_data)
    dist_matrix = np.zeros((N_eval, N_train))
    start_time = time.time()
    est_remaining = time_est_steps
    is_symmetric = eval_data is train_data

    with torch.no_grad():
        if is_symmetric:
            for i in range(N_eval):
                for j in range(i + 1, N_train):
                    dist = distance_fn(eval_data[i], train_data[j])
                    dist_val = dist.item() if hasattr(dist, "item") else dist
                    dist_matrix[i, j] = dist_val
                    dist_matrix[j, i] = dist_val
        else:
            for i in range(N_eval):
                for j in range(N_train):
                    dist = distance_fn(eval_data[i], train_data[j])
                    dist_matrix[i, j] = dist.item() if hasattr(dist, "item") else dist

                    if est_remaining > 0:
                        est_remaining -= 1
                        done = time_est_steps - est_remaining
                        per_pair = (time.time() - start_time) / done
                        total_est = per_pair * N_train * N_eval
                        print(f"    [Step {done}] Pair time: {per_pair:.3f}s | Est. total matrix time: {total_est:.1f}s")

    print(f"    [Distance Matrix Precomputed in {time.time() - start_time:.2f}s]")
    return dist_matrix


def predict_knn(dist_row: np.ndarray, train_labels: torch.Tensor, k: int = 1, is_self_eval: bool = False):
    """Predicts the class label for a single instance using k-NN over a precomputed distance row."""
    distances = [
        (dist_row[j], train_labels[j].item() if hasattr(train_labels[j], "item") else train_labels[j])
        for j in range(len(dist_row))
    ]
    if is_self_eval:
        for idx, (d, _) in enumerate(distances):
            if d < 1e-6:
                distances.pop(idx)
                break

    distances.sort(key=lambda x: x[0])
    safe_k = min(k, len(distances))
    top_k_labels = [label for _, label in distances[:safe_k]]
    return max(set(top_k_labels), key=top_k_labels.count)


def evaluate_cached(split_name: str, dist_matrix: np.ndarray, eval_labels: torch.Tensor, train_labels: torch.Tensor, k_values: list, is_self_eval: bool = False):
    """Evaluates k-NN classification accuracy across multiple k values using a cached distance matrix."""
    N_eval = len(eval_labels)
    best_acc = 0.0
    for k in k_values:
        correct = 0
        for i in range(N_eval):
            target = eval_labels[i].item() if hasattr(eval_labels[i], "item") else eval_labels[i]
            pred = predict_knn(dist_matrix[i], train_labels, k=k, is_self_eval=is_self_eval)
            if pred == target:
                correct += 1
        acc = (correct / N_eval) * 100.0
        best_acc = max(best_acc, acc)
        print(f"   -> {split_name} (k={k:2d}) Accuracy: {acc:6.2f}%")
    print(f"   => {split_name} Peak k-NN Accuracy: {best_acc:6.2f}%")
    return best_acc


def print_trivial_baseline(y_train: torch.Tensor, y_val: torch.Tensor, y_test: torch.Tensor):
    """Computes and displays the Zero-R (majority class) baseline accuracy."""
    def calc_majority(labels):
        if labels is None or len(labels) == 0:
            return 0.0, None
        cleaned = [l.item() if hasattr(l, "item") else l for l in labels]
        maj_class, maj_count = Counter(cleaned).most_common(1)[0]
        return (maj_count / len(labels)) * 100.0, maj_class

    tr_acc, tr_c = calc_majority(y_train)
    va_acc, va_c = calc_majority(y_val)
    te_acc, te_c = calc_majority(y_test)

    print("\n" + "=" * 50)
    print("[Trivial Classifier Baseline (Zero-R Floor)]")
    print(f"  -> Train Majority Accuracy : {tr_acc:.2f}% (Class {tr_c})")
    if y_val is not None:
        print(f"  -> Val   Majority Accuracy : {va_acc:.2f}% (Class {va_c})")
    print(f"  -> Test  Majority Accuracy : {te_acc:.2f}% (Class {te_c})")
    print("=" * 50)
