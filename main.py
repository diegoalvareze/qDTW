#!/usr/bin/env python3
"""
Main CLI Entry Point for Quantum Dynamic Time Warping (qDTW) Experiments
========================================================================
Executes classical baselines, untrained quantum kernels, and Triplet/Pairwise
trained qDTW models across univariate and multivariate time series benchmarks.
"""

import argparse
import pennylane as qml
import torch

from qdtw import (
    check_quantum_scale,
    evaluate_cached,
    get_quantum_circuit_and_weights,
    hybrid_multivariate_qdtw,
    load_public_dataset,
    multivariate_classical_dtw,
    multivariate_euclidean,
    precompute_distance_matrix,
    prepare_custom_dataset,
    print_model_specs,
    print_trivial_baseline,
    set_seed,
    train_quantum_metric,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Quantum Dynamic Time Warping (qDTW) for Multivariate Time Series Classification"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="BasicMotions",
        help="Dataset name: 'BasicMotions', 'RacketSports', 'FingerMovements', or 'CUSTOM' (for SHHS Apnea).",
    )
    parser.add_argument(
        "--custom_instances",
        type=str,
        default="testdata/oficial_large_dataset_v2/instances_large_dataset_v2.npy",
        help="Path to custom instances .npy file when --dataset CUSTOM is used.",
    )
    parser.add_argument(
        "--custom_labels",
        type=str,
        default="testdata/oficial_large_dataset_v2/labels_large_dataset_v2.npy",
        help="Path to custom labels .npy file when --dataset CUSTOM is used.",
    )
    parser.add_argument(
        "--ansatz",
        type=str,
        default="pre_embedding",
        choices=[
            "pre_embedding",
            "univariate_reupload",
            "multi_adjoint_reupload",
            "local_obs",
            "local_reupload",
            "no_entanglement",
        ],
        help="Quantum circuit architecture to evaluate (default: 'pre_embedding').",
    )
    parser.add_argument("--layers", type=int, default=3, help="Number of circuit layers L (default: 3).")
    parser.add_argument(
        "--max_channels",
        type=int,
        default=None,
        help="Maximum number of channels C to include (e.g. 1..8 for dimensional scaling).",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=None,
        help="Optional maximum sequence length T (resamples via FFT if shorter than native length).",
    )
    parser.add_argument("--window", type=int, default=15, help="Sakoe-Chiba warping window W (default: 15).")
    parser.add_argument(
        "--loss",
        type=str,
        default="TRIPLET",
        choices=["TRIPLET", "PAIRWISE"],
        help="Metric learning objective function (default: TRIPLET).",
    )
    parser.add_argument(
        "--base_margin",
        type=float,
        default=0.01,
        help="Base margin scalar beta; dynamic margin alpha = beta * T (default: 0.01).",
    )
    parser.add_argument("--epochs", type=int, default=30, help="Maximum training epochs (default: 30).")
    parser.add_argument("--batch_size", type=int, default=32, help="Mini-batch size (default: 32).")
    parser.add_argument("--lr", type=float, default=0.001, help="Initial learning rate (default: 0.001).")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (default: 5).")
    parser.add_argument(
        "--samples_per_epoch",
        type=int,
        default=1000,
        help="Number of triplets/pairs sampled per training epoch (default: 1000).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42).")
    parser.add_argument("--skip_training", action="store_true", help="Evaluate only classical and untrained qDTW baselines.")
    parser.add_argument("--draw_circuit", action="store_true", help="Print ASCII diagram of the quantum circuit.")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    # Native CPU execution avoids PCIe transfer overhead for small statevector simulations
    device = torch.device("cpu")
    k_values = [1, 3, 5, 7, 11]

    print(f"Initializing qDTW Benchmark on device: {device}")
    print(f"Configuration -> Dataset: {args.dataset} | Ansatz: {args.ansatz} | Layers (L): {args.layers} | Loss: {args.loss}")

    if args.dataset.upper() == "CUSTOM":
        channels_arg = args.max_channels if args.max_channels is not None else "all"
        X_train, Y_train, X_val, Y_val, X_test, Y_test, n_qubits = prepare_custom_dataset(
            instances_path=args.custom_instances,
            labels_path=args.custom_labels,
            channels=channels_arg,
            target_length=args.max_length,
            max_instances=800,
            val_proportion=0.10,
            test_proportion=0.20,
            device=device,
        )
    else:
        max_ch = args.max_channels
        if args.dataset == "FingerMovements" and max_ch is None:
            max_ch = 8
        X_train, Y_train, X_val, Y_val, X_test, Y_test, n_qubits = load_public_dataset(
            dataset_name=args.dataset,
            target_length=args.max_length,
            val_proportion=0.10,
            test_proportion=0.20,
            max_instances=800,
            max_channels=max_ch,
            device=device,
        )

    _, seq_len, _ = X_train.shape
    print(f"\n[Dataset Summary] Train: {len(X_train)} | Val: {len(X_val) if X_val is not None else 0} | Test: {len(X_test)}")
    print(f"[Sequence Dimensions] Length (T): {seq_len} | Channels/Qubits (C): {n_qubits} | Sakoe-Chiba Window (W): {args.window}")

    check_quantum_scale(X_train, "X_train")
    check_quantum_scale(X_val, "X_val")

    # 0. Zero-R Trivial Baseline
    print_trivial_baseline(Y_train, Y_val, Y_test)

    # 1. Classical Euclidean Baseline
    print_model_specs("Classical Euclidean k-NN", is_quantum=False)
    euc_matrix = precompute_distance_matrix(X_test, X_train, multivariate_euclidean)
    evaluate_cached("Test (Classical Euclidean)", euc_matrix, Y_test, Y_train, k_values)

    # 2. Classical DTW Baseline
    print_model_specs("Classical DTW k-NN", is_quantum=False)
    cdtw_fn = lambda a, b: multivariate_classical_dtw(a, b, window=args.window)
    cdtw_matrix = precompute_distance_matrix(X_test, X_train, cdtw_fn)
    evaluate_cached("Test (Classical DTW)", cdtw_matrix, Y_test, Y_train, k_values)

    # 3. Build Quantum Circuit & Untrained qDTW Baseline
    quantum_circuit, random_weights = get_quantum_circuit_and_weights(
        ansatz_type=args.ansatz, n_qubits=n_qubits, n_layers=args.layers, device=device
    )

    if args.draw_circuit:
        print("\n" + "=" * 50)
        print("QUANTUM CIRCUIT ARCHITECTURE")
        print("=" * 50)
        print(qml.draw(quantum_circuit, level=2)(X_train[0], X_train[1], random_weights))
        print("=" * 50)

    print_model_specs(f"Untrained qDTW ({args.ansatz})", is_quantum=True, quantum_circuit=quantum_circuit, weights=random_weights, device=device)
    untrained_fn = lambda a, b: hybrid_multivariate_qdtw(a, b, random_weights, quantum_circuit, window=args.window, device=device)
    untrained_matrix = precompute_distance_matrix(X_test, X_train, untrained_fn)
    evaluate_cached("Test (Untrained qDTW)", untrained_matrix, Y_test, Y_train, k_values)

    # 4. Trained qDTW Evaluation
    if not args.skip_training:
        dynamic_margin = args.base_margin * seq_len
        trained_weights = train_quantum_metric(
            train_data=X_train,
            train_labels=Y_train,
            quantum_circuit=quantum_circuit,
            initial_weights=random_weights,
            val_data=X_val,
            val_labels=Y_val,
            loss_type=args.loss,
            window=args.window,
            max_epochs=args.epochs,
            margin=dynamic_margin,
            patience=args.patience,
            samples_per_epoch=args.samples_per_epoch,
            lr=args.lr,
            batch_size=args.batch_size,
            device=device,
        )

        print("\n[Final Evaluation of Trained qDTW]")
        trained_fn = lambda a, b: hybrid_multivariate_qdtw(a, b, trained_weights, quantum_circuit, window=args.window, device=device)
        trained_matrix = precompute_distance_matrix(X_test, X_train, trained_fn)
        evaluate_cached("Test (Trained qDTW)", trained_matrix, Y_test, Y_train, k_values)


if __name__ == "__main__":
    main()
