"""
Quantum Circuit Architectures
=============================
Factory functions building the Parameterized Quantum Circuits (PQCs) for pointwise
quantum distance calculation, including the proposed Unified Pre-Embedding Adjoint Ansatz
and all ablation topologies.
"""

import math
import pennylane as qml
import torch


def build_circuit_pre_embedding_adjoint(n_qubits: int):
    """
    Proposed Architecture: Unified Pre-Embedding Adjoint Ansatz.
    Decouples trainable entanglement V(theta) from classical data encoding S(x),
    evaluating sequence overlap via global ground-state fidelity |<0|V^dagger S^dagger(y) S(x) V|0>|^2.
    """
    dev = qml.device("default.qubit", wires=n_qubits)

    def ansatz_embedding(x, weights):
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        for q in range(n_qubits):
            qml.RX(x[:, q], wires=q)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(vec1, vec2, weights):
        ansatz_embedding(vec1, weights)
        qml.adjoint(ansatz_embedding)(vec2, weights)
        zero_state = [0] * n_qubits
        return qml.expval(qml.Projector(zero_state, wires=range(n_qubits)))

    return circuit


def build_circuit_univariate_reuploading(n_qubits: int = 1):
    """
    Ablation I: Univariate Adjoint Data Re-uploading Circuit.
    Uses classical data as a multiplicative frequency scaling factor against trainable parameters
    bounded to [0, 1], terminating in a ground-state projector measurement.
    """
    dev = qml.device("default.qubit", wires=1)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(vec1, vec2, weights):
        num_layers = weights.shape[0]

        def reuploading_block(x):
            for layer in range(num_layers):
                qml.RY(x * weights[layer, 0], wires=0)
                qml.RZ(x * weights[layer, 1], wires=0)

        reuploading_block(vec1[:, 0])
        qml.adjoint(reuploading_block)(vec2[:, 0])
        return qml.expval(qml.Projector([0], wires=[0]))

    return circuit


def build_circuit_multi_adjoint_reuploading(n_qubits: int):
    """
    Ablation II: Multivariate Adjoint Re-uploading Circuit.
    Repeatedly interleaves trainable entanglement V(theta_l) and classical data S(x)
    across L layers before computing the global adjoint overlap.
    """
    dev = qml.device("default.qubit", wires=n_qubits)

    def state_preparation(x, weights):
        n_layers = weights.shape[0]
        for l in range(n_layers):
            qml.StronglyEntanglingLayers(weights[l : l + 1], wires=range(n_qubits))
            for q in range(n_qubits):
                qml.RX(x[:, q], wires=q)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(vec1, vec2, weights):
        state_preparation(vec1, weights)
        qml.adjoint(state_preparation)(vec2, weights)
        zero_state = [0] * n_qubits
        return qml.expval(qml.Projector(zero_state, wires=range(n_qubits)))

    return circuit


def build_circuit_local_observable(n_qubits: int):
    """
    Ablation III: Post-Embedding Local Observable (<Z_0>) Circuit.
    Loads both time steps sequentially via RX and RY rotations, applies post-embedding
    trainable entanglement, and extracts a scalar expectation value on Qubit 0.
    """
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(vec1, vec2, weights):
        for q in range(n_qubits):
            qml.RX(vec1[:, q], wires=q)
            qml.RY(vec2[:, q], wires=q)
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return qml.expval(qml.PauliZ(0))

    return circuit


def build_circuit_local_reuploading(n_qubits: int):
    """
    Supplementary Ablation: Post-Embedding Re-uploading with Local Observable (<Z_0>).
    """
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(vec1, vec2, weights):
        num_layers = weights.shape[0]
        for layer in range(num_layers):
            for q in range(n_qubits):
                qml.RX(vec1[:, q], wires=q)
                qml.RY(vec2[:, q], wires=q)
            qml.StronglyEntanglingLayers(weights[layer : layer + 1], wires=range(n_qubits))
        return qml.expval(qml.PauliZ(0))

    return circuit


def build_circuit_pre_embedding_no_entanglement(n_qubits: int):
    """
    Supplementary Ablation: Unentangled Pre-Embedding Adjoint Ansatz.
    Uses single-qubit Euler rotations without CNOT entanglement rings.
    """
    dev = qml.device("default.qubit", wires=n_qubits)

    def ansatz_embedding(x, weights):
        for layer_weights in weights:
            for q in range(n_qubits):
                qml.Rot(layer_weights[q, 0], layer_weights[q, 1], layer_weights[q, 2], wires=q)
        for q in range(n_qubits):
            qml.RX(x[:, q], wires=q)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(vec1, vec2, weights):
        ansatz_embedding(vec1, weights)
        qml.adjoint(ansatz_embedding)(vec2, weights)
        zero_state = [0] * n_qubits
        return qml.expval(qml.Projector(zero_state, wires=range(n_qubits)))

    return circuit


def get_quantum_circuit_and_weights(ansatz_type: str, n_qubits: int, n_layers: int, device: torch.device):
    """Factory router returning the requested QNode and properly initialized weight tensor."""
    if ansatz_type == "pre_embedding":
        circuit = build_circuit_pre_embedding_adjoint(n_qubits)
        weights = torch.empty((n_layers, n_qubits, 3), device=device).uniform_(-math.pi, math.pi)
    elif ansatz_type == "univariate_reupload":
        if n_qubits != 1:
            raise ValueError("univariate_reupload requires n_qubits == 1 (set --max_channels 1).")
        circuit = build_circuit_univariate_reuploading(1)
        # Multiplicative weights initialized in [0, 1] to prevent phase wrapping beyond pi
        weights = torch.rand((n_layers, 2), device=device)
    elif ansatz_type == "multi_adjoint_reupload":
        circuit = build_circuit_multi_adjoint_reuploading(n_qubits)
        weights = torch.empty((n_layers, n_qubits, 3), device=device).uniform_(-math.pi, math.pi)
    elif ansatz_type == "local_obs":
        circuit = build_circuit_local_observable(n_qubits)
        weights = torch.empty((n_layers, n_qubits, 3), device=device).uniform_(-math.pi, math.pi)
    elif ansatz_type == "local_reupload":
        circuit = build_circuit_local_reuploading(n_qubits)
        weights = torch.empty((n_layers, n_qubits, 3), device=device).uniform_(-math.pi, math.pi)
    elif ansatz_type == "no_entanglement":
        circuit = build_circuit_pre_embedding_no_entanglement(n_qubits)
        weights = torch.empty((n_layers, n_qubits, 3), device=device).uniform_(-math.pi, math.pi)
    else:
        raise ValueError(f"Unknown ansatz_type: {ansatz_type}")

    weights.requires_grad_(True)
    return circuit, weights
