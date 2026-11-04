"""
Quantum Metric Learning Optimization Engine
===========================================
Implements Triplet Margin Loss and Pairwise Contrastive Loss optimization over qDTW
matrices, including gradient norm clipping, L2 regularization, ReduceLROnPlateau
scheduling, early stopping, and barren plateau gradient variance diagnostics.
"""

import time
import numpy as np
import torch
import torch.optim as optim

from .dtw import hybrid_multivariate_qdtw


def train_quantum_metric(
    train_data: torch.Tensor,
    train_labels: torch.Tensor,
    quantum_circuit,
    initial_weights: torch.Tensor,
    val_data: torch.Tensor = None,
    val_labels: torch.Tensor = None,
    loss_type: str = "TRIPLET",
    window: int = 15,
    max_epochs: int = 30,
    margin: float = 0.5,
    patience: int = 5,
    samples_per_epoch: int = 1000,
    lr: float = 0.001,
    batch_size: int = 32,
    device: torch.device = torch.device("cpu"),
):
    """
    Unified training engine supporting both Triplet Margin Loss and Pairwise Contrastive Loss.
    Includes L2 weight decay (1e-4), gradient norm clipping (1.0), ReduceLROnPlateau scheduling,
    and gradient variance diagnostics for monitoring barren plateaus.
    """
    weights = initial_weights.clone().detach().requires_grad_(True)
    optimizer = optim.Adam([weights], lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)

    use_validation = val_data is not None and len(val_data) > 0
    best_val_loss = float("inf")
    best_weights = weights.clone().detach()
    epochs_no_improve = 0
    N_train = len(train_data)
    train_labels_np = np.array([l.item() if hasattr(l, "item") else l for l in train_labels])

    print(f"\n[Training Quantum Metric | Loss: {loss_type}]")
    print(f"Max Epochs: {max_epochs} | LR: {lr} | Dynamic Margin: {margin:.4f} | Batch Size: {batch_size}")

    start_train_time = time.time()

    for epoch in range(max_epochs):
        epoch_start = time.time()
        train_loss = 0.0
        epoch_grad_mags, epoch_grad_vars = [], []
        epoch_same_dists, epoch_diff_dists = [], []

        optimizer.zero_grad(set_to_none=True)

        if loss_type.upper() == "TRIPLET":
            epoch_samples = []
            for _ in range(samples_per_epoch):
                a_idx = np.random.randint(N_train)
                a_lbl = train_labels_np[a_idx]
                pos_idx = np.random.choice(np.where(train_labels_np == a_lbl)[0])
                neg_idx = np.random.choice(np.where(train_labels_np != a_lbl)[0])
                epoch_samples.append((a_idx, pos_idx, neg_idx))

            for idx, (a, p, n) in enumerate(epoch_samples):
                dist_pos = hybrid_multivariate_qdtw(train_data[a], train_data[p], weights, quantum_circuit, window, device)
                dist_neg = hybrid_multivariate_qdtw(train_data[a], train_data[n], weights, quantum_circuit, window, device)
                raw_loss = torch.relu(dist_pos - dist_neg + margin)

                epoch_same_dists.append(dist_pos.item())
                epoch_diff_dists.append(dist_neg.item())

                loss = raw_loss / batch_size
                loss.backward()

                if (idx + 1) % batch_size == 0 or (idx + 1) == len(epoch_samples):
                    if weights.grad is not None:
                        epoch_grad_mags.append(torch.norm(weights.grad).item())
                        epoch_grad_vars.append(torch.var(weights.grad).item() if weights.grad.numel() > 1 else 0.0)
                    torch.nn.utils.clip_grad_norm_([weights], max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                train_loss += raw_loss.item()

        elif loss_type.upper() == "PAIRWISE":
            epoch_samples = [tuple(np.random.choice(N_train, 2, replace=False)) for _ in range(samples_per_epoch)]
            for idx, (i, j) in enumerate(epoch_samples):
                dist = hybrid_multivariate_qdtw(train_data[i], train_data[j], weights, quantum_circuit, window, device)
                if train_labels_np[i] == train_labels_np[j]:
                    raw_loss = dist
                    epoch_same_dists.append(dist.item())
                else:
                    raw_loss = torch.relu(margin - dist)
                    epoch_diff_dists.append(dist.item())

                loss = raw_loss / batch_size
                loss.backward()

                if (idx + 1) % batch_size == 0 or (idx + 1) == len(epoch_samples):
                    if weights.grad is not None:
                        epoch_grad_mags.append(torch.norm(weights.grad).item())
                        epoch_grad_vars.append(torch.var(weights.grad).item() if weights.grad.numel() > 1 else 0.0)
                    torch.nn.utils.clip_grad_norm_([weights], max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                train_loss += raw_loss.item()
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}")

        avg_train_loss = train_loss / samples_per_epoch
        avg_grad_mag = np.mean(epoch_grad_mags) if epoch_grad_mags else 0.0
        avg_grad_var = np.mean(epoch_grad_vars) if epoch_grad_vars else 0.0
        avg_same = np.mean(epoch_same_dists) if epoch_same_dists else 0.0
        avg_diff = np.mean(epoch_diff_dists) if epoch_diff_dists else 0.0

        # Validation Phase
        if use_validation:
            val_loss = 0.0
            val_correct = 0
            with torch.no_grad():
                for i in range(len(val_data)):
                    v_label = val_labels[i].item()
                    val_sample_dists, pos_dists, neg_dists = [], [], []
                    for j in range(N_train):
                        dist = hybrid_multivariate_qdtw(val_data[i], train_data[j], weights, quantum_circuit, window, device)
                        t_label = train_labels_np[j]
                        val_sample_dists.append((dist, t_label))
                        if v_label == t_label:
                            pos_dists.append(dist)
                        else:
                            neg_dists.append(dist)

                        if loss_type.upper() == "PAIRWISE":
                            val_loss += dist if v_label == t_label else max(0.0, margin - dist)

                    val_sample_dists.sort(key=lambda x: x[0])
                    if val_sample_dists[0][1] == v_label:
                        val_correct += 1

                    if loss_type.upper() == "TRIPLET" and pos_dists and neg_dists:
                        val_loss += max(0.0, min(pos_dists) - min(neg_dists) + margin)

            avg_val_loss = val_loss / len(val_data) if loss_type.upper() == "TRIPLET" else val_loss / (len(val_data) * N_train)
            val_acc = (val_correct / len(val_data)) * 100.0
            scheduler.step(avg_val_loss)
            current_lr = optimizer.param_groups[0]["lr"]

            print(
                f"  Epoch {epoch + 1:2d}/{max_epochs} | LR: {current_lr:.5f} | "
                f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val 1-NN Acc: {val_acc:.1f}% | "
                f"SameDist: {avg_same:.3f} | DiffDist: {avg_diff:.3f} | "
                f"GradVar: {avg_grad_var:.6f} | Time: {time.time() - epoch_start:.1f}s"
            )

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_weights = weights.clone().detach()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f"  [Early Stopping] Triggered after {epoch + 1} epochs.")
                break
        else:
            print(
                f"  Epoch {epoch + 1:2d}/{max_epochs} | Train Loss: {avg_train_loss:.4f} | "
                f"SameDist: {avg_same:.3f} | DiffDist: {avg_diff:.3f} | "
                f"GradMag: {avg_grad_mag:.4f} | GradVar: {avg_grad_var:.6f} | Time: {time.time() - epoch_start:.1f}s"
            )
            best_weights = weights.clone().detach()

    print(f"[Training Complete in {time.time() - start_train_time:.2f}s]")
    return best_weights
