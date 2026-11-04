# Quantum Dynamic Time Warping (qDTW) for Multivariate Time Series Classification

Source code for reproducibility of experiments described in the following paper:
> **"Quantum Dynamic Time Warping for Multivariate Time Series Classification"**  
> Diego Alvarez-Estevez, Alejandro Mayorga-Redondo, and Eduardo Mosqueira-Rey  
> *ICT Research Center (CITIC), Universidade da Coruña, Spain*

Related ArXiv.org pre-print: https://doi.org/10.48550/arXiv.2606.27815

---

## Repository Structure

The codebase is organized as follows:

```text
main.py                      # CLI entry point to run benchmarks and ablations
requirements.txt             # Python package dependencies
README.md                    # Documentation and usage instructions
qdtw/                        # Core modular package
├── __init__.py              # Package exports
├── classical.py             # Classical Euclidean and Sakoe-Chiba DTW baselines
├── quantum_circuits.py      # PQC factory (Pre-Embedding Adjoint, Re-uploading, Local <Z_0>)
├── dtw.py                   # Batched Dual-Path Hybrid Quantum DTW engine
├── data.py                  # Datasets loading and preprocessing
├── training.py              # Triplet Margin & Pairwise Contrastive metric learning loops
└── utils.py                 # Distance matrix caching, k-NN evaluation, and diagnostics
```

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/diegoalvareze/qDTW.git
   ```

2. Install the required dependencies (Python 3.10+ recommended):
   ```bash
   pip install -r requirements.txt
   ```

---

## Reproducing Paper Experiments

### 1. Universal Performance & Dimensional Scaling (Table 2)
Run the proposed **Unified Pre-Embedding Adjoint Ansatz** (`--ansatz pre_embedding`, $L=3$, Triplet Loss, $\beta=0.01$) across any number of spatial channels $C$:

```bash
# BasicMotions (Full C=6 channels)
python main.py --dataset BasicMotions --ansatz pre_embedding --layers 3 --max_channels 6

# RacketSports (Full C=6 channels)
python main.py --dataset RacketSports --ansatz pre_embedding --layers 3 --max_channels 6

# FingerMovements (Truncated to C=8 channels for state-vector simulation)
python main.py --dataset FingerMovements --ansatz pre_embedding --layers 3 --max_channels 8

# Dimensional scaling sweep (e.g., testing C=1..6 on BasicMotions)
for c in 1 2 3 4 5 6; do
    python main.py --dataset BasicMotions --ansatz pre_embedding --layers 3 --max_channels $c
done
```

### 2. Ablation I: Univariate Data Re-uploading (Table 3)
Evaluate the multiplicative **Univariate Adjoint Re-uploading** circuit on single-channel ($C=1$) sequences:

```bash
python main.py --dataset FingerMovements --ansatz univariate_reupload --layers 3 --max_channels 1
```

### 3. Ablation II: Multivariate Data Re-uploading (Table 4)
Evaluate the **Multivariate Adjoint Re-uploading** architecture ($L=3$) on multi-qubit registers:

```bash
python main.py --dataset FingerMovements --ansatz multi_adjoint_reupload --layers 3 --max_channels 8
```

### 4. Ablation III: Post-Embedding Local Observable $\langle Z_0 \rangle$ (Table 5)
Contrast the Global Adjoint fidelity measurement against quantum pooling on qubit 0 (`local_obs`):

```bash
python main.py --dataset BasicMotions --ansatz local_obs --layers 3 --max_channels 6
```

### 5. Circuit Depth ($L \in \{1, 3, 5\}$) and Loss Topology ($\beta \in \{0.01, 0.1\}$) Ablations (Appendices A & B)
```bash
# Circuit depth ablation (Appendix A)
python main.py --dataset BasicMotions --ansatz pre_embedding --layers 5 --max_channels 6

# Pairwise Contrastive Loss vs. Triplet Margin Loss (Appendix B)
python main.py --dataset FingerMovements --ansatz pre_embedding --layers 3 --max_channels 8 --loss PAIRWISE --base_margin 0.01
```

---

## Datasets & Access

* **UEA Multivariate Time Series Classification Archive (`BasicMotions`, `RacketSports`, `FingerMovements`):** Automatically downloaded and cached via `tslearn.datasets.UCR_UEA_datasets` on first execution (also available at [timeseriesclassification.com](https://www.timeseriesclassification.com)).
* **Custom`Apnea` clinical subset:** Derived from the publicly accessible **Sleep Heart Health Study (SHHS)** repository hosted by the National Sleep Research Resource ([https://sleepdata.org/datasets/shhs](https://sleepdata.org/datasets/shhs), DOI: `10.25822/ghy8-ks59`). Researchers can request access to the raw polysomnography recordings directly through NSRR upon signing the corresponding Data Access and Use Agreement (DAUA). Once preprocessed into `.npy` format, run via:
  ```bash
  python main.py --dataset CUSTOM --custom_instances path/to/instances.npy --custom_labels path/to/labels.npy
  ```

---

## Acknowledgments

We appreciate the efforts of the UEA team in creating, cleaning, and curating three of the datasets we used.

The Sleep Heart Health Study (SHHS) was supported by National Heart, Lung, and Blood Institute cooperative agreements U01HL53916 (University of California, Davis), U01HL53931 (New York University), U01HL53934 (University of Minnesota), U01HL53937 and U01HL64360 (Johns Hopkins University), U01HL53938 (University of Arizona), U01HL53940 (University of Washington), U01HL53941 (Boston University), and U01HL63463 (Case Western Reserve University). The National Sleep Research Resource was supported by the National Heart, Lung, and Blood Institute (R24 HL114473, 75N92019R002).

This study has been supported by project RYC2022-038121-I, funded by MCIN/AEI/10.13039/501100011033 and European Social Fund Plus (ESF+), project PID2023-147422OB-I00 funded by MCIU/AEI/10.13039/501100011033 and the European FEDER program, and by project ED431F 2025/35 funded by Xunta de Galicia.