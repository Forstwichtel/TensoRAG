# TensoRAG: Multilinear Generalized Singular Value Decomposition (ML-GSVD)

[![Interactive Demo](https://img.shields.io/badge/Streamlit-Interactive_Demo-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://tensorag.streamlit.app/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)

A high-performance Python/NumPy implementation of the Multilinear Generalized Singular Value Decomposition (ML-GSVD), based on the pioneering mathematical research of Dr. Liana Khamidullina and Prof. Martin Haardt (Technische Universität Ilmenau).

⚠️ **Disclaimer:** This is an independent, private hobby project developed by @Forstwichtel. It is not officially affiliated with, endorsed by, or in any way connected to the authors or the Technische Universität Ilmenau. This repository serves solely as an independent implementation of their published academic findings.

🌐 *Read this documentation in [German / Deutsch](README_DE.md).*

---

## 👥 Authors & Contributors

*   **Main Author:** Forstwichtel [forstwichtel@gmail.com]
*   **Co-Author:** Helferlein [bot]

---

## 📌 Introduction & Mathematical Background

**TensoRAG** implements the **ML-GSVD**, a major mathematical breakthrough that extends the classical 2-matrix Generalized SVD (GSVD) to an arbitrary number of $K \ge 2$ matrices (viewed as slices of a 3D tensor $\mathcal{H}$) sharing a common dimension. 

Unlike prior attempts (such as the HO-GSVD), the ML-GSVD strictly preserves the core properties of the GSVD, specifically the **exact column-orthogonality** of the left factor matrices $\mathbf{B}_k$.

Mathematically, it decomposes a set of $K$ matrices $\mathbf{H}_k \in \mathbb{C}^{J_k \times I}$ simultaneously into:
$$\mathbf{H}_k \approx \mathbf{B}_k \cdot \mathbf{C}_k \cdot \mathbf{A}^H \quad \text{for } k = 1, \dots, K$$

*   **$\mathbf{A}^H \in \mathbb{C}^{Q \times I}$**: The **Shared Representation Basis** (the "core matrix" representing the global common subspace).
*   **$\mathbf{B}_k \in \mathbb{C}^{J_k \times Q}$**: The **Left Orthogonal Factors** ($\mathbf{B}_k^H \mathbf{B}_k = \mathbf{I}$), keeping the geometric structures intact.
*   **$\mathbf{C}_k \in \mathbb{R}^{Q \times Q}$**: The **Coefficient Matrices** (real, non-negative diagonal matrices representing the singular values for each specific slice).

---

## 🚀 Key Features

*   **Simultaneous Multi-Matrix Factorization:** Jointly decomposes $K \ge 2$ matrices with different row dimensions but a shared column dimension.
*   **True Orthogonal Procrustes Solver:** Solves the orthogonal factor updates via stable polar decomposition (using SVD) to guarantee strict column-orthogonality down to machine precision ($\sim 10^{-15}$ / standard float64 limits).
*   **Alternating Least Squares (ALS):** Implements the robust *Direct Fitting* iterative optimization algorithm.
*   **Complex & Real Support:** Fully compatible with both real-valued data (e.g., neural network weights) and complex-valued data (e.g., wireless signal processing / MIMO channel matrices).
*   **Low-Rank Compression:** Optimal for reducing parameter footprint in deep learning (e.g., compressing Attention layers) and streamlining large-scale Vector Search / RAG databases.
*   **Interactive AI Agent Simulation:** Features a live, step-by-step simulation demonstrating how an autonomous AI Agent uses the compressed ML-GSVD vector database as a retrieval tool to answer queries with minimum memory overhead.


---

## 📂 Repository Structure

```text
├── LICENSE                 # Apache-2.0 License Text
├── NOTICE                  # Copyright & academic attribution statements
├── README.md               # Project documentation and guide (this file)
├── tensorag.py             # Single-file production-ready implementation of ML-GSVD
├── tensorag_demo.py        # Complete RAG simulation demonstrating vector compression
└── tensorag_benchmark.py   # Speed & latency comparison benchmark script
```

---

## 💻 Quick Start & Usage

This library is packaged as a single, lightweight Python module with no external dependencies other than **NumPy**.

```python
import numpy as np
from tensorag import MultilinearGSVD

# 1. Generate synthetic data (e.g., 4 matrix slices sharing a column dimension of 100)
K, I = 4, 100
row_dimensions = [64, 48, 80, 120]  # Rows can vary per slice!
H_list = [np.random.randn(dim, I) for dim in row_dimensions]

# 2. Instantiate and run ML-GSVD to compress to a target subspace rank of Q = 16
Q = 16
gsvd = MultilinearGSVD(target_rank=Q, max_iter=50, tol=1e-6)
gsvd.fit(H_list)

# 3. Access the factors
print("Shared Basis A shape:", gsvd.A.shape)  # Expected: (100, 16)
for k in range(K):
    print(f"Slice {k} - Orthogonal factor B_k shape:", gsvd.B[k].shape)  # e.g., (64, 16)
    print(f"Slice {k} - Coefficients C_k diagonal:", gsvd.C[k, :])

# 4. Reconstruct and measure error
H_0_rec = gsvd.reconstruct(0)
reconstruction_error = np.linalg.norm(H_list[0] - H_0_rec)
print(f"Slice 0 Reconstruction Error: {reconstruction_error:.4f}")
```

---

## 📜 Academic Attribution & Citation

If you use this code or algorithm in your research or commercial applications, please cite the underlying academic publications that made this work possible:

### Primary Thesis
> **Khamidullina, Liana (2024).**  
> *Tensor decompositions and algorithms for efficient multidimensional signal processing.*  
> Dissertation, Technische Universität Ilmenau.  
> URN: [urn:nbn:de:gbv:ilm1-2024000104](https://nbn-resolving.org/urn:nbn:de:gbv:ilm1-2024000104)  
> DOI: [10.22032/dbt.59389](https://doi.org/10.22032/dbt.59389)

### Key Journal Publication
> **L. Khamidullina, A. L. F. de Almeida, and M. Haardt,**  
> "Multilinear Generalized Singular Value Decomposition (ML-GSVD) and Its Application to Multiuser MIMO Systems,"  \
> *IEEE Transactions on Signal Processing*, vol. 70, pp. 2783-2797, 2022.  
> DOI: [10.1109/TSP.2022.3178902](https://doi.org/10.1109/TSP.2022.3178902)

---

## ⚖️ License & GDPR Compliance

### License
This project is licensed under the **Apache License, Version 2.0**. You are free to copy, modify, distribute, and perform the work, even for commercial purposes, under the terms of the license. See the [LICENSE](LICENSE) file for more details.

### GDPR / Privacy
*   **Zero Telemetry:** This code is entirely offline and air-gapped. It does not collect, store, track, or transmit any user metrics, system data, or personal identifiers.
*   **Privacy-First:** Ensure that any datasets or embeddings you process with this library are fully anonymized. The authors of this repository do not have access to any data you run through this algorithm.
