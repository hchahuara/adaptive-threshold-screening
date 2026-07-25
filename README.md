# Adaptive Thresholding for Safe Screening in Regularized Optimization

## Overview

This repository contains the reference implementation developed during the research presented in the master's thesis **"Enhancing Safe Screening Rules Using Adaptive Thresholding for Regularized Optimization Problems"** by **Hector Francisco Chahuara Silva (2025)**.

The library implements optimization routines for sparse regularized learning problems based on **Accelerated Proximal Gradient (APG/FISTA)** together with feature screening strategies. Its main research contribution is the incorporation of an **adaptive thresholding scheme** into safe screening procedures in order to improve the identification of inactive variables during optimization.

The repository also includes the experimental notebooks used to reproduce the computational studies reported in the thesis.

This software is released for **research, demonstration, and reproducibility purposes**.

---

## Implemented Methodology

The implemented optimization framework considers optimization problems of the general form

\[
\min_x f(x) + \lambda g(x),
\]

where

- \(f(\cdot)\) is a differentiable loss function,
- \(g(\cdot)\) is a sparsity-promoting regularizer.

The implementation combines:

- Accelerated Proximal Gradient (APG/FISTA) optimization,
- proximal operators for sparse regularization,
- dynamic support refinement,
- screening strategies for identifying inactive variables,
- an adaptive thresholding procedure that modifies the screening stage.

From the available implementation, the optimizer periodically evaluates screening information during the optimization process and refines the active support before continuing the iterations.

The repository supports several loss functions and regularization models that are used throughout the included experiments.

This description intentionally remains at a high level. Algorithmic details, theoretical guarantees, and mathematical derivations are presented in the accompanying publications rather than duplicated here.

---

## Repository Organization

```text
.
├── source/
│   Core implementation of the optimization framework.
│
│   ├── model.py
│   │     Mathematical models, loss functions,
│   │     regularizers, proximal operators, gradients,
│   │     and related routines.
│   │
│   ├── optimizer.py
│   │     Accelerated Proximal Gradient optimizer.
│   │
│   ├── screening.py
│   │     Screening strategies and support refinement.
│   │
│   ├── thresholding.py
│   │     Adaptive thresholding methods used during screening.
│   │
│   └── util.py
│         Utility functions.
│
├── examples/
│   Jupyter notebooks reproducing the experiments.
│
├── datasets/
│   Included links to externally hosted datasets.
│
└── results/
    Default location for generated figures and experiment outputs.
```

---

## Getting Started

The primary interface to the implementation is through the Jupyter notebooks located in the `examples` directory.

The notebooks demonstrate the optimization framework in several application domains, including:

- sparse reconstruction,
- signal denoising,
- image classification,
- biomedical imaging,
- inverse problems.

Some notebooks automatically download the required datasets, while others expect them to be available locally. See `datasets/README.md` for dataset availability and download instructions.

---

## Experiments

The notebooks included in this repository correspond to the experiments reported in the **master's thesis**.

The associated conference publication presents a reduced subset of these experiments. The implementation differences between the thesis experiments and those reported in the conference paper are minimal and mainly involve changing a small number of experimental coefficients (approximately five parameter values). Consequently, this repository follows the more complete experimental setup presented in the thesis.

---

## Intended Use

This repository is intended for

- research reproducibility,
- academic reference,
- experimentation with screening methods for sparse optimization,
- educational purposes.

The code was developed as research software and should not be considered a production-ready optimization library.

---

## Implementation Notes

- The implementation is organized around modular optimization components, separating mathematical models, optimization routines, screening procedures, thresholding methods, and utility functions.
- The primary user interface consists of the Jupyter notebooks located in the `examples` directory.
- The repository has been published to accompany the research work and facilitate reproducibility of the reported experiments.
- Only datasets that can be redistributed are included in this repository. Additional datasets must be obtained from their original sources (see `datasets/README.md`).

---

## Limitations

The current repository has several limitations.

- It is distributed primarily as research code rather than as a packaged Python library.
- Installation scripts, dependency management, package configuration, and automated testing are not included.
- Some experimental datasets are not distributed with the repository because of their size or licensing restrictions.
- Reproducing every experiment may therefore require obtaining the corresponding datasets separately.
- No claims are made regarding cross-platform compatibility beyond the environments used during the original research.

---

## References

### Master's Thesis

**Hector Francisco Chahuara Silva.**

*Enhancing Safe Screening Rules Using Adaptive Thresholding for Regularized Optimization Problems.*

Master's Thesis, Pontificia Universidad Católica del Perú (PUCP), 2025.

Repository:
https://tesis.pucp.edu.pe/items/0f10c09d-0614-4ddb-9281-dabfcf00e8b8

---

### Conference Paper

**Hector Chahuara and Paul Rodriguez.**

*Enhancing Safe Screening Rules with Adaptive Thresholding for Non-Overlapping Group Sparse Norm Regularized Problems.*

Proceedings of the **2023 24th International Conference on Digital Signal Processing (DSP)**, IEEE, 2023.

DOI:
https://doi.org/10.1109/DSP58604.2023.10167966

---

## License

No license has currently been specified for this repository.

If you intend others to reuse or build upon this work, consider adding an appropriate open-source license.
