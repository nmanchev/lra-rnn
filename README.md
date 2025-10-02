# Can Local Representation Alignment RNNs Solve Temporal Tasks?  

This repository provides the implementation for the paper **“Can Local Representation Alignment RNNs Solve Temporal Tasks?”**

---

## 📄 Paper

- **Title**: Can Local Representation Alignment RNNs Solve Temporal Tasks?
- **Authors**: Nikolay Manchev, Luis C. Garcia-Peraza-Herrera
- **Abstract (short)**: Proposes and evaluates a local, target-propagation-based training method (LRA) for RNNs as an alternative to BPTT, explores gradient instabilities and introduces a gradient regularization to improve performance on temporal tasks.

---

## 🛠 Repository Structure

- `lra2_reg.py` — main implementation of the regularized LRA RNN  
- `permutation.py`, `tempOrder.py`, `tempOrder3bit.py`, `addition.py` — task scripts (random permutation, temporal order, 3-bit temporal order)  
- `gridsearch.py` — hyperparameter search framework  

---

## ▶️ Quick Start

1. Clone this repository:
   ```bash
   git clone https://github.com/nmanchev/lra-rnn.git
   cd lra-rnn```
   
Install dependencies (e.g. via `conda env create -f environment.yml`).

Use `gridsearch.py` to sweep over hyperparameters for best results.