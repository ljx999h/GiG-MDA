# GiG-MDA — Implementation Code

> **GiG-MDA: Integrating Graph-in-Graph Embeddings with Multi-Modal Biological Features for Robust Drug-Disease Association Prediction**
>
> Wen Li, Quan Hu, Bo Liu, Tianen Mai, Wei Zhang, Xiaojie Zhang, Yanyan Yu
>
> *Manuscript under review — citation will be added upon publication.*

---

## Overview

This directory contains the complete implementation of the GiG-MDA framework.
The method predicts drug-disease associations (MDA) by combining three components:

| Component | Dimension | Description |
|-----------|-----------|-------------|
| **MiRAGE features** | 22D | 3 disease + 7 drug similarity scores, cross-multiplied variants, plus 2 topological counts |
| **GiGs embeddings** | 128D | Graph-in-Graph latent factors (64D drug + 64D disease), preserving full graph structure |
| **XGBoost classifier** | — | Gradient-boosted tree classifier with GPU acceleration, replacing the original Random Forest |

**Total feature vector:** 150 dimensions (22 MiRAGE + 128 GiGs).

---

## Datasets

Three benchmark datasets are used, located under `../data/`:

| Dataset | Path | Description |
|---------|------|-------------|
| **DDCD** | `../data/DDCD/` | Main dataset (~2.1M drug-disease pairs, 1:63 imbalance) |
| **F-Dataset** | `../data/F-Dataset/` | Benchmark F |
| **C-Dataset** | `../data/C-Dataset/` | Benchmark C |

---

## File Map

### Core Pipeline

| File | Purpose |
|------|---------|
| `DDCC_main.py` | Full training pipeline for DDCD dataset |
| `gigs_model.py` | GiGS (Graph-in-Graph) model definition |
| `MiRAGE _DDCD.ipynb` | Jupyter notebook: DDCD experiments |
| `MiRAGE_F.ipynb` | Jupyter notebook: F-Dataset experiments |
| `MiRAGE_C.ipynb` | Jupyter notebook: C-Dataset experiments |

### GiGS Pretraining

| File | Purpose |
|------|---------|
| `pretrain_gigs_C.py` | Pretrain GiGS embeddings on C-Dataset |
| `pretrain_gigs_F.py` | Pretrain GiGS embeddings on F-Dataset |

Pre-trained embeddings are saved in `model/`:

| File | Shape |
|------|-------|
| `model/gigs_dataDDCD.pkl` | Drug X: (1410,64), Disease Y: (1573,64) |
| `model/gigs_dataF.pkl` | Drug X, Disease Y |
| `model/gigs_dataC.pkl` | Drug X, Disease Y |

### Negative Sampling

| File | Purpose |
|------|---------|
| `negative_sampling_DDCD.py` | Random negative sampling for DDCD |
| `negative_sampling_C.py` | Random negative sampling for C-Dataset |
| `negative_sampling_F.py` | Random negative sampling for F-Dataset |

### Evaluation & Comparison

| File | Purpose |
|------|---------|
| `eval_sota_baselines.py` | Evaluate against SOTA baselines (DRHGCN, HINGRL, DRWBNCF, DDAGDL, AMDGT, PREDICT) |
| `eval_C.py` | Evaluate on C-Dataset |
| `eval_F.py` | Evaluate on F-Dataset |
| `eval_helpers_4variants.py` | Shared helpers for 4 evaluation variants |
| `clf_comparison.py` | Compare classifiers (XGBoost / RF / DT / KNN / LR) on DDCD with 5-Fold CV |
| `xgb_hyperparam_F.py` | XGBoost hyperparameter sweep on F-Dataset |
| `run_topk_xgb.py` | Top-k prediction with XGBoost |

### Ablation Study

| File | Purpose |
|------|---------|
| `ablation_study.py` | Full ablation study: 8 variants × 5-Fold CV, tests each feature group's contribution |

Ablation variants:
1. Full Model (MiRAGE) — 150D
2. w/o GiGs (No Graph) — 22D
3. GiGs Embedding Only — 128D
4. w/o Count Features
5. w/o Drug Features
6. w/o Disease Features
7. Count + GiGs Only
8. Similarity Only

### Contribution Analysis

| File | Purpose |
|------|---------|
| `contribution_demo.py` | Contribution analysis on DDCD |
| `contribution_demo_C.py` | Contribution analysis on C-Dataset |
| `contribution_demo_F.py` | Contribution analysis on F-Dataset |

### Visualization & Utilities

| File | Purpose |
|------|---------|
| `generate_figures.py` | Generate all paper figures (ROC, PR, calibration curves, feature importance) |
| `run_roc_pr_data.py` | Generate ROC/PR curve data |
| `gen_C_table.py` | Generate result tables for C-Dataset |
| `my_C_result.py` | C-Dataset result formatting |

### Output

Results are saved in `results/`:

| File | Description |
|------|-------------|
| `results/MiRAGE_score_DDCD.csv` | Full DDCD predictions |
| `results/MiRAGE_score_C.csv` | Full C-Dataset predictions |
| `results/MiRAGE_score_F.csv` | Full F-Dataset predictions |
| `results/final_predictions.csv` | Final prediction output |
| `results/feature_importance_final.png` | Feature importance plot |
| `results/confusion_matrix_optimized.png` | Confusion matrix |
| `results/ablation/` | Ablation study results (CSV + figures) |

---

## Quick Start

### Prerequisites

```bash
pip install numpy pandas scikit-learn xgboost matplotlib seaborn pikepdf pypdf reportlab
```

### 1. Pretrain GiGS Embeddings (if not already done)

```bash
python pretrain_gigs_F.py
python pretrain_gigs_C.py
```

Pre-trained `.pkl` files for DDCD are already in `model/`.

### 2. Run Full Training Pipeline

```bash
python DDCC_main.py
```

Or use Jupyter notebooks for interactive exploration:

```bash
jupyter notebook "MiRAGE _DDCD.ipynb"
```

### 3. Run Ablation Study

```bash
python ablation_study.py
```

Output: `results/ablation/ablation_summary.csv`, `ablation_all_folds.csv`, and comparison figures.

### 4. Compare Classifiers

```bash
python clf_comparison.py
```

Runs 5 classifiers (LR, KNN, DT, RF, XGBoost) × 5-Fold CV on DDCD.
Set `SAMPLE_SIZE` for quick runs or `SAMPLE_SIZE = None` for full dataset.

### 5. Evaluate Against SOTA Baselines

```bash
python eval_sota_baselines.py
```

### 6. Generate Paper Figures

```bash
python generate_figures.py
python run_roc_pr_data.py
```

---

## Key Results (DDCD Dataset)

### Classifier Comparison (150D features, 5-Fold CV, 200K sample)

| Model | AUROC | AUPR | F1 |
|-------|-------|------|-----|
| LogisticRegression | 0.9385 | 0.3577 | 0.2819 |
| KNN | 0.7702 | 0.2202 | 0.2786 |
| Decision Tree | 0.6361 | 0.0850 | 0.2711 |
| Random Forest | 0.9539 | 0.4050 | 0.2424 |
| **XGBoost** | **0.9543** | **0.4112** | **0.3133** |

### Ablation Study (XGBoost, 150D, 5-Fold CV)

| Variant | AUROC | AUPR | F1 |
|---------|-------|------|-----|
| Full Model (MiRAGE) | **0.9864** | 0.8162 | 0.7562 |
| Count + GiGs Only | 0.9827 | **0.8268** | **0.7898** |
| w/o Drug Features | 0.9847 | 0.8210 | 0.7729 |
| w/o Disease Features | 0.9858 | 0.8247 | 0.7696 |
| w/o GiGs (No Graph) | 0.9721 | 0.7211 | 0.6761 |
| w/o Count Features | 0.9735 | 0.5647 | 0.5496 |
| Similarity Only | 0.9614 | 0.5038 | 0.4997 |
| GiGs Embedding Only | 0.9368 | 0.3180 | 0.3672 |

---

## Configuration Notes

- `XGB_N_ESTIMATORS = 500`, `XGB_MAX_DEPTH = 10`, `XGB_LEARNING_RATE = 0.05`
- GPU auto-detection: `tree_method='hist', device='cuda'` (falls back to CPU)
- Full DDCD dataset: 2,142,939 rows, ~33,760 positive samples (1:63 imbalance)
- Use `SAMPLE_SIZE = 200000` for quick validation; `None` for full results
- Random seed fixed at 42 for reproducibility

---

## Key Findings

1. **XGBoost outperforms Random Forest** on AUPR and F1 under severe class imbalance (1:63).
2. **GiGs embeddings (128D) + MiRAGE (22D)** complement each other: GiGs provides graph-structural signal, MiRAGE provides domain-knowledge similarity scores.
3. **Count features** (`count_drug`, `count_disease`) are indispensable — removing them causes the largest AUPR drop (0.2515).
4. **Similarity scores alone are insufficient** — they achieve only 0.5038 AUPR, while adding GiGs + Count lifts it to 0.8268.

---

## Citation

This repository is the implementation of the GiG-MDA framework. The manuscript
is currently under review; a formal citation will be added once it is published.

For the related MiRAGE method on which the local feature extraction is based,
please cite:

```bibtex
@article{hassanali2024mirage,
  title={MiRAGE: mining relationships for advanced generative
         evaluation in drug repositioning},
  author={Hassanali Aragh, A. and Givehchian, P. and Moslemi Amirani, R.
          and Masumshah, R. and Eslahchi, C.},
  journal={Briefings in Bioinformatics},
  volume={25},
  number={4},
  pages={bbae337},
  year={2024}
}
```

---

## Contact

- Wen Li — wenli@hnu.edu.cn
- Quan Hu — hu3074487452@163.com
