# GiG-MDA

**GiG-MDA: Integrating Graph-in-Graph Embeddings with Multi-Modal Biological Features for Robust Drug-Disease Association Prediction**

Wen Li, Quan Hu, Bo Liu, Tianen Mai, Wei Zhang, Xiaojie Zhang, Yanyan Yu

## Overview

GiG-MDA is a dual-channel framework for drug-disease association (DDA) prediction. It combines:

| Component | Dimension | Description |
|-----------|-----------|-------------|
| **MiRAGE features** | 22D / 18D | Local neighborhood similarity scores (3 or 1 disease + 7 drug modalities) |
| **GiGs embeddings** | 128D | Graph-in-Graph latent factors (64D drug + 64D disease) via GRMF |
| **XGBoost classifier** | — | Gradient-boosted trees with 5-fold CV and anti-leakage safeguards |

The final feature vector is **150-D** (DDCD: 22 + 128) or **146-D** (C/F: 18 + 128).

## Repository Structure

```
├── code/                 # All implementation scripts
│   ├── *.py              # Pipeline, evaluation, ablation, figures
│   ├── model/*.pkl       # Pre-trained GiGs embeddings (drug & disease latent vectors)
│   ├── *.ipynb           # Per-dataset feature generation notebooks
│   └── README.md         # Detailed code documentation
├── data/                 # Benchmark datasets
│   ├── DDCD/             # Main dataset (1,410 drugs × 1,573 diseases)
│   ├── C-Dataset/        # 663 drugs × 409 diseases
│   └── F-Dataset/        # 592 drugs × 313 diseases
├── download_data.py      # Restore large DDCD files from Zenodo
├── requirements.txt
└── .gitignore
```

## Datasets

Three benchmark datasets are used. The C- and F-Datasets are included in full.

**DDCD** — The two largest files (`data/DDCD/Evaluation/train.csv`, `test.csv`) exceed
GitHub's 100 MB per-file limit and are hosted on Zenodo
([DOI: 10.5281/zenodo.21883418](https://doi.org/10.5281/zenodo.21883418)). To reproduce
DDCD experiments, run:

```bash
python download_data.py
```

This restores the files to `data/DDCD/Evaluation/`. Verify with `python download_data.py --check`.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (DDCD only) Restore large data files
python download_data.py

# 3. Reproduce main results
python code/DDCC_main.py            # DDCD 5-fold CV + test evaluation (RF baseline)
python code/ablation_study.py       # DDCD ablation study (XGBoost)
python code/clf_comparison.py       # DDCD classifier comparison (XGBoost vs others)
python code/eval_C.py               # C-Dataset evaluation
python code/eval_F.py               # F-Dataset evaluation
```

> **Note:** The paper's XGBoost 5-fold CV results are produced by `ablation_study.py`
> and `clf_comparison.py`. `DDCC_main.py` reproduces the Random Forest baseline.

## Reproducing the Paper's Results

| Experiment | Script | Dataset |
|-----------|--------|---------|
| State-of-the-art comparison | `eval_sota_baselines.py` | DDCD |
| Ablation study (8 variants) | `ablation_study.py` | DDCD |
| Classifier comparison | `clf_comparison.py` | DDCD |
| XGBoost hyperparameter sweep | `xgb_hyperparam_F.py` | F-Dataset |
| ROC/PR curves | `run_roc_pr_data.py` | F, DDCD |
| Feature importance | `run_topk_xgb.py` | DDCD |

## License

TBD — add your license before publishing.

## Contact

Quan Hu — hu3074487452@163.com
