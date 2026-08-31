# GiG-MDA: Structure- and Topology-Driven Features for Cold-Start Drug–Disease Association Prediction under a Leakage-Aware Protocol

This code package reproduces all experiments in the manuscript (submitted to *Pharmaceuticals*). The pipeline implements a leakage-aware evaluation protocol (pair-disjoint splits; all features, embeddings, and negatives constructed fold-locally) and the cold-start (cold-drug) evaluation of molecular structure features (MoLFormer / ECFP32), graph-regularized embeddings (GRMF), and a lightweight GCN.

## 1. Environment

```bash
pip install -r requirements.txt
```

- Python 3.9+ (Anaconda recommended)
- If no GPU is available, set `device='cpu'` in `XGB_CONFIG` in `code/r2_config.py`.

## 2. Data

The three benchmark datasets (C-Dataset, F-Dataset, DDCD) are the standard drug–disease association benchmarks:

- **C-Dataset / F-Dataset**: 663/591 drugs, 409/313 diseases, 2,532/1,932 verified associations. Drug features from DrugBank; disease features from CTD.
- **DDCD**: 1,410 drugs, 1,573 diseases, 42,200 associations, curated from CTD.

Download links, directory layout, and the pre-computed feature files (including MoLFormer embeddings, GIPK similarity matrices, split manifests, and per-seed results) are described in `data/README.md`. The largest DDCD files are hosted on Zenodo (DOI: 10.5281/zenodo.21883418); restore them with the download script in the repository.

## 3. Reproducing the Experiments

All random seeds: **42, 7, 123, 2024, 999** (regular evaluation) and **42, 7, 123, 2024** (cold-start evaluation). Run all commands from the package root.

### 3.1 Data split and audit (Section 3.1)

```bash
python code/data_split.py --dataset C --random-state 42
python code/audit_split.py --dataset C
```

### 3.2 Feature construction (Sections 2.3, 2.4)

```bash
# GBA features with the leak-safe (r2train) neighbor source
python code/build_mirage_features.py --dataset C --neighbor-source r2train
# Leak-prone source for the clean feature-source ablation (Section 3.1)
python code/build_mirage_features.py --dataset C --neighbor-source mapping80 --out results/score_C_mapping80.csv
```

### 3.3 Reliable negative mining (committee voting, Section 2.5)

```bash
python code/negative_mining_oof.py --dataset C --random-state 42 --score code/results/MiRAGE_score_C_r2.csv
```

### 3.4 GRMF embeddings (Section 2.4)

```bash
python code/pretrain_gigs_split.py --dataset C --random-state 42
python code/verify_grmf_objective.py     # finite-difference verification of the update rules
```

### 3.5 Regular (warm-entity) comparison (Section 3.2)

```bash
python code/eval_holdout_protocol.py --dataset C --random-state 42
# or the full multi-seed ledger:
python code/run_multiseed.py --seeds 42 7 123 2024 999 --datasets C F DDCD
```

### 3.6 Cold-start evaluation (Sections 3.3)

```bash
python code/cold_split.py --dataset C --mode cold-drug --random-state 42
python code/cold_eval.py --dataset C --seed 42 --mode cold-drug
# multi-seed orchestration:
python code/run_multiseed_cold.py --seeds 42 7 123 2024 --datasets C DDCD
```

### 3.7 Boundary and robustness analyses (Section 3.7)

```bash
# Cold-disease
python code/complete_colddis.py
# Pretraining control (MolEmb32 vs ECFP32 vs Shuffled32)
python code/compare_pretrain_ablation.py --dataset C --seeds 42 7 123 2024
# Scaffold-disjoint (chemical cold-start)
python code/scaffold_split.py --dataset C
# Source-level leakage check (features without conditions/category)
```

### 3.8 Deep-method family probe (GCN, Section 3.8)

```bash
python code/gcn_eval.py --dataset C --seed 42 --mode regular
python code/gcn_eval.py --dataset C --seed 42 --mode cold-drug
```

### 3.9 Clean feature-source ablation (Section 3.1)

```bash
python code/ablate_neighbor_source.py C results/score_C_mapping80.csv
```

### 3.10 Negative-sampling comparison and calibration (Sections 3.5, 3.6)

```bash
python code/compare_negative_sampling.py
python code/evaluate_calibration.py --dataset C
```

### 3.11 Figures

```bash
python code/generate_R3_figures.py
```

## 4. Results

Pre-computed per-seed results are provided in `results/`:

| File | Contents |
|---|---|
| `results_manifest.csv` | Regular-setting AUPR/AUROC/F1 per model, dataset, and seed (per-seed feature construction) |
| `cold_start_results.csv` | Cold-drug AUPR per split seed for base / +MolEmb32 / +GRMF / +Both |
| `gcn_results.csv` | Lightweight GCN AUPR under the unified protocol (regular: 5 seeds; cold-drug: 4 seeds per dataset) |
| `colddis_results.csv` | Cold-disease boundary analysis (C-Dataset 4 + DDCD 3 splits) |
| `scaffold_results.csv` | Scaffold-disjoint (chemical cold-start) boundary analysis (4 splits) |
| `feature_source_ablation.csv` | Clean feature-source ablation: leak-safe vs leak-prone source (C-Dataset, DDCD, seed 42) |
| `pretrain_ablation.csv` | Cold-drug per-seed lifts for MolEmb32 / ECFP32 / Shuffled32 |
| `calibration_{C,F,DDCD}.csv` | ECE before/after Platt/isotonic, P@10/R@100 |
| `negative_sampling_compare.csv` | OOF vs random negative sampling |
| `grmf_hyperparam_search_C.csv` | GRMF hyperparameter sensitivity search |

**Numerical precision note**: the CSVs store full-precision AUPR; the relative lifts quoted in the manuscript are rounded to one decimal place from these values, so a lift may differ from a value recomputed from the CSV by up to 0.1--0.2 percentage points.

## 5. Code–paper correspondence (important)

**The code is authoritative.** Every number in the manuscript is produced by this code; the manuscript equations describe the implementation. During review rounds the following discrepancies between earlier equation drafts and the code were found and **fixed in the manuscript** (not in the code):

| Location in manuscript | Paper now states | Implementation |
|---|---|---|
| Eq. (1), GBA features | maxima computed **with the target pair excluded (leave-one-out)** | `leave_one_out_max` in `build_mirage_features.py` |
| Section 2.4, GIP kernels | bandwidth $b_D = \mathrm{mean}_{i,i'}\|\mathbf{a}_{i\cdot}-\mathbf{a}_{i'\cdot}\|^2$ | `gipk()` in `pretrain_gigs_split.py` |
| Section 2.4, GRMF objective & updates | objective without $\tfrac12$ on the $\lambda_2$ term; multiplicative updates as in `docs/GRMF_update_rules.md` | `gigs_model.py` (finite-difference verified to $10^{-9}$) |
| Section 2.4, hyperparameters | $k=64$, $\lambda=(0.1,0.01,0.1)$, 150 iterations | defaults of `gigs_model.py` + explicit arguments in `pretrain_gigs_split.py` |
| MoLFormer PCA | PCA fitted on **unique training drugs** (each drug once), then mapped back to pair rows | `cold_eval.py`, `build_results_ledger.py`, `compare_pretrain_ablation.py` |
| Shuffled32 | **drug-level permutation**: unique drug--embedding correspondences permuted, every row of a drug shares the permuted vector | `mol_feats(shuffle=True)` in `compare_pretrain_ablation.py` |
| Regular multi-seed protocol | GBA features **rebuilt per split seed** (training-local construction) | `run_multiseed.py` calls `build_mirage_features` per seed |

If you find any remaining mismatch between an equation and this code, treat the code as correct and report the discrepancy.

## 6. License

For research use. Contact the corresponding author for details.
