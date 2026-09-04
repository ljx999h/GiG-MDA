# A Leakage-Aware Protocol for Cold-Start Drug–Disease Association Prediction: What Molecular-Structure and Graph-Regularized Features Contribute—and What They Do Not

(GiG-MDA: dual-channel framework evaluated in the manuscript)

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

### 3.11 Case study (Section 3.9): therapeutic-candidate literature evidence

论文 3.9 节案例研究按"部署模型三表"执行, 并以 **CTD therapeutic-only 子集**为已知集
(评审 P0-2: DDCD 原 mapping 混有毒性/不良反应关系, 药物重定向语义需治疗关系):

1. 从官方 CTD 下载 (CTD_chemicals_diseases.csv.gz,
   https://ctdbase.org/reports/CTD_chemicals_diseases.csv.gz) 筛 DirectEvidence
   含 'therapeutic' 的行, 疾病侧 MeSH 直接匹配, 药物侧名称规范化匹配
   (盐型后缀剥离 + 同义词表, 含审计文件):

```bash
# 放到 data/ctd_raw/CTD_chemicals_diseases.csv.gz 后执行:
python code/build_therapeutic_mapping.py     # v1 精确匹配 (基线)
python code/match_therapeutic_v3.py          # v3 最终匹配 → data/DDCD/Mapping/mapping_therapeutic.csv
# 输出: 13,579 治疗关联 (1,178 药物 / 1,030 疾病); 审计 data/ctd_raw/therapeutic_audit.csv
```

2. 部署特征 (邻居 = therapeutic 映射, 目标对留一) 与部署模型三表:

```bash
python code/build_mirage_features.py --dataset DDCD --neighbor-source full \
    --mapping-full data/DDCD/Mapping/mapping_therapeutic.csv \
    --out code/results/MiRAGE_score_DDCD_deploy_therapeutic.csv
python code/case_study_deploy.py --feat-file code/results/MiRAGE_score_DDCD_deploy_therapeutic.csv \
    --drug DB01234 --disease MESH:D013274
# → results/case_study_deploy_top15_{global,drug,gastric}.csv 与每池随机对照
```

3. 文献证据: Crossref 标题级检索 + 人工核验 (治疗方向标准: 阴性/不活跃试验、
   病例报道、体内药效均算 direct; 不良反应/毒性方向与机制类证据归 none):

```bash
python code/case_study_deploy_crossref.py
# 核验结果: results/case_study_deploy_evidence_*_verified.json (含引用/PMID/DOI)
```

4. 随机对照 (P0-1, 20 次 drug-level 重复):

```bash
python code/compare_pretrain_ablation.py --dataset C    --reps 20 --out results/R2/pretrain_ablation_C.csv
python code/compare_pretrain_ablation.py --dataset DDCD --reps 20 --out results/R2/pretrain_ablation.csv
```

**Evidence classification rule.** 与证据文件 (`case_study_deploy_evidence_*_verified.json`)
一致: `direct` = 核验到药物对该疾病具治疗作用的已发表文献 (试验含阴性、队列、
治疗性病例、in-vivo 药效、保护性关联); 药物类别级/机制级/药物诱发的不良反应方向
匹配保守归 `none`。标题级筛选未能确认的同样归 `none`; 原始 Crossref 检索列表
(`case_study_deploy_evidence_*_*.json`) 随代码保留以便审计。

**Superseded by revision.** 上一版冷药物 top-30 案例 (`case_study_cold.py` 及
`results/case_study_cold_*` / `case_study_evidence_v2_*`) 以及全量映射版部署三表
(`*_mtx.csv`) 均已随论文修订被取代; 旧产物保留在本包作为实验记录, 不再被论文引用。

### 3.12 Figures

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
| `negative_sampling_compare.csv` | Committee-consensus filter vs random negative sampling (single split, seed 42) |
| `grmf_hyperparam_search_C.csv` | GRMF hyperparameter sensitivity search |
| `pretrain_ablation.csv` | 随机对照 20 次重复 (drug-level; AUROC+AUPR 每 rep 一行): MiRAGE/MolEmb32/ECFP32/Shuffled32/Gaussian32 (表 9 数据源) |
| `pretrain_ablation_auc_summary.csv` | 每 (dataset, seed) 的 AUROC mean±SD over 20 reps (任务补报) |
| `regularization_diagnostic.csv` | base 特征在 colsample_bytree=0.3 / max_depth=4 下的 AUPR (Discussion 机制段数据源, code/regularization_diagnostic.py) |

| `pretrain_ablation.csv` | 随机对照 20 次重复 (drug-level): 每 rep 一行, 含 MiRAGE/MolEmb32/ECFP32/Shuffled32/Gaussian32 (表 9 数据源) |
| `case_study_deploy_top15_{global,drug,gastric}.csv` | Case study (Section 3.9), 部署模型三表 (已知集 = CTD therapeutic 13,579 对): 全局 / Dexamethasone / 胃癌 |
| `case_study_deploy_random15_{global,drug,gastric}.csv` | 同三个池的 15 对均匀随机对照 |
| `case_study_deploy_full_ranking.csv` | 部署模型对 2,204,351 个 novel 候选的全排序 |
| `case_study_deploy_summary.json` | 池规模、分数区间与模型元信息 |
| `case_study_deploy_evidence_top15_{global,drug,gastric}.json` / `_verified.json` | 主表 45 对 Crossref 检索列表与治疗方向人工核验 (direct/none + 引用) |
| `case_study_deploy_evidence_random15_{global,drug,gastric}.json` / `_verified.json` | 对照 45 对同协议检索与核验 |

| `case_study_cold_top30_{mol,base}.csv` / `case_study_cold_random30_{mol,base}.csv` | Cold-drug case study (Section 3.9): top-30 ranked pairs and 30-pair random control (DDCD, cold split seed 42) for the molecular-channel and base models, with labels and scores |
| `case_study_cold_summary_{mol,base}.json` | P@30 of the top-30 vs random control per variant |
| `case_study_full_ranking_base.csv` | Full base-model ranking of the 443,586 cold candidates (source of the Base rank column in Table 12) |
| `case_study_top30_with_base_rank.csv` | Mol-channel top-30 with the base-model rank of each pair |
| `case_study_stats.json` | Fisher exact tests, drug-concentration checks |
| `cold_dot_results.csv` | Per-seed cold-start AUPR of the base model vs the MiRAGE+dot variant (C, DDCD x 4 seeds; source of the 1-D dot-product cold lifts in the representation-channel ablation; regenerate with `code/verify_dot_cold.py`) |
| `case_study_evidence_v2_top30.csv` / `case_study_evidence_v2_random30.csv` | Per-pair evidence classification (verified/direct/none; `indirect?` retained) with retrieved Crossref titles |

| `case_study_deploy_top15_{global,mtx,gastric}.csv` | Case study (Section 3.9), 部署模型三表: 全局 / Methotrexate / 胃癌 (Stomach Neoplasms) 的 top-15 novel 对 (已知关联已剔除), 含分数 |
| `case_study_deploy_random15_{global,mtx,gastric}.csv` | 同三个池的 15 对均匀随机对照 (证据基率) |
| `case_study_deploy_full_ranking.csv` | 部署模型对 2,175,730 个 novel 候选的全排序 (表 13/14/15 的数据源) |
| `case_study_deploy_summary.json` | 池规模、分数区间与模型元信息 |
| `case_study_deploy_evidence_top15_{global,mtx,gastric}.json` | 每对 Crossref 标题级检索结果 (top-5 命中) |
| `case_study_deploy_evidence_top15_{global,mtx,gastric}_verified.json` | 主表 45 对人工核验结果 (direct/none + 引用/PMID/DOI) |
| `case_study_deploy_evidence_random15_{global,mtx,gastric}.json` | 对照 45 对 Crossref 标题级检索结果 |
| `case_study_deploy_evidence_random15_{global,mtx,gastric}_verified.json` / `_screened.json` | 对照 45 对核验结果 (与主表同协议的深核验 + 纯标题级筛选) |

> 旧冷案例产物 (`case_study_cold_top30_*`、`case_study_evidence_v2_*` 等) 对应论文上一版 3.9 节, 保留为实验记录, 不再被论文引用。

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

**Legacy configuration note.** `r2_config.FEAT_18_MOL` / `FEAT_22_MOL` insert MoLFormer-derived *similarity columns* (`p_score_MolFormer`) into the GBA feature set and belong to the R2-era pipeline. The R3 manuscript's molecular channel (**MolEmb32**) is the *embedding path*: MoLFormer 768-d embeddings (produced by `build_molformer_features.py`, stored in `code/results/molformer/`) PCA-projected to 32 dimensions on unique training drugs only (implemented in `cold_eval.py`, `build_results_ledger.py`, and `compare_pretrain_ablation.py`). No script in this package reads `FEAT_*_MOL`; they are kept for compatibility with historical R2 artifacts. The `docs/GRMF_update_rules.md` derivation likewise describes the R3 equations.

## 6. License

For research use. Contact the corresponding author for details.
