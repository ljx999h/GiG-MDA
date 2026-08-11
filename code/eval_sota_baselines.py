"""
6 个经典 SOTA baselines for drug-disease association prediction
- Logistic Regression (LR)
- PREDICT (Gottlieb 2011): 5-similarity score + Logistic Regression
- KNN (k=5) over the MiRAGE+GiGs-dot feature space
- Random Forest (RF)
- XGBoost (GPU)
- LightGBM (CPU)

Protocol (identical to eval_helpers_4variants.evaluate_cv):
  - StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
  - Features: MiRAGE features (22 / 18) + GiGs 1-D dot product
  - Threshold selected on validation set via precision_recall_curve (F1 max)
  - Report 5-fold Mean +/- Std for AUROC, AUPRC, Accuracy, Precision, Recall, F1

Outputs (under results/baselines_sota/):
  - all_summary.csv  : 3 datasets x 6 baselines summary
  - {dataset}_{baseline}.csv : per-fold detail
  - baselines_comparison.png : bar-chart comparison
  - baselines_vs_4variants.csv : merged table of baselines + 4 MiRAGE variants
"""

import os
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    f1_score, precision_score, recall_score, precision_recall_curve,
)

import xgboost as xgb
import lightgbm as lgb

# =======================================================================
# Configuration
# =======================================================================
N_FOLDS = 5
RANDOM_SEED = 42

# Maximum rows used for training (to keep runtime manageable on DDCD ~2.1M).
# Subsample once with stratified sampling (preserve positive class).
MAX_TRAIN_ROWS = 600_000

# Dataset registry
DATASETS = {
    "DDCD": {
        "train_csv": "data/DDCD/Evaluation/train.csv",
        "gigs_pkl":  "code/model/gigs_dataDDCD.pkl",
        "mirage_features": [
            'count_drug', 'count_disease',
            'q_score_Description', 'q_score_Pathway', 'q_score_Slim',
            'p_score_Target', 'p_score_Category', 'p_score_Conditions',
            'p_score_Description', 'p_score_Mechanism', 'p_score_Pharmacodynamics', 'p_score_Smile',
            'adj_q_score_Description', 'adj_q_score_Pathway', 'adj_q_score_Slim',
            'adj_p_score_Target', 'adj_p_score_Category', 'adj_p_score_Conditions',
            'adj_p_score_Description', 'adj_p_score_Mechanism', 'adj_p_score_Pharmacodynamics',
            'adj_p_score_Smile',
        ],
        "results_dir": "results/baselines_sota",
    },
    "C-Dataset": {
        "train_csv": "data/C-Dataset/Evaluation/train.csv",
        "gigs_pkl":  "code/model/gigs_dataC.pkl",
        "mirage_features": [
            'count_drug', 'count_disease',
            'q_score_PS',
            'p_score_Target', 'p_score_Category', 'p_score_Conditions',
            'p_score_Description', 'p_score_Mechanism', 'p_score_Pharmacodynamics', 'p_score_Smile',
            'adj_q_score_PS',
            'adj_p_score_Target', 'adj_p_score_Category', 'adj_p_score_Conditions',
            'adj_p_score_Description', 'adj_p_score_Mechanism', 'adj_p_score_Pharmacodynamics', 'adj_p_score_Smile',
        ],
        "results_dir": "results/baselines_sota",
    },
    "F-Dataset": {
        "train_csv": "data/F-Dataset/Evaluation/train.csv",
        "gigs_pkl":  "code/model/gigs_dataF.pkl",
        "mirage_features": [
            'count_drug', 'count_disease',
            'q_score_PS',
            'p_score_Target', 'p_score_Category', 'p_score_Conditions',
            'p_score_Description', 'p_score_Mechanism', 'p_score_Pharmacodynamics', 'p_score_Smile',
            'adj_q_score_PS',
            'adj_p_score_Target', 'adj_p_score_Category', 'adj_p_score_Conditions',
            'adj_p_score_Description', 'adj_p_score_Mechanism', 'adj_p_score_Pharmacodynamics', 'adj_p_score_Smile',
        ],
        "results_dir": "results/baselines_sota",
    },
}

RESULTS_DIR = "results/baselines_sota"
os.makedirs(RESULTS_DIR, exist_ok=True)


# =======================================================================
# Helpers
# =======================================================================
def _to_lookup_key(v, mapping):
    """ID lookup that tolerates int / string / numeric-string variants."""
    if v in mapping:
        return mapping[v]
    v_str = str(v).strip()
    if v_str in mapping:
        return mapping[v_str]
    try:
        v_int = int(v)
        if v_int in mapping:
            return mapping[v_int]
    except (ValueError, TypeError):
        pass
    return -1


def add_gigs_dot(df, gigs_data):
    """Add 1-D GiGs dot-product feature to df in-place (matches eval_helpers)."""
    if gigs_data is None:
        df['score_gigs'] = 0.0
        return ['score_gigs']
    X_emb = gigs_data["X"]; Y_emb = gigs_data["Y"]
    d2i = gigs_data["drug_to_idx"]; s2i = gigs_data["disease_to_idx"]

    d_idx = np.array([_to_lookup_key(d, d2i) for d in df['drugID']], dtype=np.int32)
    s_idx = np.array([_to_lookup_key(s, s2i) for s in df['diseaseID']], dtype=np.int32)
    valid = (d_idx >= 0) & (s_idx >= 0)

    scores = np.zeros(len(df), dtype=np.float32)
    if valid.any():
        scores[valid] = np.sum(
            X_emb[d_idx[valid]] * Y_emb[s_idx[valid]], axis=1
        ).astype(np.float32)
    df['score_gigs'] = scores
    return ['score_gigs']


def stratified_subsample(X, y, max_n, seed=RANDOM_SEED):
    """Stratified random subsample, preserving all positives if n_pos < max_n."""
    if max_n is None or len(X) <= max_n:
        return X.reset_index(drop=True), y.reset_index(drop=True)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    rng = np.random.RandomState(seed)
    n_pos = len(pos_idx)
    if n_pos >= max_n:
        keep = rng.choice(pos_idx, max_n, replace=False)
    else:
        n_neg = max_n - n_pos
        n_neg = min(n_neg, len(neg_idx))
        keep = np.concatenate([pos_idx, rng.choice(neg_idx, n_neg, replace=False)])
    rng.shuffle(keep)
    return X.iloc[keep].reset_index(drop=True), y.iloc[keep].reset_index(drop=True)


# =======================================================================
# 6 baseline classifiers
# =======================================================================
def make_lr():
    """Standardised features are handled outside (scaler fit per fold)."""
    return LogisticRegression(
        max_iter=1000, C=1.0, solver='liblinear', random_state=RANDOM_SEED
    )


def make_predict(mirage_features):
    """
    PREDICT (Gottlieb 2011). Faithful to the paper:
      score(d, s) = sum over 5 similarity components
    In our MiRAGE feature space the closest analogues are the 5 strongest
    single-similarity scores per (drug, disease) pair. We use:
      - 4 drug-side scores (Target, Mechanism, Pharmacodynamics, Description)
      - 1 disease-side score (the only disease similarity in MiRAGE)
    Each component is standardised before summation; the aggregated score is
    then fed to a Logistic Regression classifier (PREDICT uses score-based
    association rather than direct thresholding).
    """
    drug_sim = [f for f in mirage_features
                if f.startswith('p_score_') and f != 'p_score_Smile']
    disease_sim = [f for f in mirage_features if f.startswith('q_score_')]

    # The 5 similarities used by PREDICT, mapped to MiRAGE where possible.
    predict_components = drug_sim[:4] + disease_sim[:1]
    if len(predict_components) == 0:
        # Fallback: use first 5 mirage features
        predict_components = mirage_features[:5]

    # Build a small Pipeline-style wrapper: sum-std + LR
    class PREDICTClassifier:
        def __init__(self, components, seed=RANDOM_SEED):
            self.components = components
            self.scaler = StandardScaler()
            self.clf = LogisticRegression(
                max_iter=1000, C=1.0, solver='liblinear', random_state=seed
            )

        def fit(self, X, y):
            sub = X[self.components].values
            self.scaler.fit(sub)
            z = self.scaler.transform(sub)
            score = z.sum(axis=1)
            self.clf.fit(score.reshape(-1, 1), y)
            return self

        def predict_proba(self, X):
            sub = X[self.components].values
            z = self.scaler.transform(sub)
            score = z.sum(axis=1).reshape(-1, 1)
            return self.clf.predict_proba(score)

    return PREDICTClassifier(predict_components)


def make_knn():
    return KNeighborsClassifier(n_neighbors=5, n_jobs=-1, weights='distance')


def make_rf():
    return RandomForestClassifier(
        n_estimators=100, max_depth=12, n_jobs=-1, random_state=RANDOM_SEED
    )


def make_xgb():
    return xgb.XGBClassifier(
        n_estimators=500, max_depth=10, learning_rate=0.1,
        device='cuda', tree_method='hist', eval_metric='logloss',
        random_state=RANDOM_SEED, verbosity=0
    )


def make_lgb():
    # LightGBM GPU disabled (build may not support it); CPU is fast enough
    # on the MiRAGE feature space.
    return lgb.LGBMClassifier(
        n_estimators=500, max_depth=10, learning_rate=0.1,
        n_jobs=-1, random_state=RANDOM_SEED, verbose=-1
    )


# Each entry: name -> dict(factory, needs_scaler, notes).
# PREDICT is built inside evaluate_cv_baseline because it needs mirage_features.
BASELINES = {
    "LR": {
        "factory":      make_lr,
        "needs_scaler": True,
        "notes":        "Logistic Regression (liblinear)",
    },
    "PREDICT": {
        "factory":      None,
        "needs_scaler": False,
        "notes":        "Gottlieb 2011 (5-sim + LR)",
    },
    "KNN": {
        "factory":      make_knn,
        "needs_scaler": True,
        "notes":        "k=5, distance-weighted",
    },
    "RandomForest": {
        "factory":      make_rf,
        "needs_scaler": False,
        "notes":        "n=100, d=12 (paper baseline)",
    },
    "XGBoost": {
        "factory":      make_xgb,
        "needs_scaler": False,
        "notes":        "n=500, d=10, GPU",
    },
    "LightGBM": {
        "factory":      make_lgb,
        "needs_scaler": False,
        "notes":        "n=500, d=10, CPU",
    },
}


# =======================================================================
# Core CV evaluation (re-uses exact protocol from eval_helpers_4variants)
# =======================================================================
def evaluate_cv_baseline(X, y, baseline_name, mirage_features):
    """Single baseline, 5-fold CV. Returns (per_fold_df, seconds)."""
    cfg = BASELINES[baseline_name]
    factory_fn = cfg["factory"]
    needs_scaler = cfg["needs_scaler"]

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    fold_metrics = []
    t0 = time.time()

    for fold, (tr, va) in enumerate(skf.split(X, y), 1):
        X_tr, X_va = X.iloc[tr], X.iloc[va]
        y_tr, y_va = y.iloc[tr], y.iloc[va]

        if needs_scaler:
            scaler = StandardScaler()
            X_tr_s = pd.DataFrame(
                scaler.fit_transform(X_tr),
                columns=X_tr.columns, index=X_tr.index
            )
            X_va_s = pd.DataFrame(
                scaler.transform(X_va),
                columns=X_va.columns, index=X_va.index
            )
        else:
            X_tr_s, X_va_s = X_tr, X_va

        if baseline_name == "PREDICT":
            clf = make_predict(mirage_features)
        else:
            clf = factory_fn()

        clf.fit(X_tr_s, y_tr)
        y_prob = clf.predict_proba(X_va_s)[:, 1]

        # Validation-set threshold via precision_recall_curve (F1 max)
        p, r, t = precision_recall_curve(y_va, y_prob)
        f1_s = 2 * (p * r) / (p + r + 1e-9)
        best = f1_s.argmax()
        thresh = t[best] if best < len(t) else 0.5
        y_pred = (y_prob >= thresh).astype(int)

        fold_metrics.append({
            'Fold': fold,
            'Threshold': float(thresh),
            'AUROC':      roc_auc_score(y_va, y_prob),
            'AUPRC':      average_precision_score(y_va, y_prob),
            'Accuracy':   accuracy_score(y_va, y_pred),
            'Precision':  precision_score(y_va, y_pred, zero_division=0),
            'Recall':     recall_score(y_va, y_pred, zero_division=0),
            'F1-Score':   f1_score(y_va, y_pred, zero_division=0),
        })
    elapsed = time.time() - t0
    return pd.DataFrame(fold_metrics), elapsed


def summarise(fold_df):
    out = {}
    for m in ['AUROC', 'AUPRC', 'Accuracy', 'Precision', 'Recall', 'F1-Score']:
        out[f'{m}_mean'] = fold_df[m].mean()
        out[f'{m}_std']  = fold_df[m].std()
    return out


def fmt(mean, std):
    return f"{mean:.4f} +/- {std:.4f}"


# =======================================================================
# Main driver
# =======================================================================
def run_dataset(name, cfg):
    print()
    print("=" * 90)
    print(f" Dataset: {name}")
    print("=" * 90)

    # 1) Load data + inject GiGs 1-D dot
    print(f"[load] {cfg['train_csv']}")
    df = pd.read_csv(cfg['train_csv']).fillna(0.0)
    print(f"       raw rows: {len(df):,} | "
          f"pos {(df['label']==1).sum():,} | neg {(df['label']==0).sum():,}")

    if not os.path.exists(cfg['gigs_pkl']):
        raise FileNotFoundError(cfg['gigs_pkl'])
    with open(cfg['gigs_pkl'], 'rb') as f:
        gigs_data = pickle.load(f)

    gigs_cols = add_gigs_dot(df, gigs_data)
    feature_cols = cfg['mirage_features'] + gigs_cols
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing features: {missing}")

    X = df[feature_cols]
    y = df['label']
    print(f"[feat] final feature dim = {X.shape[1]} "
          f"(MiRAGE {len(cfg['mirage_features'])} + GiGs-dot 1)")

    # 2) Stratified subsample (only if dataset > MAX_TRAIN_ROWS)
    if len(X) > MAX_TRAIN_ROWS:
        X, y = stratified_subsample(X, y, MAX_TRAIN_ROWS)
        print(f"[subsample] -> {len(X):,} rows "
              f"(pos {(y==1).sum():,} | neg {(y==0).sum():,})")

    # 3) Run 6 baselines
    results_summary = []
    per_baseline = {}
    print(f"\n[baselines] running 6 baselines on {name} ({len(X):,} rows) ...")
    print("-" * 90)
    for bl_name, bl_cfg in BASELINES.items():
        notes = bl_cfg["notes"]
        print(f"  -> {bl_name:<11s} ({notes}) ...", end=" ", flush=True)
        t0 = time.time()
        fold_df, elapsed = evaluate_cv_baseline(
            X, y, bl_name, cfg['mirage_features']
        )
        elapsed = time.time() - t0
        row = {
            'Dataset':     name,
            'Baseline':    bl_name,
            'Notes':       notes,
            'Dim':         X.shape[1],
            'Rows':        len(X),
            'Elapsed_sec': round(elapsed, 1),
        }
        row.update(summarise(fold_df))
        results_summary.append(row)
        per_baseline[bl_name] = fold_df
        print(f"AUROC={row['AUROC_mean']:.4f}+/-{row['AUROC_std']:.4f}  "
              f"AUPRC={row['AUPRC_mean']:.4f}  "
              f"F1={row['F1-Score_mean']:.4f}  "
              f"[{elapsed:.1f}s]")

    summary_df = pd.DataFrame(results_summary)

    # 4) Per-baseline per-fold CSVs
    for bl_name, fold_df in per_baseline.items():
        out_csv = os.path.join(RESULTS_DIR, f"{name}_{bl_name}.csv")
        fold_df.to_csv(out_csv, index=False)

    # 5) Per-dataset pretty-print
    print()
    print("-" * 90)
    print(f" {name} -- 6 Baselines (5-Fold CV, Mean +/- Std)")
    print("-" * 90)
    header = f"{'Baseline':<13s} {'Dim':>4s}  {'AUROC':<18s} {'AUPRC':<18s} {'F1':<18s} {'Acc':<18s}"
    print(header)
    print("-" * len(header))
    for _, r in summary_df.iterrows():
        print(f"{r['Baseline']:<13s} {int(r['Dim']):>4d}  "
              f"{fmt(r['AUROC_mean'],    r['AUROC_std']):<18s} "
              f"{fmt(r['AUPRC_mean'],    r['AUPRC_std']):<18s} "
              f"{fmt(r['F1-Score_mean'], r['F1-Score_std']):<18s} "
              f"{fmt(r['Accuracy_mean'], r['Accuracy_std']):<18s}")

    return summary_df


def merge_with_4variants(all_summary):
    """Append the 4 MiRAGE variants from existing summary CSVs."""
    extra_files = {
        "DDCD":     "results/4variants_summary.csv",
        "C-Dataset":"results/C_Dataset/4variants_summary.csv",
        "F-Dataset":"results/F_Dataset/4variants_summary.csv",
    }
    rows = []
    for ds, path in extra_files.items():
        if not os.path.exists(path):
            print(f"  [warn] no 4-variant file at {path}, skip")
            continue
        v = pd.read_csv(path)
        # v columns: Method, Classifier, Dimension, AUROC_mean, AUROC_std, AUPRC_mean, AUPRC_std, F1_mean, F1_std, Acc_mean, Rec_mean
        for _, r in v.iterrows():
            rows.append({
                'Dataset':    ds,
                'Baseline':   r['Method'],
                'Notes':      f"MiRAGE variant ({r['Classifier']}, {int(r['Dimension'])}D)",
                'Dim':        int(r['Dimension']),
                'Rows':       np.nan,
                'AUROC_mean': float(r['AUROC_mean']),
                'AUROC_std':  float(r['AUROC_std']),
                'AUPRC_mean': float(r['AUPRC_mean']),
                'AUPRC_std':  float(r['AUPRC_std']),
                'F1-Score_mean': float(r['F1_mean']),
                'F1-Score_std':  float(r['F1_std']),
                'Accuracy_mean': float(r['Acc_mean']),
                'Accuracy_std':  0.0,
                'Precision_mean': np.nan,
                'Precision_std':  np.nan,
                'Recall_mean':    float(r['Rec_mean']),
                'Recall_std':     0.0,
            })
    if not rows:
        return all_summary.copy()
    extra_df = pd.DataFrame(rows)
    merged = pd.concat([all_summary, extra_df], ignore_index=True, sort=False)
    return merged


def plot_comparison(merged_df, out_path):
    """Per-dataset grouped bar-chart of AUROC / AUPRC / F1 across all methods."""
    metrics = ['AUROC_mean', 'AUPRC_mean', 'F1-Score_mean']
    metric_labels = ['AUROC', 'AUPRC', 'F1-Score']
    datasets = merged_df['Dataset'].unique().tolist()

    fig, axes = plt.subplots(len(metrics), 1, figsize=(14, 4.5 * len(metrics)))
    if len(metrics) == 1:
        axes = [axes]

    palette = plt.cm.tab20(np.linspace(0, 1, 20))
    for ax, metric, label in zip(axes, metrics, metric_labels):
        x = np.arange(len(datasets))
        method_names = merged_df['Baseline'].unique().tolist()
        n_methods = len(method_names)
        width = 0.85 / max(n_methods, 1)
        for i, m in enumerate(method_names):
            ys, errs = [], []
            for ds in datasets:
                sel = merged_df[(merged_df['Dataset'] == ds) &
                                (merged_df['Baseline'] == m)]
                if len(sel) == 0:
                    ys.append(np.nan); errs.append(0)
                else:
                    row = sel.iloc[0]
                    metric_std = metric.replace('_mean', '_std')
                    ys.append(float(row[metric]))
                    errs.append(float(row.get(metric_std, 0)) or 0)
            ax.bar(x + (i - n_methods/2) * width + width/2, ys,
                   width=width, yerr=errs, capsize=2,
                   label=m, color=palette[i % len(palette)],
                   edgecolor='black', linewidth=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, fontsize=11)
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(f'{label} (5-Fold CV Mean +/- Std)', fontsize=12)
        ax.set_ylim(0, 1.05)
        ax.grid(axis='y', alpha=0.3)
        ax.legend(ncol=4, fontsize=8, loc='lower right')
    fig.suptitle('SOTA Baselines vs MiRAGE variants (3 datasets)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close()
    print(f"[plot] saved {out_path}")


def main():
    t_all = time.time()
    all_rows = []
    for name, cfg in DATASETS.items():
        ds_summary = run_dataset(name, cfg)
        all_rows.append(ds_summary)
        all_summary = pd.concat(all_rows, ignore_index=True)

        # Save running all_summary as we go (in case of crash)
        all_summary_path = os.path.join(RESULTS_DIR, 'all_summary.csv')
        all_summary.to_csv(all_summary_path, index=False)
        print(f"[save] {all_summary_path}")

    # Merge with 4 MiRAGE variants
    merged = merge_with_4variants(all_summary)
    merged_path = os.path.join(RESULTS_DIR, 'baselines_vs_4variants.csv')
    # Reorder columns
    front_cols = ['Dataset', 'Baseline', 'Notes', 'Dim', 'Rows',
                  'AUROC_mean', 'AUROC_std', 'AUPRC_mean', 'AUPRC_std',
                  'F1-Score_mean', 'F1-Score_std',
                  'Accuracy_mean', 'Accuracy_std',
                  'Precision_mean', 'Precision_std',
                  'Recall_mean', 'Recall_std']
    cols = [c for c in front_cols if c in merged.columns] + \
           [c for c in merged.columns if c not in front_cols]
    merged = merged[cols]
    merged.to_csv(merged_path, index=False)
    print(f"[save] {merged_path}")

    # Plot comparison
    plot_comparison(merged, os.path.join(RESULTS_DIR, 'baselines_comparison.png'))

    # Best baseline vs best MiRAGE variant per dataset
    print()
    print("=" * 90)
    print(" Best baseline vs best MiRAGE variant (5-Fold CV)")
    print("=" * 90)
    for ds in merged['Dataset'].unique():
        sub = merged[merged['Dataset'] == ds]
        baselines_only = sub[~sub['Baseline'].str.contains('1D|128D', regex=True)]
        mirage_only    = sub[sub['Baseline'].str.contains('1D|128D', regex=True)]
        if len(baselines_only) == 0 or len(mirage_only) == 0:
            continue
        best_b = baselines_only.sort_values('AUROC_mean', ascending=False).iloc[0]
        best_m = mirage_only.sort_values('AUROC_mean', ascending=False).iloc[0]
        lift_a = (best_m['AUROC_mean'] - best_b['AUROC_mean']) / max(best_b['AUROC_mean'], 1e-9) * 100
        lift_p = (best_m['AUPRC_mean'] - best_b['AUPRC_mean']) / max(best_b['AUPRC_mean'], 1e-9) * 100
        lift_f = (best_m['F1-Score_mean'] - best_b['F1-Score_mean']) / max(best_b['F1-Score_mean'], 1e-9) * 100
        print(f"\n  [{ds}]")
        print(f"    best baseline : {best_b['Baseline']:<20s} "
              f"AUROC={best_b['AUROC_mean']:.4f}  "
              f"AUPRC={best_b['AUPRC_mean']:.4f}  "
              f"F1={best_b['F1-Score_mean']:.4f}")
        print(f"    best MiRAGE   : {best_m['Baseline']:<20s} "
              f"AUROC={best_m['AUROC_mean']:.4f}  "
              f"AUPRC={best_m['AUPRC_mean']:.4f}  "
              f"F1={best_m['F1-Score_mean']:.4f}")
        print(f"    lift          : AUROC +{lift_a:.2f}%  "
              f"AUPRC +{lift_p:.2f}%  F1 +{lift_f:.2f}%")

    print(f"\n[done] all in {time.time()-t_all:.1f}s")


if __name__ == "__main__":
    main()
