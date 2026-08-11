"""
XGBoost hyperparameter tuning on F-dataset (equivalent to GIG-MDA Table VIII for RF)
  n_estimators: 300/400/500/600/700 (matching original)
  max_depth: 8/10 (replacing gini/entropy)
  5-Fold CV, threshold per fold on validation set
"""
import os, pickle, time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    accuracy_score, recall_score, f1_score, precision_recall_curve
)
from xgboost import XGBClassifier

# ======================= config =======================
TRAIN_CSV    = "data/F-Dataset/Evaluation/train.csv"
GIGS_PKL     = "code/model/gigs_dataF.pkl"
COL_LABEL    = 'label'
N_FOLDS      = 5
RANDOM_SEED  = 42

# 18-dim MiRAGE (F-Dataset: 1 disease PS + 7 drug) + 128-dim GiGs embed = 146-dim total
FEAT_MIRAGE_18 = (
    ['count_drug', 'count_disease'] +
    ['q_score_PS'] +
    ['adj_q_score_PS'] +
    ['p_score_Target', 'p_score_Category', 'p_score_Conditions',
     'p_score_Description', 'p_score_Mechanism', 'p_score_Pharmacodynamics',
     'p_score_Smile'] +
    ['adj_p_score_Target', 'adj_p_score_Category', 'adj_p_score_Conditions',
     'adj_p_score_Description', 'adj_p_score_Mechanism',
     'adj_p_score_Pharmacodynamics', 'adj_p_score_Smile']
)
FEAT_GIGS_128 = (
    [f'gigs_drug_emb_{i}' for i in range(64)] +
    [f'gigs_disease_emb_{i}' for i in range(64)]
)


def inject_gigs(df, gigs_data):
    if gigs_data is None:
        for i in range(64):
            df[f'gigs_drug_emb_{i}'] = 0.0
            df[f'gigs_disease_emb_{i}'] = 0.0
        return df
    X, Y = gigs_data['X'], gigs_data['Y']
    d2i, s2i = gigs_data['drug_to_idx'], gigs_data['disease_to_idx']
    drug_col    = 'drugID'    if 'drugID'    in df.columns else 'DrugID'
    disease_col = 'diseaseID' if 'diseaseID' in df.columns else 'DiseaseID'
    drug_idx    = df[drug_col].astype(str).str.strip().map(d2i)
    disease_idx = df[disease_col].astype(str).str.strip().map(s2i)
    valid = drug_idx.notna() & disease_idx.notna()
    emb = np.zeros((len(df), 128), dtype=np.float32)
    if valid.any():
        d = drug_idx[valid].astype(int).values
        s = disease_idx[valid].astype(int).values
        emb[valid.values, :64] = X[d]
        emb[valid.values, 64:] = Y[s]
    emb_df = pd.DataFrame(emb, columns=FEAT_GIGS_128, index=df.index)
    return pd.concat([df, emb_df], axis=1)


def evaluate_xgb(X, y, n_est, max_d):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    folds = []
    for tr, va in skf.split(X, y):
        clf = XGBClassifier(
            n_estimators=n_est, max_depth=max_d, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            tree_method='hist', device='cuda',
            random_state=RANDOM_SEED, eval_metric='logloss', verbosity=0
        )
        clf.fit(X.iloc[tr], y.iloc[tr])
        y_prob = clf.predict_proba(X.iloc[va])[:, 1]
        p, r, t = precision_recall_curve(y.iloc[va], y_prob)
        f1_s = 2 * (p * r) / (p + r + 1e-9)
        best = f1_s.argmax()
        thresh = t[best] if best < len(t) else 0.5
        y_pred = (y_prob >= thresh).astype(int)
        folds.append({
            'AUROC': roc_auc_score(y.iloc[va], y_prob),
            'AUPRC': average_precision_score(y.iloc[va], y_prob),
            'Recall': recall_score(y.iloc[va], y_pred),
            'Accuracy': accuracy_score(y.iloc[va], y_pred),
            'F1': f1_score(y.iloc[va], y_pred),
        })
    df_metrics = pd.DataFrame(folds)
    return {k: df_metrics[k].mean() for k in df_metrics.columns}


def main():
    print("="*70)
    print(" XGBoost hyperparameter tuning on F-Dataset")
    print("="*70)

    df = pd.read_csv(TRAIN_CSV).fillna(0.0)
    print(f"  Loaded: {len(df):,} rows")

    with open(GIGS_PKL, "rb") as f:
        gigs_data = pickle.load(f)
    df = inject_gigs(df, gigs_data)
    available = [c for c in FEAT_MIRAGE_18 + FEAT_GIGS_128 if c in df.columns]
    X = df[available]
    y = df[COL_LABEL]
    print(f"  Features: {X.shape[1]} | Pos: {(y==1).sum():,} | Neg: {(y==0).sum():,}")

    # Parameter grid
    n_estimators_list = [300, 400, 500, 600, 700]
    max_depth_list = [8, 10]

    results = []
    for n_est in n_estimators_list:
        for md in max_depth_list:
            t0 = time.time()
            print(f"  n={n_est}, d={md}", end=" ... ", flush=True)
            m = evaluate_xgb(X, y, n_est, md)
            elapsed = time.time() - t0
            results.append({
                'n_estimators': n_est,
                'max_depth': md,
                'AUROC': m['AUROC'],
                'AUPRC': m['AUPRC'],
                'Recall': m['Recall'],
                'Accuracy': m['Accuracy'],
                'F1-score': m['F1'],
                'Time(s)': elapsed
            })
            print(f"AUROC={m['AUROC']:.5f} F1={m['F1']:.5f} ({elapsed:.0f}s)")

    # Print LaTeX table
    print("\n\n" + "="*70)
    print(" LaTeX Table")
    print("="*70)
    print(r"""
\begin{table*}[ht]
\centering
\caption{Performance of XGBoost models with different hyperparameters on the F-dataset.}
\label{tab:xgb_f_dataset}
\begin{tabular}{ccrrrrr}
\toprule
\multicolumn{1}{c}{$n\_\mathrm{estimators}$} &
\multicolumn{1}{c}{$\mathrm{max\_depth}$} &
AUROC & AUPRC & Recall & Accuracy & F1-score \\
\midrule""")

    for r in results:
        n, d = r['n_estimators'], r['max_depth']
        auroc, auprc = r['AUROC'], r['AUPRC']
        rec, acc, f1 = r['Recall'], r['Accuracy'], r['F1-score']
        # bold the best F1
        best_f1 = max(r['F1-score'] for r in results)
        if abs(r['F1-score'] - best_f1) < 1e-6:
            print(f"\\textbf{{{n}}} & \\textbf{{{d}}}  & \\textbf{{{auroc:.5f}}} & \\textbf{{{auprc:.5f}}} & \\textbf{{{rec:.5f}}} & \\textbf{{{acc:.5f}}} & \\textbf{{{f1:.5f}}} \\\\")
        else:
            print(f"{n} & {d}  & {auroc:.5f} & {auprc:.5f} & {rec:.5f} & {acc:.5f} & {f1:.5f} \\\\")

    print(r"""\bottomrule
\end{tabular}
\end{table*}
""")

    # Save raw CSV
    pd.DataFrame(results).to_csv(
        "results/xgb_hyperparam_F.csv", index=False
    )
    print("  Saved to results/xgb_hyperparam_F.csv")


if __name__ == "__main__":
    main()
