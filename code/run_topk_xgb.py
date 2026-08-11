"""
XGBoost Top-k Feature Importance on DDCD (150-dim)
 替换论文 Table X (旧 RF 数据)
 输出: results/figures_data/topk_xgb.csv → LaTeX 表格
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

RANDOM_SEED = 42
N_FOLDS = 5
SAMPLE_SIZE = 500000  # DDCD 2.1M 行 → 采样 50万 加速

TRAIN_CSV = "data/DDCD/Evaluation/train.csv"
GIGS_PKL  = "code/model/gigs_dataDDCD.pkl"
OUTPUT_CSV = "results/figures_data/topk_xgb.csv"

FEAT_MIRAGE_22 = (
    ['count_drug', 'count_disease'] +
    ['q_score_Description', 'q_score_Pathway', 'q_score_Slim'] +
    ['adj_q_score_Description', 'adj_q_score_Pathway', 'adj_q_score_Slim'] +
    ['p_score_Target', 'p_score_Category', 'p_score_Conditions',
     'p_score_Description', 'p_score_Mechanism', 'p_score_Pharmacodynamics', 'p_score_Smile'] +
    ['adj_p_score_Target', 'adj_p_score_Category', 'adj_p_score_Conditions',
     'adj_p_score_Description', 'adj_p_score_Mechanism', 'adj_p_score_Pharmacodynamics',
     'adj_p_score_Smile']
)
FEAT_GIGS_128 = [f'gigs_drug_emb_{i}' for i in range(64)] + \
                [f'gigs_disease_emb_{i}' for i in range(64)]


def inject_gigs(df, gigs_data):
    if gigs_data is None:
        for c in FEAT_GIGS_128:
            df[c] = 0.0
        return df
    Xm, Ym = gigs_data['X'], gigs_data['Y']
    d2i, s2i = gigs_data['drug_to_idx'], gigs_data['disease_to_idx']
    dc = 'drugID' if 'drugID' in df.columns else 'DrugID'
    sc = 'diseaseID' if 'diseaseID' in df.columns else 'DiseaseID'
    di = np.array([d2i.get(str(d).strip(), -1) for d in df[dc]], dtype=np.int32)
    si = np.array([s2i.get(str(s).strip(), -1) for s in df[sc]], dtype=np.int32)
    v = (di >= 0) & (si >= 0)
    emb = np.zeros((len(df), 128), dtype=np.float32)
    if v.any():
        emb[v, :64] = Xm[di[v]]
        emb[v, 64:] = Ym[si[v]]
    return pd.concat([df, pd.DataFrame(emb, columns=FEAT_GIGS_128, index=df.index)], axis=1)


def evaluate_features(X, y):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    folds = []
    for tr, va in skf.split(X, y):
        spw = (y.iloc[tr] == 0).sum() / max(1, (y.iloc[tr] == 1).sum())
        clf = XGBClassifier(
            n_estimators=500, max_depth=10, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=spw,
            tree_method='hist', device='cuda',
            random_state=RANDOM_SEED, verbosity=0
        )
        clf.fit(X.iloc[tr], y.iloc[tr])
        yp = clf.predict_proba(X.iloc[va])[:, 1]
        p, r, t = precision_recall_curve(y.iloc[va], yp)
        f1s = 2 * (p * r) / (p + r + 1e-9)
        th = t[f1s.argmax()] if f1s.argmax() < len(t) else 0.5
        ypred = (yp >= th).astype(int)
        folds.append({
            'AUROC': roc_auc_score(y.iloc[va], yp),
            'AUPRC': average_precision_score(y.iloc[va], yp),
            'Accuracy': accuracy_score(y.iloc[va], ypred),
            'Recall': recall_score(y.iloc[va], ypred),
            'F1': f1_score(y.iloc[va], ypred),
        })
    return {k: np.mean([f[k] for f in folds]) for k in folds[0]}


def main():
    print("=" * 60)
    print(" XGBoost Top-k Feature Importance (DDCD, 150-dim)")
    print("=" * 60)

    df = pd.read_csv(TRAIN_CSV).fillna(0.0)
    gigs = pickle.load(open(GIGS_PKL, 'rb'))
    df = inject_gigs(df, gigs)
    all_cols = [c for c in FEAT_MIRAGE_22 + FEAT_GIGS_128 if c in df.columns]
    X = df[all_cols]
    y = df['label']

    # 采样
    if SAMPLE_SIZE and len(X) > SAMPLE_SIZE:
        pos = np.where(y == 1)[0]
        neg = np.random.RandomState(RANDOM_SEED).choice(
            np.where(y == 0)[0], SAMPLE_SIZE - len(pos), replace=False)
        keep = np.concatenate([pos, neg])
        X = X.iloc[keep].reset_index(drop=True)
        y = y.iloc[keep].reset_index(drop=True)
    print(f"  {len(X):,} rows, {X.shape[1]} features, pos={(y==1).sum():,}")

    # Step 1: 全量训练 → 特征重要性
    print("\n[1/2] 训练 XGBoost 获取特征重要性...")
    t0 = time.time()
    spw = (y == 0).sum() / max(1, (y == 1).sum())
    clf = XGBClassifier(
        n_estimators=500, max_depth=10, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw, tree_method='hist', device='cuda',
        random_state=RANDOM_SEED, verbosity=0
    )
    clf.fit(X, y)

    importances = clf.feature_importances_
    feat_rank = pd.DataFrame({
        'Feature': all_cols, 'Importance': importances
    }).sort_values('Importance', ascending=False).reset_index(drop=True)
    top20 = feat_rank['Feature'].head(20).tolist()
    print(f"  Top-5: {top20[:5]}")
    print(f"  Done ({time.time()-t0:.0f}s)")

    # Step 2: Top-k 顺序添加
    print(f"\n[2/2] Top-k 顺序特征评估 (k=1..20)...")
    results = []
    for k in range(1, 21):
        cols_k = top20[:k]
        t1 = time.time()
        m = evaluate_features(X[cols_k], y)
        results.append({
            'k': k,
            'AUROC': m['AUROC'], 'AUPRC': m['AUPRC'],
            'Accuracy': m['Accuracy'], 'Recall': m['Recall'], 'F1-score': m['F1'],
            'Time(s)': round(time.time() - t1, 0),
        })
        print(f"  k={k:2d}  AUROC={m['AUROC']:.5f}  AUPRC={m['AUPRC']:.5f}  "
              f"F1={m['F1']:.5f}  ({results[-1]['Time(s)']:.0f}s)")

    res_df = pd.DataFrame(results)
    res_df.to_csv(OUTPUT_CSV, index=False)

    # LaTeX 表
    print(f"\n{'='*60}")
    print(" LaTeX Table (替换论文 TABLE X)")
    print(f"{'='*60}")
    print(r"\begin{table}[ht]")
    print(r"\centering")
    print(r"\small")
    print(r"\caption{Performance metrics with top-$k$ features on the DDCD dataset (XGBoost, 150-dim).}")
    print(r"\label{tab:topk_results}")
    print(r"\begin{tabular}{crrrrr}")
    print(r"\toprule")
    print(r"$k$ & AUROC & AUPRC & Accuracy & Recall & F1-Score \\")
    print(r"\midrule")
    for _, r in res_df.iterrows():
        print(f"{int(r['k'])} & {r['AUROC']:.5f} & {r['AUPRC']:.5f} & "
              f"{r['Accuracy']:.5f} & {r['Recall']:.5f} & {r['F1-score']:.5f} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

    print(f"\n  Saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
