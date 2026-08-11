"""
4 变体评估共享模块: 1D/128D × RF/XGBoost
 让 DDCC_main.py, eval_C.py, eval_F.py, ablation_study.py 都能调用
"""
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    f1_score, precision_score, recall_score, precision_recall_curve
)
import xgboost as xgb

# 默认参数
N_FOLDS = 5
RANDOM_SEED = 42
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 12
XGB_N_ESTIMATORS = 500
XGB_MAX_DEPTH = 10


def _to_lookup_key(v, mapping):
    """兼容整数/字符串 ID (DDCD 整数, F-Dataset 字符串, C-Dataset 整数)"""
    if v in mapping: return mapping[v]
    v_str = str(v).strip()
    if v_str in mapping: return mapping[v_str]
    try:
        v_int = int(v)
        if v_int in mapping: return mapping[v_int]
    except (ValueError, TypeError): pass
    return -1


def get_gigs_features(df, gigs_data, mode):
    """
    获取 GiGs 特征
    mode='dot': 1 维点积
    mode='embed': 128 维嵌入拼接
    """
    X_emb = gigs_data["X"]; Y_emb = gigs_data["Y"]
    d2i, s2i = gigs_data["drug_to_idx"], gigs_data["disease_to_idx"]
    k = X_emb.shape[1]

    d_idx = np.array([_to_lookup_key(d, d2i) for d in df['drugID']], dtype=np.int32)
    s_idx = np.array([_to_lookup_key(s, s2i) for s in df['diseaseID']], dtype=np.int32)
    valid = (d_idx >= 0) & (s_idx >= 0)
    n = len(df)

    if mode == 'dot':
        feats = np.zeros(n, dtype=np.float32)
        if valid.any():
            feats[valid] = np.sum(X_emb[d_idx[valid]] * Y_emb[s_idx[valid]], axis=1)
        return feats.reshape(-1, 1), ['score_gigs']
    else:
        d_feats = np.zeros((n, k), dtype=np.float32)
        s_feats = np.zeros((n, k), dtype=np.float32)
        if valid.any():
            d_feats[valid] = X_emb[d_idx[valid]]
            s_feats[valid] = Y_emb[s_idx[valid]]
        feats = np.hstack([d_feats, s_feats])
        cols = [f'gigs_drug_emb_{i}' for i in range(k)] + [f'gigs_disease_emb_{i}' for i in range(k)]
        return feats, cols


def inject_gigs_columns(df, gigs_data, mode):
    """
    将 GiGs 特征列直接拼接到 df 上, 返回新的 df
    """
    feats, cols = get_gigs_features(df, gigs_data, mode)
    for i, col in enumerate(cols):
        df[col] = feats[:, i]
    return df, cols


def evaluate_cv(X, y, classifier='rf'):
    """
    5-Fold CV 评估 (单分类器)
    classifier: 'rf' (论文方法) | 'xgb' (扩展, 适合高维稀疏)
    """
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    fold_metrics = []
    for fold, (tr, va) in enumerate(skf.split(X, y), 1):
        if classifier == 'xgb':
            spw = (y.iloc[tr] == 0).sum() / max(1, (y.iloc[tr] == 1).sum())
            clf = xgb.XGBClassifier(
                n_estimators=XGB_N_ESTIMATORS, max_depth=XGB_MAX_DEPTH,
                learning_rate=0.1, device='cuda', tree_method='hist',
                scale_pos_weight=spw, eval_metric='logloss',
                random_state=RANDOM_SEED, verbosity=0
            )
        else:
            clf = RandomForestClassifier(
                n_estimators=RF_N_ESTIMATORS, max_depth=RF_MAX_DEPTH,
                n_jobs=-1, random_state=RANDOM_SEED
            )
        clf.fit(X.iloc[tr], y.iloc[tr])
        y_prob = clf.predict_proba(X.iloc[va])[:, 1]

        p, r, t = precision_recall_curve(y.iloc[va], y_prob)
        f1_s = 2 * (p * r) / (p + r + 1e-9)
        best = f1_s.argmax()
        thresh = t[best] if best < len(t) else 0.5
        y_pred = (y_prob >= thresh).astype(int)

        fold_metrics.append({
            'Fold': fold,
            'AUROC': roc_auc_score(y.iloc[va], y_prob),
            'AUPRC': average_precision_score(y.iloc[va], y_prob),
            'Accuracy': accuracy_score(y.iloc[va], y_pred),
            'Precision': precision_score(y.iloc[va], y_pred, zero_division=0),
            'Recall': recall_score(y.iloc[va], y_pred, zero_division=0),
            'F1-Score': f1_score(y.iloc[va], y_pred, zero_division=0),
        })
    return pd.DataFrame(fold_metrics)


def run_4_variants(X_dot, X_emb, y, dataset_name=""):
    """
    跑 4 变体: 1D×RF, 128D×RF, 1D×XGB, 128D×XGB
    返回: summary_df (4行), all_metrics (dict)
    """
    variants = [
        ('1D Dot + RF (论文方法)', X_dot, 'rf'),
        ('128D Embed + RF (扩展)', X_emb, 'rf'),
        ('1D Dot + XGBoost (扩展)', X_dot, 'xgb'),
        ('128D Embed + XGBoost (你的最优)', X_emb, 'xgb'),
    ]

    print(f"\n[{dataset_name}] 4 变体 5-Fold CV:")
    print("=" * 90)
    print(f"{'方法':<35s} {'维度':<6s} {'AUROC':<14s} {'AUPRC':<14s} {'F1':<14s}")
    print("-" * 90)

    summary_rows = []
    all_metrics = {}

    for name, X, clf_type in variants:
        print(f"  跑 {name} ({clf_type.upper()})...", end=" ", flush=True)
        m = evaluate_cv(X, y, classifier=clf_type)
        row = {
            'Method': name,
            'Classifier': clf_type.upper(),
            'Dimension': X.shape[1],
            'AUROC_mean': m['AUROC'].mean(),
            'AUROC_std': m['AUROC'].std(),
            'AUPRC_mean': m['AUPRC'].mean(),
            'AUPRC_std': m['AUPRC'].std(),
            'F1_mean': m['F1-Score'].mean(),
            'F1_std': m['F1-Score'].std(),
            'Acc_mean': m['Accuracy'].mean(),
            'Rec_mean': m['Recall'].mean(),
        }
        summary_rows.append(row)
        all_metrics[name] = m
        print(f"AUROC={row['AUROC_mean']:.4f}±{row['AUROC_std']:.4f}  "
              f"AUPRC={row['AUPRC_mean']:.4f}  F1={row['F1_mean']:.4f}")

    summary_df = pd.DataFrame(summary_rows)

    # 提升: 你的最优 vs 论文方法 (loc[3] vs loc[0])
    auroc_lift = (summary_df.loc[3, 'AUROC_mean'] - summary_df.loc[0, 'AUROC_mean']) / summary_df.loc[0, 'AUROC_mean'] * 100
    auprc_lift = (summary_df.loc[3, 'AUPRC_mean'] - summary_df.loc[0, 'AUPRC_mean']) / summary_df.loc[0, 'AUPRC_mean'] * 100
    f1_lift    = (summary_df.loc[3, 'F1_mean'] - summary_df.loc[0, 'F1_mean']) / summary_df.loc[0, 'F1_mean'] * 100

    print("-" * 90)
    print(f"  提升 (你的最优 vs 论文方法):")
    print(f"    AUROC: +{auroc_lift:.2f}%")
    print(f"    AUPRC: +{auprc_lift:.2f}%")
    print(f"    F1   : +{f1_lift:.2f}%")
    print("=" * 90)

    return summary_df, all_metrics, {
        'auroc_lift': auroc_lift,
        'auprc_lift': auprc_lift,
        'f1_lift': f1_lift,
    }
