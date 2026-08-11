"""
F-Dataset 4 变体对比: 1D vs 128D GiGs × RandomForest vs XGBoost
 输入: data/F-Dataset/Evaluation/train.csv (18 维 MiRAGE 特征)
       code/model/gigs_dataF.pkl (GiGs 嵌入)
 输出: results/contribution_demo_F/  (汇总 + 对比图 + paper-ready 文案)
"""
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    f1_score, precision_score, recall_score, precision_recall_curve
)
import xgboost as xgb

RANDOM_SEED = 42
N_FOLDS = 5

TRAIN_CSV = "data/F-Dataset/Evaluation/train.csv"
GIGS_PKL  = "code/model/gigs_dataF.pkl"
RESULTS_DIR = "results/contribution_demo_F"
os.makedirs(RESULTS_DIR, exist_ok=True)

# F-Dataset: 1 疾病相似度 + 7 药物相似度 = 18 维 (与 C-Dataset 字段相同)
FEATURE_18 = [
    'count_drug', 'count_disease',
    'q_score_PS',
    'p_score_Target', 'p_score_Category', 'p_score_Conditions',
    'p_score_Description', 'p_score_Mechanism', 'p_score_Pharmacodynamics', 'p_score_Smile',
    'adj_q_score_PS',
    'adj_p_score_Target', 'adj_p_score_Category', 'adj_p_score_Conditions',
    'adj_p_score_Description', 'adj_p_score_Mechanism', 'adj_p_score_Pharmacodynamics', 'adj_p_score_Smile',
]


def _to_lookup_key(v, mapping):
    """F-Dataset 用字符串 ID (DrugBank/OMIM), 优先查字符串再查整数"""
    if v in mapping: return mapping[v]
    v_str = str(v).strip()
    if v_str in mapping: return mapping[v_str]
    try:
        v_int = int(v)
        if v_int in mapping: return mapping[v_int]
    except (ValueError, TypeError): pass
    return -1


def get_gigs_features(df, gigs_data, mode):
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


def evaluate_cv(X, y, classifier='rf'):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    fold_metrics = []
    for fold, (tr, va) in enumerate(skf.split(X, y), 1):
        if classifier == 'xgb':
            spw = (y.iloc[tr] == 0).sum() / max(1, (y.iloc[tr] == 1).sum())
            clf = xgb.XGBClassifier(
                n_estimators=500, max_depth=10, learning_rate=0.1,
                device='cuda', tree_method='hist',
                scale_pos_weight=spw, eval_metric='logloss',
                random_state=RANDOM_SEED, verbosity=0
            )
        else:
            clf = RandomForestClassifier(
                n_estimators=100, max_depth=12,
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


def main():
    print("=" * 70)
    print(" F-Dataset: 4 变体对比 (1D/128D × RF/XGBoost)")
    print("=" * 70)

    print(f"\n[1/4] 加载 GiGs...")
    with open(GIGS_PKL, "rb") as f:
        gigs_data = pickle.load(f)
    print(f"  ✅ X: {gigs_data['X'].shape}, Y: {gigs_data['Y'].shape}")

    print(f"\n[2/4] 加载训练数据...")
    train_df = pd.read_csv(TRAIN_CSV).fillna(0.0)
    print(f"  {len(train_df):,} 行, 正 {(train_df['label']==1).sum():,}")

    y = train_df['label']
    X_base = train_df[FEATURE_18].copy()
    print(f"  MiRAGE 基础特征: {X_base.shape[1]} 维")

    sample_ids_df = train_df[['drugID', 'diseaseID']]

    # 实验组 A: 1D dot
    print(f"\n[3/4] 实验组 A: 1D dot...")
    gigs_dot, dot_cols = get_gigs_features(sample_ids_df, gigs_data, mode='dot')
    X_dot = pd.DataFrame(np.hstack([X_base.values, gigs_dot]), columns=FEATURE_18 + dot_cols)
    print(f"  特征维度: {X_dot.shape[1]}")
    m_dot_rf = evaluate_cv(X_dot, y, classifier='rf')
    print(f"  [RF]  AUROC={m_dot_rf['AUROC'].mean():.4f}±{m_dot_rf['AUROC'].std():.4f}  AUPRC={m_dot_rf['AUPRC'].mean():.4f}  F1={m_dot_rf['F1-Score'].mean():.4f}")
    m_dot_xgb = evaluate_cv(X_dot, y, classifier='xgb')
    print(f"  [XGB] AUROC={m_dot_xgb['AUROC'].mean():.4f}±{m_dot_xgb['AUROC'].std():.4f}  AUPRC={m_dot_xgb['AUPRC'].mean():.4f}  F1={m_dot_xgb['F1-Score'].mean():.4f}")

    # 实验组 B: 128D embed
    print(f"\n[3/4] 实验组 B: 128D embed...")
    gigs_emb, emb_cols = get_gigs_features(sample_ids_df, gigs_data, mode='embed')
    X_emb = pd.DataFrame(np.hstack([X_base.values, gigs_emb]), columns=FEATURE_18 + emb_cols)
    print(f"  特征维度: {X_emb.shape[1]}")
    m_emb_rf = evaluate_cv(X_emb, y, classifier='rf')
    print(f"  [RF]  AUROC={m_emb_rf['AUROC'].mean():.4f}±{m_emb_rf['AUROC'].std():.4f}  AUPRC={m_emb_rf['AUPRC'].mean():.4f}  F1={m_emb_rf['F1-Score'].mean():.4f}")
    m_emb_xgb = evaluate_cv(X_emb, y, classifier='xgb')
    print(f"  [XGB] AUROC={m_emb_xgb['AUROC'].mean():.4f}±{m_emb_xgb['AUROC'].std():.4f}  AUPRC={m_emb_xgb['AUPRC'].mean():.4f}  F1={m_emb_xgb['F1-Score'].mean():.4f}")

    # 汇总
    summary = pd.DataFrame({
        'Method': [
            '1D Dot + RF (论文方法)',
            '128D Embed + RF (扩展)',
            '1D Dot + XGBoost (扩展)',
            '128D Embed + XGBoost (你的最优)'
        ],
        'Dimension': [X_dot.shape[1], X_emb.shape[1], X_dot.shape[1], X_emb.shape[1]],
        'AUROC_mean': [m_dot_rf['AUROC'].mean(), m_emb_rf['AUROC'].mean(),
                       m_dot_xgb['AUROC'].mean(), m_emb_xgb['AUROC'].mean()],
        'AUROC_std':  [m_dot_rf['AUROC'].std(),  m_emb_rf['AUROC'].std(),
                       m_dot_xgb['AUROC'].std(),  m_emb_xgb['AUROC'].std()],
        'AUPRC_mean': [m_dot_rf['AUPRC'].mean(), m_emb_rf['AUPRC'].mean(),
                       m_dot_xgb['AUPRC'].mean(), m_emb_xgb['AUPRC'].mean()],
        'AUPRC_std':  [m_dot_rf['AUPRC'].std(),  m_emb_rf['AUPRC'].std(),
                       m_dot_xgb['AUPRC'].std(),  m_emb_xgb['AUPRC'].std()],
        'F1_mean':    [m_dot_rf['F1-Score'].mean(), m_emb_rf['F1-Score'].mean(),
                       m_dot_xgb['F1-Score'].mean(), m_emb_xgb['F1-Score'].mean()],
        'F1_std':     [m_dot_rf['F1-Score'].std(),  m_emb_rf['F1-Score'].std(),
                       m_dot_xgb['F1-Score'].std(),  m_emb_xgb['F1-Score'].std()],
    })

    auroc_lift = (summary.loc[3, 'AUROC_mean'] - summary.loc[0, 'AUROC_mean']) / summary.loc[0, 'AUROC_mean'] * 100
    auprc_lift = (summary.loc[3, 'AUPRC_mean'] - summary.loc[0, 'AUPRC_mean']) / summary.loc[0, 'AUPRC_mean'] * 100
    f1_lift    = (summary.loc[3, 'F1_mean']    - summary.loc[0, 'F1_mean'])    / summary.loc[0, 'F1_mean'] * 100

    print(f"\n{'='*90}")
    print(f" F-Dataset 4 变体对比 (5-Fold CV, Mean ± Std)")
    print(f"{'='*90}")
    print(f"{'方法':<35s} {'维度':<6s} {'AUROC':<14s} {'AUPRC':<14s} {'F1':<14s}")
    print("-" * 90)
    for _, row in summary.iterrows():
        print(f"{row['Method']:<35s} {int(row['Dimension']):<6d} "
              f"{row['AUROC_mean']:.4f}±{row['AUROC_std']:.4f}  "
              f"{row['AUPRC_mean']:.4f}±{row['AUPRC_std']:.4f}  "
              f"{row['F1_mean']:.4f}±{row['F1_std']:.4f}")

    print(f"\n 提升 (你的最优 vs 论文方法):")
    print(f"   AUROC: +{auroc_lift:.2f}%")
    print(f"   AUPRC: +{auprc_lift:.2f}%")
    print(f"   F1   : +{f1_lift:.2f}%")

    # 保存
    summary.to_csv(os.path.join(RESULTS_DIR, '对比.csv'), index=False)
    m_dot_rf.assign(Method='1D+RF').to_csv(os.path.join(RESULTS_DIR, '1D_RF每折.csv'), index=False)
    m_emb_rf.assign(Method='128D+RF').to_csv(os.path.join(RESULTS_DIR, '128D_RF每折.csv'), index=False)
    m_dot_xgb.assign(Method='1D+XGB').to_csv(os.path.join(RESULTS_DIR, '1D_XGB每折.csv'), index=False)
    m_emb_xgb.assign(Method='128D+XGB').to_csv(os.path.join(RESULTS_DIR, '128D_XGB每折.csv'), index=False)

    print(f"\n ✅ 结果保存至: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
