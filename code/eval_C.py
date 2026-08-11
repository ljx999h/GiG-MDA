"""
C-Dataset 一站式评估 (5-Fold CV + 消融)
 输入: data/C-Dataset/Evaluation/train.csv (18 维 MiRAGE 特征)
       code/model/gigs_dataC.pkl (GiGs 嵌入)
 输出: results/C_Dataset/  (汇总 + 对比图)
"""
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    f1_score, precision_score, recall_score, precision_recall_curve
)

N_FOLDS = 5
RANDOM_SEED = 42
QUICK_SAMPLE = None  # None=全量

# 路径
TRAIN_CSV = "data/C-Dataset/Evaluation/train.csv"
TEST_CSV  = "data/C-Dataset/Evaluation/test.csv"
GIGS_PKL  = "code/model/gigs_dataC.pkl"
RESULTS_DIR = "results/C_Dataset"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ================================================================
# 18 维 MiRAGE 特征
# ================================================================
FEAT_COUNT = ['count_drug', 'count_disease']
FEAT_DISEASE_RAW = ['q_score_PS']
FEAT_DRUG_RAW = ['p_score_Target', 'p_score_Category', 'p_score_Conditions',
                 'p_score_Description', 'p_score_Mechanism', 'p_score_Pharmacodynamics', 'p_score_Smile']
FEAT_DISEASE_ADJ = ['adj_q_score_PS']
FEAT_DRUG_ADJ = ['adj_p_score_Target', 'adj_p_score_Category', 'adj_p_score_Conditions',
                 'adj_p_score_Description', 'adj_p_score_Mechanism', 'adj_p_score_Pharmacodynamics', 'adj_p_score_Smile']
FEAT_ALL_18 = (FEAT_COUNT + FEAT_DISEASE_RAW + FEAT_DRUG_RAW +
               FEAT_DISEASE_ADJ + FEAT_DRUG_ADJ)

FEAT_GIGS = ([f'gigs_drug_emb_{i}' for i in range(64)] +
             [f'gigs_disease_emb_{i}' for i in range(64)])


def inject_gigs(df, gigs_data):
    """向量化注入 128 维 GiGs 嵌入"""
    if gigs_data is None:
        for i in range(64):
            df[f'gigs_drug_emb_{i}'] = 0.0
            df[f'gigs_disease_emb_{i}'] = 0.0
        return df

    X_emb, Y_emb = gigs_data['X'], gigs_data['Y']
    d2i, s2i = gigs_data['drug_to_idx'], gigs_data['disease_to_idx']
    k = X_emb.shape[1]

    # C-Dataset 用整数 ID
    d_idx = np.array([d2i.get(int(d), -1) for d in df['drugID']], dtype=np.int32)
    s_idx = np.array([s2i.get(int(s), -1) for s in df['diseaseID']], dtype=np.int32)
    valid = (d_idx >= 0) & (s_idx >= 0)
    n = len(df)

    emb_data = np.zeros((n, 128), dtype=np.float32)
    if valid.any():
        emb_data[valid, :64] = X_emb[d_idx[valid]]
        emb_data[valid, 64:] = Y_emb[s_idx[valid]]

    emb_cols = [f'gigs_drug_emb_{i}' for i in range(64)] + [f'gigs_disease_emb_{i}' for i in range(64)]
    emb_df = pd.DataFrame(emb_data, columns=emb_cols, index=df.index)
    return pd.concat([df, emb_df], axis=1)


def evaluate_cv(X, y):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    fold_metrics = []
    for fold, (tr, va) in enumerate(skf.split(X, y), 1):
        # 论文: RandomForest n_estimators=100, max_depth=12
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
    print(" C-Dataset 一站式评估 (RandomForest + 5-Fold CV (论文方法))")
    print("=" * 70)

    # 加载 GiGs
    print(f"\n[1] 加载 GiGs...")
    gigs_data = None
    if os.path.exists(GIGS_PKL):
        with open(GIGS_PKL, 'rb') as f:
            gigs_data = pickle.load(f)
        print(f"  ✅ X: {gigs_data['X'].shape}, Y: {gigs_data['Y'].shape}")
    else:
        print(f"  [WARN] GiGs 未找到, 将跳过 GiGs 相关实验")

    # 加载训练数据
    print(f"\n[2] 加载训练数据...")
    train_df = pd.read_csv(TRAIN_CSV).fillna(0.0)
    train_df = inject_gigs(train_df, gigs_data)
    print(f"  {len(train_df):,} 行, 正 {(train_df['label']==1).sum():,}")

    y = train_df['label']
    X_base = train_df[FEAT_ALL_18].copy()

    # 采样
    if QUICK_SAMPLE and len(train_df) > QUICK_SAMPLE:
        pos_mask = y == 1
        n_pos = pos_mask.sum()
        n_neg_sample = QUICK_SAMPLE - n_pos
        neg_idx = np.random.RandomState(RANDOM_SEED).choice(
            np.where(~pos_mask)[0], n_neg_sample, replace=False)
        keep = np.concatenate([np.where(pos_mask)[0], neg_idx])
        X_base = X_base.iloc[keep].reset_index(drop=True)
        y = y.iloc[keep].reset_index(drop=True)
        print(f"  采样: {len(X_base):,} | 正 {(y==1).sum():,} | 负 {(y==0).sum():,}")
    else:
        keep = np.arange(len(train_df))  # 全量 (用于消融变体索引)

    # ==============================================================
    # 消融实验 (8 变体, 不含 w/o Cross-Multiplication)
    # ==============================================================
    print(f"\n[3] 消融实验 (8 变体 × 5-Fold CV)...")

    ABLATION_VARIANTS = [
        ("Full Model (MiRAGE)", FEAT_ALL_18 + FEAT_GIGS, "18维MiRAGE + 128维GiGs"),
        ("w/o GiGs (No Graph)", FEAT_ALL_18, "仅18维MiRAGE"),
        ("GiGs Embedding Only (128D)", FEAT_GIGS, "仅GiGs嵌入"),
        ("w/o Count Features", FEAT_DISEASE_RAW + FEAT_DRUG_RAW + FEAT_DISEASE_ADJ + FEAT_DRUG_ADJ + FEAT_GIGS, "无count_*"),
        ("w/o Drug Features", FEAT_COUNT + FEAT_DISEASE_RAW + FEAT_DISEASE_ADJ + FEAT_GIGS, "无药物特征"),
        ("w/o Disease Features", FEAT_COUNT + FEAT_DRUG_RAW + FEAT_DRUG_ADJ + FEAT_GIGS, "无疾病特征"),
        ("Count + GiGs Only", FEAT_COUNT + FEAT_GIGS, "仅拓扑+图嵌入"),
        ("Similarity Only (no Count, no GiGs)",
         FEAT_DISEASE_RAW + FEAT_DRUG_RAW + FEAT_DISEASE_ADJ + FEAT_DRUG_ADJ, "仅相似度分数"),
    ]

    ablation_results = []
    for name, feats, desc in ABLATION_VARIANTS:
        print(f"\n  >>> {name}")
        X_variant = train_df[feats].iloc[keep].reset_index(drop=True) if QUICK_SAMPLE else train_df[feats]
        m = evaluate_cv(X_variant, y)
        row = {'Variant': name, 'Description': desc, '#Feat': len(feats)}
        for k in ['AUROC', 'AUPRC', 'Accuracy', 'F1-Score']:
            row[f'{k}_mean'] = m[k].mean()
            row[f'{k}_std'] = m[k].std()
        ablation_results.append(row)
        print(f"      AUROC={row['AUROC_mean']:.4f}±{row['AUROC_std']:.4f}  "
              f"AUPRC={row['AUPRC_mean']:.4f}±{row['AUPRC_std']:.4f}  "
              f"F1={row['F1-Score_mean']:.4f}±{row['F1-Score_std']:.4f}")

    # ==============================================================
    # 汇总
    # ==============================================================
    print(f"\n[5] 汇总结果...")

    summary_df = pd.DataFrame(ablation_results).sort_values('AUROC_mean', ascending=False)
    summary_df.to_csv(os.path.join(RESULTS_DIR, 'ablation_summary.csv'), index=False)

    # 打印消融结果
    print(f"\n{'='*70}")
    print(f" 消融实验结果 (按 AUROC 排序)")
    print(f"{'='*70}")
    print(f"{'变体':<35s} {'#Feat':<6s} {'AUROC':<14s} {'AUPRC':<14s} {'F1':<14s}")
    for _, r in summary_df.iterrows():
        print(f"{r['Variant']:<35s} {int(r['#Feat']):<6d} "
              f"{r['AUROC_mean']:.4f}±{r['AUROC_std']:.4f}  "
              f"{r['AUPRC_mean']:.4f}±{r['AUPRC_std']:.4f}  "
              f"{r['F1-Score_mean']:.4f}±{r['F1-Score_std']:.4f}")

    # 可视化
    plot_results(summary_df)

    # ----------------------------------------------------------
    # 4 变体扩展实验 (1D/128D × RF/XGBoost)
    # ----------------------------------------------------------
    print(f"\n[4-变体] 1D/128D × RF/XGBoost 扩展实验...")
    try:
        from eval_helpers_4variants import get_gigs_features, run_4_variants
        sample_ids = train_df[['drugID', 'diseaseID']]
        dot_feats, dot_cols = get_gigs_features(sample_ids, gigs_data, mode='dot')
        X_dot = train_df[FEAT_ALL_18].copy()
        for i, col in enumerate(dot_cols):
            X_dot[col] = dot_feats[:, i]
        emb_feats, emb_cols = get_gigs_features(sample_ids, gigs_data, mode='embed')
        X_emb = train_df[FEAT_ALL_18].copy()
        for i, col in enumerate(emb_cols):
            X_emb[col] = emb_feats[:, i]
        summary_4v, all_metrics, lifts = run_4_variants(X_dot, X_emb, y, dataset_name="C-Dataset")
        summary_4v.to_csv(os.path.join(RESULTS_DIR, '4variants_summary.csv'), index=False)
        for name, m in all_metrics.items():
            safe = name.replace(' ', '_').replace('(', '').replace(')', '').replace('+', '_')
            m.to_csv(os.path.join(RESULTS_DIR, f'4v_{safe}.csv'), index=False)
        print(f"\n  4 变体结果已保存至: {RESULTS_DIR}/4v_*.csv")
    except Exception as e:
        print(f"  [WARN] 4 变体实验失败: {e}")

    print(f"\n ✅ 所有结果已保存至: {RESULTS_DIR}/")


def plot_results(summary_df):
    """单图: 消融实验 6 指标对比"""
    fig, ax = plt.subplots(figsize=(10, max(5, len(summary_df) * 0.5)))

    df = summary_df.sort_values('AUROC_mean', ascending=True)
    y_pos = range(len(df))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(df)))

    # 水平柱状图 (AUROC 排序)
    ax.barh(y_pos, df['AUROC_mean'].values,
            xerr=df['AUROC_std'].values, color=colors,
            edgecolor='black', alpha=0.85, capsize=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df['Variant'].values, fontsize=10)
    ax.set_xlabel('AUROC (5-Fold CV)', fontsize=11)
    ax.set_title('C-Dataset Ablation Study (XGBoost GPU, 18D MiRAGE + 128D GiGs)',
                 fontsize=12, fontweight='bold')

    # 标注每条柱子的 AUROC 数值
    for i, (m_val, s_val) in enumerate(zip(df['AUROC_mean'].values, df['AUROC_std'].values)):
        ax.text(m_val + 0.002, i, f'{m_val:.4f}', va='center', fontsize=9, fontweight='bold')

    ax.grid(axis='x', alpha=0.3)
    ax.set_xlim(0.5, 1.02)

    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, 'eval_summary.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    main()
