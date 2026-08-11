"""
=============================================================================
 MiRAGE 全面消融实验 (Ablation Study)
 论文: BIB 2024, bbae337
 审稿人 Issue #5: 消融实验不足以证明各模块的关键贡献

 设计原则:
   1. 每个消融变体运行 5-Fold CV, 报告 Mean ± Std
   2. 阈值在验证集上选取 (不在测试集上调优)
   3. 覆盖三大类消融: 特征组 / 负采样策略 / 分类器
=============================================================================
"""
import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    f1_score, precision_score, recall_score, precision_recall_curve
)

# XGBoost: 唯一分类器 (已替代 sklearn RandomForest)
try:
    import xgboost as xgb
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
    # 检测 GPU 可用性
    try:
        _test_params = {'tree_method': 'hist', 'device': 'cuda'}
        _ = xgb.XGBClassifier(n_estimators=1, **_test_params)
        XGB_DEVICE = 'cuda'
    except Exception:
        XGB_DEVICE = 'cpu'
except ImportError:
    XGB_AVAILABLE = False
    XGB_DEVICE = 'cpu'
    print("[ERROR] xgboost 未安装, 请先 `pip install xgboost`")

# ================================================================
# 配置
# ================================================================
TRAIN_CSV = "data/DDCD/Evaluation/train.csv"
TEST_CSV  = "data/DDCD/Evaluation/test.csv"
GIGS_PKL  = "code/model/gigs_dataDDCD.pkl"

COL_LABEL = 'label'
N_FOLDS = 5
RANDOM_SEED = 42
RESULTS_DIR = 'results/ablation'
os.makedirs(RESULTS_DIR, exist_ok=True)

# 快速模式: 训练集采样大小 (None=全量)
# 全量 ~2.1M 行跑 9 变体×5 折 ≈ 很慢, 设 500000 可快速验证
QUICK_SAMPLE = None  # None=全量, 500000=快速验证

# GPU 加速: XGBoost (device='cuda' 自动检测; 无 GPU 时回退 CPU)
XGB_N_ESTIMATORS = 500   # XGBoost 树数 — 原200太保守
XGB_MAX_DEPTH = 10       # XGBoost 深度 — 原8太浅,交叉特征学不到
XGB_LEARNING_RATE = 0.05  # 学习率
XGB_SUBSAMPLE = 0.8       # 行采样
XGB_COLSAMPLE = 0.8       # 列采样

# ================================================================
# 特征分组定义
# ================================================================
# 拓扑计数 (2)
FEAT_COUNT = ['count_drug', 'count_disease']

# 疾病相似度原始 (3)
FEAT_DISEASE_RAW = ['q_score_Description', 'q_score_Pathway', 'q_score_Slim']
# 疾病相似度交叉乘法 (3)
FEAT_DISEASE_ADJ = ['adj_q_score_Description', 'adj_q_score_Pathway', 'adj_q_score_Slim']

# 药物相似度原始 (7)
FEAT_DRUG_RAW = ['p_score_Target', 'p_score_Category', 'p_score_Conditions',
                 'p_score_Description', 'p_score_Mechanism', 'p_score_Pharmacodynamics',
                 'p_score_Smile']
# 药物相似度交叉乘法 (7)
FEAT_DRUG_ADJ = ['adj_p_score_Target', 'adj_p_score_Category', 'adj_p_score_Conditions',
                 'adj_p_score_Description', 'adj_p_score_Mechanism', 'adj_p_score_Pharmacodynamics',
                 'adj_p_score_Smile']

# 汇总
FEAT_ALL_22 = (FEAT_COUNT + FEAT_DISEASE_RAW + FEAT_DRUG_RAW +
               FEAT_DISEASE_ADJ + FEAT_DRUG_ADJ)

# GiGs 特征 (注入, 128维嵌入, 你的改进方案)
FEAT_GIGS_EMB = ([f'gigs_drug_emb_{i}' for i in range(64)] +
                 [f'gigs_disease_emb_{i}' for i in range(64)])
FEAT_GIGS = FEAT_GIGS_EMB


# ================================================================
# 消融变体定义
# ================================================================
# 格式: (变体名, 特征列表, 描述)
#
# ★ 设计原则:
#   - Full Model 包含 22维MiRAGE + 128维GiGs嵌入 (你的改进方案)
#   - 不再对比点积方案 (已在 contribution_demo.py 中独立对比)
#   - 重点证明: 128维GiGs嵌入是真正的核心贡献
ABLATION_VARIANTS = [
    # --- 核心对照 ---
    ("Full Model (MiRAGE)", FEAT_ALL_22 + FEAT_GIGS,
     "完整模型: 22维MiRAGE + 128维GiGs药物/疾病嵌入"),

    ("w/o GiGs (No Graph)", FEAT_ALL_22,
     "移除所有GiGs图特征, 仅保留22维MiRAGE"),

    ("GiGs Embedding Only (128D)",
     FEAT_GIGS,
     "仅128维GiGs嵌入, 无MiRAGE特征 — 测试图嵌入独立预测能力"),

    # --- 特征组消融 ---
    ("w/o Count Features",
     FEAT_DISEASE_RAW + FEAT_DRUG_RAW + FEAT_DISEASE_ADJ + FEAT_DRUG_ADJ + FEAT_GIGS,
     "移除count_drug/count_disease拓扑计数特征"),

    ("w/o Drug Features",
     FEAT_COUNT + FEAT_DISEASE_RAW + FEAT_DISEASE_ADJ + FEAT_GIGS,
     "移除全部7种药物特征 (原始+交叉)"),

    ("w/o Disease Features",
     FEAT_COUNT + FEAT_DRUG_RAW + FEAT_DRUG_ADJ + FEAT_GIGS_EMB,
     "移除全部3种疾病特征 (原始+交叉)"),

    # --- 极端消融 ---
    ("Count + GiGs Only",
     FEAT_COUNT + FEAT_GIGS_EMB,
     "仅拓扑计数+图嵌入, 移除所有相似度分数"),

    ("Similarity Only (no Count, no GiGs)",
     FEAT_DISEASE_RAW + FEAT_DRUG_RAW + FEAT_DISEASE_ADJ + FEAT_DRUG_ADJ,
     "仅相似度分数, 无拓扑信息无图特征"),
]

# ================================================================


def inject_gigs(df, gigs_data):
    """
    向量化注入 GiGs 图特征 (无泄漏: 基于 mapping80 训练)

    注入内容 (128 维, 你的改进方案):
      - gigs_drug_emb_0..63:     drug 嵌入 (64维)
      - gigs_disease_emb_0..63:  disease 嵌入 (64维)
      → 共 128 维 GiGs 特征, 保留完整图结构信息

    论文原方案只用 1 维点积压缩, 大量信息丢失.
    新方案用 128 维嵌入让 XGBoost 自己学交互.
    """
    if gigs_data is None:
        for i in range(64):
            df[f'gigs_drug_emb_{i}'] = 0.0
            df[f'gigs_disease_emb_{i}'] = 0.0
        return df

    X, Y = gigs_data['X'], gigs_data['Y']  # (1410, 64), (1573, 64)
    d2i = gigs_data['drug_to_idx']
    s2i = gigs_data['disease_to_idx']

    # 识别 ID 列名
    drug_col = 'drugID' if 'drugID' in df.columns else 'DrugID'
    disease_col = 'diseaseID' if 'diseaseID' in df.columns else 'DiseaseID'

    # 向量化: 将 drug/disease ID 映射为矩阵索引
    drug_idx = df[drug_col].astype(str).str.strip().map(d2i)
    disease_idx = df[disease_col].astype(str).str.strip().map(s2i)
    valid_mask = drug_idx.notna() & disease_idx.notna()

    # 准备嵌入数组 (128维)
    emb_data = np.zeros((len(df), 128), dtype=np.float32)

    if valid_mask.any():
        d_idx_arr = drug_idx[valid_mask].astype(int).values
        s_idx_arr = disease_idx[valid_mask].astype(int).values

        # drug embedding (64维) → cols 0..63
        emb_data[valid_mask.values, :64] = X[d_idx_arr].astype(np.float32)
        # disease embedding (64维) → cols 64..127
        emb_data[valid_mask.values, 64:] = Y[s_idx_arr].astype(np.float32)

    # 构造列名
    emb_cols = [f'gigs_drug_emb_{i}' for i in range(64)] + \
               [f'gigs_disease_emb_{i}' for i in range(64)]

    # 一次性 concat (避免 fragmentation warning)
    emb_df = pd.DataFrame(emb_data, columns=emb_cols, index=df.index)
    df = pd.concat([df, emb_df], axis=1)
    return df


def evaluate_variant_cv(X_train_full, y_train_full, feature_list, variant_name):
    """
    对单个消融变体运行 5-Fold CV

    Returns
    -------
    metrics_df : DataFrame — 每折指标
    summary : dict — 均值±标准差
    """
    # 过滤可用特征
    available = [c for c in feature_list if c in X_train_full.columns]
    missing = [c for c in feature_list if c not in X_train_full.columns]
    if missing:
        print(f"    [WARN] {variant_name}: 缺失 {len(missing)} 特征")

    X = X_train_full[available]

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_train_full), start=1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y_train_full.iloc[train_idx], y_train_full.iloc[val_idx]

        # XGBoost (替代原 sklearn RandomForest)
        clf = XGBClassifier(
            n_estimators=XGB_N_ESTIMATORS,
            max_depth=XGB_MAX_DEPTH,
            learning_rate=XGB_LEARNING_RATE,
            subsample=XGB_SUBSAMPLE,
            colsample_bytree=XGB_COLSAMPLE,
            tree_method='hist',
            device=XGB_DEVICE,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            eval_metric='logloss',
        )
        clf.fit(X_tr, y_tr)
        y_prob = clf.predict_proba(X_val)[:, 1]

        # 在验证集上选阈值 (非测试集!)
        precisions, recalls, thresholds = precision_recall_curve(y_val, y_prob)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
        best_idx = np.argmax(f1_scores)
        best_thresh = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

        y_pred = (y_prob >= best_thresh).astype(int)

        fold_results.append({
            'Fold': fold,
            'Threshold': best_thresh,
            'AUROC': roc_auc_score(y_val, y_prob),
            'AUPRC': average_precision_score(y_val, y_prob),
            'Accuracy': accuracy_score(y_val, y_pred),
            'Precision': precision_score(y_val, y_pred, zero_division=0),
            'Recall': recall_score(y_val, y_pred, zero_division=0),
            'F1-Score': f1_score(y_val, y_pred, zero_division=0),
        })

    metrics_df = pd.DataFrame(fold_results)

    # 汇总
    summary = {'Variant': variant_name, 'Features': len(available)}
    for m in ['AUROC', 'AUPRC', 'Accuracy', 'Precision', 'Recall', 'F1-Score']:
        summary[m] = f"{metrics_df[m].mean():.4f} ± {metrics_df[m].std():.4f}"
        summary[f'{m}_mean'] = metrics_df[m].mean()
        summary[f'{m}_std'] = metrics_df[m].std()

    return metrics_df, summary


def main():
    print("=" * 70)
    print(" MiRAGE 全面消融实验 (5-Fold CV)")
    if XGB_AVAILABLE:
        device_tag = "🚀 GPU (cuda)" if XGB_DEVICE == 'cuda' else "🖥️  CPU"
        print(f" {device_tag} 模式: XGBoost (n={XGB_N_ESTIMATORS}, d={XGB_MAX_DEPTH}, lr={XGB_LEARNING_RATE})")
    else:
        raise RuntimeError("XGBoost 未安装,无法运行消融实验")
    print("=" * 70)

    # ----------------------------------------------------------
    # 1. 加载 GiGs
    # ----------------------------------------------------------
    print(f"\n[1/4] 加载 GiGs...")
    gigs_data = None
    if os.path.exists(GIGS_PKL):
        with open(GIGS_PKL, "rb") as f:
            gigs_data = pickle.load(f)
        print(f"  OK — X: {gigs_data['X'].shape}, Y: {gigs_data['Y'].shape}")
    else:
        print(f"  [WARN] GiGs 未找到, 将跳过 GiGs 相关实验")

    # ----------------------------------------------------------
    # 2. 加载训练数据
    # ----------------------------------------------------------
    print(f"\n[2/4] 加载训练数据...")
    train_df = pd.read_csv(TRAIN_CSV).fillna(0.0)
    train_df = inject_gigs(train_df, gigs_data)
    print(f"  {len(train_df):,} 行 | 正样本 {(train_df[COL_LABEL]==1).sum():,}")

    X_train = train_df.drop(columns=[COL_LABEL], errors='ignore')
    y_train = train_df[COL_LABEL]

    # 移除 ID 列和非特征列
    id_cols = [c for c in X_train.columns if c.lower() in
               ['drugid', 'diseaseid', 'numpreds', 'numpred', 'unnamed: 0']]
    X_train = X_train.drop(columns=id_cols, errors='ignore')
    print(f"  特征列: {len(X_train.columns)}")

    # 快速模式: 采样 (保持正负比例)
    if QUICK_SAMPLE and len(train_df) > QUICK_SAMPLE:
        print(f"  ⚡ 快速模式: 采样 {QUICK_SAMPLE:,} 行 (全量 {len(train_df):,})")
        pos_mask = y_train == 1
        neg_mask = y_train == 0
        n_pos_sample = min(pos_mask.sum(), QUICK_SAMPLE // 64)  # 保持 ~1:63 比例
        n_neg_sample = QUICK_SAMPLE - n_pos_sample
        sample_idx = (np.random.RandomState(RANDOM_SEED).choice(
            np.where(neg_mask)[0], n_neg_sample, replace=False))
        keep_idx = np.concatenate([np.where(pos_mask)[0], sample_idx])
        X_train = X_train.iloc[keep_idx]
        y_train = y_train.iloc[keep_idx]
        print(f"  采样后: {len(X_train):,} 行 | "
              f"正 {(y_train==1).sum():,} | 负 {(y_train==0).sum():,}")

    # ----------------------------------------------------------
    # 3. 逐变体 5-Fold CV
    # ----------------------------------------------------------
    print(f"\n[3/4] 运行 {len(ABLATION_VARIANTS)} 个消融变体 (每个 {N_FOLDS}-Fold CV)...")
    print("-" * 70)

    all_summaries = []
    all_fold_details = []

    for variant_name, feature_list, description in ABLATION_VARIANTS:
        print(f"\n  >>> {variant_name}")
        print(f"      {description}")

        fold_df, summary = evaluate_variant_cv(
            X_train, y_train, feature_list, variant_name
        )
        all_summaries.append(summary)
        all_fold_details.append(fold_df.assign(Variant=variant_name))

        # 打印每折均值
        print(f"      AUROC={summary['AUROC']}  "
              f"AUPRC={summary['AUPRC']}  "
              f"F1={summary['F1-Score']}")

    # ----------------------------------------------------------
    # 4. 汇总 & 保存
    # ----------------------------------------------------------
    print(f"\n[4/4] 汇总结果...")

    summary_df = pd.DataFrame(all_summaries)
    # 按 AUPRC 排序 (AUROC 在 0.95-0.97 已饱和, AUPRC 更具区分力)
    summary_df = summary_df.sort_values('AUPRC_mean', ascending=False)

    # 保存
    summary_df['Classifier'] = f'XGBoost (n={XGB_N_ESTIMATORS}, d={XGB_MAX_DEPTH}, device={XGB_DEVICE})'
    summary_df.to_csv(os.path.join(RESULTS_DIR, 'ablation_summary.csv'), index=False)
    all_folds = pd.concat(all_fold_details, ignore_index=True)
    all_folds.to_csv(os.path.join(RESULTS_DIR, 'ablation_all_folds.csv'), index=False)

    # --- 打印汇总表 ---
    print(f"\n{'='*90}")
    print(f" 消融实验结果汇总 (5-Fold CV, Mean ± Std)")
    print(f"{'='*90}")
    header = f"{'Variant':<35s} {'AUROC':<18s} {'AUPRC':<18s} {'F1':<18s} {'#Feat':<6s}"
    print(header)
    print("-" * 90)
    for _, row in summary_df.iterrows():
        print(f"{row['Variant']:<35s} {row['AUROC']:<18s} {row['AUPRC']:<18s} "
              f"{row['F1-Score']:<18s} {row['Features']:<6}")

    # --- 绘制对比图 ---
    plot_ablation_results(summary_df)
    plot_feature_group_comparison(summary_df)

    print(f"\n  所有结果已保存至: {RESULTS_DIR}/")
    print(f"    ablation_summary.csv     — 汇总表")
    print(f"    ablation_all_folds.csv   — 每折详情")
    print(f"    ablation_comparison.png  — 对比图")
    print(f"    feature_group_impact.png — 特征组影响")
    print(f"{'='*70}")


def plot_ablation_results(summary_df):
    """绘制消融结果对比柱状图"""
    metrics = ['AUROC', 'AUPRC', 'F1-Score']
    variants = summary_df['Variant'].values
    n = len(variants)

    fig, axes = plt.subplots(1, 3, figsize=(18, max(5, n * 0.4)))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, n))

    for ax, metric in zip(axes, metrics):
        means = summary_df[f'{metric}_mean'].values
        stds = summary_df[f'{metric}_std'].values

        y_pos = range(n)
        ax.barh(y_pos, means, xerr=stds, color=colors, edgecolor='black', alpha=0.85)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(variants, fontsize=8)
        ax.set_xlabel(metric)
        ax.set_title(f'{metric} (5-Fold CV)')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)

        # 标数值
        for i, (m, s) in enumerate(zip(means, stds)):
            ax.text(m + 0.002, i, f'{m:.4f}', va='center', fontsize=7)

    plt.suptitle('MiRAGE Ablation Study — Feature Group Impact',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'ablation_comparison.png'), dpi=150,
                bbox_inches='tight')
    plt.close()
    print("  对比图已保存。")


def plot_feature_group_comparison(summary_df):
    """绘制特征组消融对比 (分组条形图)"""
    # 选几个关键变体做分组对比
    key_variants = [
        'Full Model (MiRAGE)',
        'w/o GiGs (No Graph)',
        'w/o Cross-Multiplication',
        'w/o Count Features',
        'Count + GiGs Only',
        'Similarity Only (no Count, no GiGs)',
    ]
    # 过滤存在的
    key_variants = [v for v in key_variants if v in summary_df['Variant'].values]

    metrics = ['AUROC', 'AUPRC', 'F1-Score']
    metric_means = {m: [] for m in metrics}
    metric_stds = {m: [] for m in metrics}

    for v in key_variants:
        row = summary_df[summary_df['Variant'] == v].iloc[0]
        for m in metrics:
            metric_means[m].append(row[f'{m}_mean'])
            metric_stds[m].append(row[f'{m}_std'])

    x = np.arange(len(key_variants))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (m, c) in enumerate(zip(metrics, ['#2196F3', '#FF9800', '#4CAF50'])):
        offset = (i - 1) * width
        ax.bar(x + offset, metric_means[m], width, yerr=metric_stds[m],
               label=m, color=c, edgecolor='black', alpha=0.85, capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels(key_variants, rotation=15, ha='right', fontsize=9)
    ax.set_ylabel('Score')
    ax.set_title('MiRAGE Ablation: Key Variants Comparison (5-Fold CV)')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0.0, 1.02)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'feature_group_impact.png'), dpi=150,
                bbox_inches='tight')
    plt.close()
    print("  特征组影响图已保存。")


if __name__ == "__main__":
    main()
