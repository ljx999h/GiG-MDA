import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, classification_report,
    confusion_matrix, accuracy_score, f1_score, precision_score, recall_score,
    precision_recall_curve
)

# 论文原文: RandomForest(n_estimators=100, max_depth=12)
# 与 GIG-MDA 论文 (Sec II.E) 和 MiRAGE 论文 (Tbl 1) 严格一致
# 扩展: 最终模型可用 XGBoost (用于特征重要性 + 实验对比)
FINAL_CLASSIFIER = 'xgb'  # 'rf' (论文) | 'xgb' (扩展)
import xgboost as xgb
XGB_N = 500
XGB_D = 10

# ============================================================
# 配置区域
# ============================================================

# DDCD 训练集 (用于五折交叉验证)
TRAIN_CSV = "data/DDCD/Evaluation/train.csv"
# DDCD 独立测试集 (五折 CV 之外的最终留出验证)
TEST_CSV  = "data/DDCD/Evaluation/test.csv"

# GiGs 预训练文件
GIGS_PKL = "code/model/gigs_dataDDCD.pkl"

# 列名配置
COL_DRUG_ID = 'drugID'
COL_DISEASE_ID = 'diseaseID'
COL_LABEL = 'label'

# 五折交叉验证折数
N_FOLDS = 5

# 随机种子 (保证可复现)
RANDOM_SEED = 42

# 结果保存目录
RESULTS_DIR = 'results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================
# DDCD 实际特征列表 (匹配 train.csv 列名)
# drugID,diseaseID,count_drug,count_disease,
# q_score_Description,q_score_Pathway,q_score_Slim,p_score_Target,
# p_score_Category,p_score_Conditions,p_score_Description,p_score_Mechanism,
# p_score_Pharmacodynamics,p_score_Smile,adj_q_score_Description,
# adj_q_score_Pathway,adj_q_score_Slim,adj_p_score_Target,adj_p_score_Category,
# adj_p_score_Conditions,adj_p_score_Description,adj_p_score_Mechanism,
# adj_p_score_Pharmacodynamics,adj_p_score_Smile,label,numPreds
# ============================================================
DDCD_FEATURES = [
    'count_drug', 'count_disease',
    'q_score_Description', 'q_score_Pathway', 'q_score_Slim',
    'p_score_Target', 'p_score_Category', 'p_score_Conditions',
    'p_score_Description', 'p_score_Mechanism', 'p_score_Pharmacodynamics', 'p_score_Smile',
    'adj_q_score_Description', 'adj_q_score_Pathway', 'adj_q_score_Slim',
    'adj_p_score_Target', 'adj_p_score_Category', 'adj_p_score_Conditions',
    'adj_p_score_Description', 'adj_p_score_Mechanism', 'adj_p_score_Pharmacodynamics',
    'adj_p_score_Smile',
]

# 汇总所有特征
ALL_CANDIDATE_FEATURES = DDCD_FEATURES

# GiGs 注入策略: 'dot' (1维, 论文原版) | 'embed' (128维, 你的改进)
GIGS_INJECT_MODE = 'embed'  # 默认改为 128 维嵌入


# ============================================================
# 辅助函数
# ============================================================

def inject_gigs_features(df, gigs_data, mode='embed'):
    """
    注入 GiGs 图特征

    mode='dot'  : 1 维点积分数 (论文原版)
    mode='embed': 128 维嵌入 (drug 64维 + disease 64维 拼接)
    """
    if gigs_data is None:
        if mode == 'dot':
            df['score_gigs'] = 0.0
        else:
            k = 64
            for i in range(k):
                df[f'gigs_drug_emb_{i}'] = 0.0
                df[f'gigs_disease_emb_{i}'] = 0.0
        return df

    X_emb = gigs_data["X"]    # (n_drugs, 64)
    Y_emb = gigs_data["Y"]    # (n_diseases, 64)
    d2i = gigs_data["drug_to_idx"]
    s2i = gigs_data["disease_to_idx"]
    k = X_emb.shape[1]

    # 识别 ID 列
    drug_col = 'drugID' if 'drugID' in df.columns else 'DrugID'
    disease_col = 'diseaseID' if 'diseaseID' in df.columns else 'DiseaseID'

    drug_ids = df[drug_col].astype(str).str.strip()
    disease_ids = df[disease_col].astype(str).str.strip()

    # 映射为索引
    d_idx_arr = np.array([d2i.get(d, -1) for d in drug_ids], dtype=np.int32)
    s_idx_arr = np.array([s2i.get(s, -1) for s in disease_ids], dtype=np.int32)
    valid_mask = (d_idx_arr >= 0) & (s_idx_arr >= 0)

    n = len(df)

    if mode == 'dot':
        df['score_gigs'] = 0.0
        if valid_mask.any():
            scores = np.zeros(n, dtype=np.float32)
            scores[valid_mask] = np.sum(
                X_emb[d_idx_arr[valid_mask]] * Y_emb[s_idx_arr[valid_mask]],
                axis=1
            )
            df['score_gigs'] = scores
    else:
        # 128 维嵌入 (drug ⊕ disease)
        drug_feats = np.zeros((n, k), dtype=np.float32)
        disease_feats = np.zeros((n, k), dtype=np.float32)
        if valid_mask.any():
            drug_feats[valid_mask] = X_emb[d_idx_arr[valid_mask]]
            disease_feats[valid_mask] = Y_emb[s_idx_arr[valid_mask]]

        for i in range(k):
            df[f'gigs_drug_emb_{i}'] = drug_feats[:, i]
            df[f'gigs_disease_emb_{i}'] = disease_feats[:, i]

    return df


def load_and_inject_gigs(filename, gigs_data, feature_cols):
    """
    加载 CSV, 注入 GiGs 图特征 (dot 或 embed), 返回 X, y, 最终特征列名
    """
    if not os.path.exists(filename):
        print(f"[ERROR] 找不到文件 {filename}")
        return None, None, None

    df = pd.read_csv(filename)
    df = df.fillna(0)
    print(f"  [LOAD] {os.path.basename(filename)}: {len(df):,} 行")

    # --- 注入 GiGs 图特征 ---
    if gigs_data:
        if GIGS_INJECT_MODE == 'dot':
            print("  [GiGs] 注入 1 维点积 (论文原版)...")
        else:
            print(f"  [GiGs] 注入 128 维嵌入 (drug⊕disease)...")
        df = inject_gigs_features(df, gigs_data, mode=GIGS_INJECT_MODE)
    else:
        df = inject_gigs_features(df, None, mode=GIGS_INJECT_MODE)

    # --- 确定最终可用特征 ---
    if GIGS_INJECT_MODE == 'dot':
        gigs_cols = ['score_gigs']
    else:
        gigs_cols = [f'gigs_drug_emb_{i}' for i in range(64)] + \
                    [f'gigs_disease_emb_{i}' for i in range(64)]

    final_features = feature_cols + gigs_cols

    # 过滤掉 CSV 中不存在的列
    missing = [c for c in final_features if c not in df.columns]
    if missing:
        print(f"  [WARN] 缺失列 ({len(missing)}): {missing[:5]}...")
        final_features = [c for c in final_features if c in df.columns]

    # 过滤掉 csv 中的 numPreds (信息泄露特征)
    final_features = [c for c in final_features if c != 'numPreds']

    print(f"  [INFO] 最终使用 {len(final_features)} 个特征")

    X = df[final_features]
    y = df[COL_LABEL]

    # 统计标签分布
    n_pos = (y == 1).sum()
    n_neg = (y == 0).sum()
    print(f"  [INFO] 正样本: {n_pos:,} ({n_pos/len(y)*100:.1f}%) | 负样本: {n_neg:,} ({n_neg/len(y)*100:.1f}%)")

    return X, y, final_features


# ============================================================
# 五折交叉验证核心逻辑
# ============================================================

def run_cross_validation(X, y, feature_names, n_folds=N_FOLDS):
    """
    执行分层五折交叉验证
    返回: cv_results (list of dict), cv_predictions (DataFrame)
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)

    cv_results = []
    all_fold_preds = []  # 收集每折的预测结果

    print(f"\n{'='*60}")
    print(f" 开始 {n_folds}-Fold 分层交叉验证 (Stratified Cross-Validation)")
    print(f" 样本总数: {len(X):,} | 特征数: {len(feature_names)}")
    print(f"{'='*60}")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        print(f"\n{'─'*50}")
        print(f"  >>> Fold {fold}/{n_folds} <<<")
        print(f"{'─'*50}")

        # 划分数据
        X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

        print(f"  训练集: {len(X_train_fold):,} | 验证集: {len(X_val_fold):,}")
        print(f"  验证集正样本: {(y_val_fold==1).sum():,} | 负样本: {(y_val_fold==0).sum():,}")

        # 训练模型 (论文: RandomForest n=100, d=12)
        # [说明] 本函数生成 RandomForest baseline 的 5 折 CV 结果;
        #        论文中 XGBoost 的 5 折 CV 结果见 clf_comparison.py / ablation_study.py
        print(f"  正在训练 RandomForest (n_estimators=100, max_depth=12)...")
        clf = RandomForestClassifier(
            n_estimators=100, max_depth=12,
            n_jobs=-1, random_state=RANDOM_SEED
        )
        clf.fit(X_train_fold, y_train_fold)
        print(f"  训练完成。")

        # 预测概率
        y_prob_fold = clf.predict_proba(X_val_fold)[:, 1]

        # 自动寻找最佳 F1 阈值
        precisions, recalls, thresholds = precision_recall_curve(y_val_fold, y_prob_fold)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)

        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
        best_f1_at_thresh = f1_scores[best_idx]

        # 二值化预测
        y_pred_fold = (y_prob_fold >= best_threshold).astype(int)

        # 计算所有指标
        auroc = roc_auc_score(y_val_fold, y_prob_fold)
        auprc = average_precision_score(y_val_fold, y_prob_fold)
        acc = accuracy_score(y_val_fold, y_pred_fold)
        f1 = f1_score(y_val_fold, y_pred_fold)
        prec = precision_score(y_val_fold, y_pred_fold, zero_division=0)
        rec = recall_score(y_val_fold, y_pred_fold, zero_division=0)

        # 记录结果
        fold_result = {
            'Fold': fold,
            'Threshold': round(best_threshold, 4),
            'AUROC': round(auroc, 4),
            'AUPRC': round(auprc, 4),
            'Accuracy': round(acc, 4),
            'Precision': round(prec, 4),
            'Recall': round(rec, 4),
            'F1-Score': round(f1, 4),
        }
        cv_results.append(fold_result)

        print(f"  ───────────────────────────────")
        print(f"  最佳阈值: {best_threshold:.4f} (F1@thresh={best_f1_at_thresh:.4f})")
        print(f"  AUROC : {auroc:.4f}")
        print(f"  AUPRC : {auprc:.4f}")
        print(f"  Acc   : {acc:.4f}  |  Prec  : {prec:.4f}")
        print(f"  Recall: {rec:.4f}  |  F1    : {f1:.4f}")

        # 保存该折预测结果 (带 fold 标记)
        fold_pred_df = pd.DataFrame({
            'Fold': fold,
            'True_Label': y_val_fold.values,
            'Probability': y_prob_fold,
            'Predicted_Label': y_pred_fold,
        })
        all_fold_preds.append(fold_pred_df)

    # --- 汇总五折结果 ---
    cv_df = pd.DataFrame(cv_results)

    # 计算均值 ± 标准差
    metric_names = ['AUROC', 'AUPRC', 'Accuracy', 'Precision', 'Recall', 'F1-Score']
    summary = {}
    for m in metric_names:
        mean_val = cv_df[m].mean()
        std_val = cv_df[m].std()
        summary[m] = f"{mean_val:.4f} ± {std_val:.4f}"

    print(f"\n{'='*60}")
    print(f" 五折交叉验证汇总结果 (Mean ± Std)")
    print(f"{'='*60}")
    for m, v in summary.items():
        print(f"  {m:<12s}: {v}")
    print(f"{'='*60}")

    # 合并所有折的预测
    cv_predictions = pd.concat(all_fold_preds, ignore_index=True)

    return cv_df, cv_predictions, summary


# ============================================================
# 独立测试集最终评估
# ============================================================

def final_evaluate_on_test(X_train_full, y_train_full, X_test, y_test, cv_threshold=None):
    """
    使用全量训练集训练最终模型, 在独立测试集上评估

    [关键修复] cv_threshold: 使用五折 CV 的平均阈值，而非在测试集上调优
    AUROC/AUPRC 是阈值无关指标，不受影响
    F1/Accuracy/Recall 使用 CV 阈值，避免测试集调参的乐观偏差
    """
    print(f"\n{'='*60}")
    print(f" 独立测试集最终评估 (Held-out Test Set)")
    print(f" 全量训练: {len(X_train_full):,} 样本 | 独立测试: {len(X_test):,} 样本")
    print(f"{'='*60}")

    # 训练 (论文: RandomForest | 扩展: XGBoost GPU)
    if FINAL_CLASSIFIER == 'xgb':
        spw = (y_train_full == 0).sum() / max(1, (y_train_full == 1).sum())
        print(f"  全量训练 XGBoost (n={XGB_N}, d={XGB_D}, scale_pos_weight={spw:.1f})...")
        clf_final = xgb.XGBClassifier(
            n_estimators=XGB_N, max_depth=XGB_D, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=spw,
            tree_method='hist', device='cuda',
            random_state=RANDOM_SEED, verbosity=0
        )
    else:
        print(f"  全量训练 RandomForest (n_estimators=100, max_depth=12)...")
        clf_final = RandomForestClassifier(
            n_estimators=100, max_depth=12,
            n_jobs=-1, random_state=RANDOM_SEED
        )
    clf_final.fit(X_train_full, y_train_full)
    print("  全量模型训练完成。")

    # 预测
    X_test = X_test[X_train_full.columns]  # 确保列顺序一致
    y_prob = clf_final.predict_proba(X_test)[:, 1]

    # [关键修复] 使用 CV 平均阈值，不在测试集上调优
    if cv_threshold is not None:
        best_threshold = cv_threshold
        print(f"  [INFO] 使用 CV 平均阈值: {best_threshold:.4f} (非测试集调优)")
    else:
        # Fallback: 如果未提供 CV 阈值，使用 0.5
        best_threshold = 0.5
        print(f"  [WARN] 未提供 CV 阈值，使用默认值 0.5")

    y_pred = (y_prob >= best_threshold).astype(int)

    # 计算指标
    auroc = roc_auc_score(y_test, y_prob)
    auprc = average_precision_score(y_test, y_prob)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)

    print(f"\n  >>> 最终测试集结果 (阈值 = CV均值 {best_threshold:.4f}) <<<")
    print(f"  AUROC     : {auroc:.4f}")
    print(f"  AUPRC     : {auprc:.4f}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")

    # 同时报告"如果使用测试集调优阈值"的结果作为参考对比
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
    best_idx_test = np.argmax(f1_scores)
    best_threshold_test = thresholds[best_idx_test] if best_idx_test < len(thresholds) else 0.5
    y_pred_test_opt = (y_prob >= best_threshold_test).astype(int)
    f1_test_opt = f1_score(y_test, y_pred_test_opt)

    print(f"\n  [参考] 若在测试集上调阈值 (不推荐，仅对比):")
    print(f"    测试集最优阈值: {best_threshold_test:.4f}")
    print(f"    测试集最优 F1  : {f1_test_opt:.4f}")

    # Classification Report (使用 CV 阈值)
    print(f"\n  Classification Report (threshold = {best_threshold:.4f}):")
    print(classification_report(y_test, y_pred, digits=4))

    return clf_final, y_prob, y_pred, best_threshold, {
        'AUROC': auroc, 'AUPRC': auprc, 'Accuracy': acc,
        'Precision': prec, 'Recall': rec, 'F1-Score': f1,
        'Threshold': best_threshold
    }


# ============================================================
# 可视化
# ============================================================

def plot_cv_results(cv_df, save_dir=RESULTS_DIR):
    """绘制五折 CV 指标分布图"""
    metric_names = ['AUROC', 'AUPRC', 'Accuracy', 'Precision', 'Recall', 'F1-Score']

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for i, m in enumerate(metric_names):
        ax = axes[i]
        values = cv_df[m].values
        folds = cv_df['Fold'].values

        # 柱状图: 每折的值
        bars = ax.bar(folds, values, color='steelblue', alpha=0.8, edgecolor='black')
        # 均值线
        mean_val = values.mean()
        ax.axhline(y=mean_val, color='red', linestyle='--', linewidth=2,
                   label=f'Mean = {mean_val:.4f}')

        # 标注数值
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{val:.4f}', ha='center', va='bottom', fontsize=9)

        ax.set_title(f'{m} (per Fold)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Fold')
        ax.set_ylabel(m)
        ax.set_ylim(max(0, values.min() - 0.05), min(1.0, values.max() + 0.08))
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle(f'{N_FOLDS}-Fold Cross-Validation Results on DDCD',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'cv_fold_metrics.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\n  五折 CV 指标图已保存至: {save_path}")


def plot_confusion_matrix(y_test, y_pred, threshold, save_dir=RESULTS_DIR):
    """混淆矩阵"""
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix (Test Set, Thresh={threshold:.2f})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    save_path = os.path.join(save_dir, 'confusion_matrix_test.png')
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  混淆矩阵已保存至: {save_path}")


def plot_feature_importance(model, feature_names, save_dir=RESULTS_DIR):
    """特征重要性 Top-20 (GiGs 嵌入列合并显示为 'GiGs_Embed_128D')"""
    importances = model.feature_importances_
    feat_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

    # 合并 GiGs 128 维嵌入为一条 "GiGs Embedding (128D)"
    gigs_mask = feat_imp_df['Feature'].str.contains('gigs_drug_emb|gigs_disease_emb', regex=True)
    gigs_total = feat_imp_df.loc[gigs_mask, 'Importance'].sum()
    feat_imp_df = feat_imp_df[~gigs_mask]
    feat_imp_df = pd.concat([
        feat_imp_df,
        pd.DataFrame([{'Feature': 'GiGs Embedding (128D)', 'Importance': gigs_total}])
    ], ignore_index=True)

    feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(feat_imp_df['Feature'][::-1], feat_imp_df['Importance'][::-1],
                   color=plt.cm.viridis(np.linspace(0.2, 0.9, 20)))
    ax.set_xlabel('Importance (Gini decrease)', fontsize=11)
    ax.set_title('Feature Importances (XGBoost, Top-20, DDCD)', fontsize=13, fontweight='bold')
    # 标注数值
    for bar, val in zip(bars, feat_imp_df['Importance'][::-1]):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=8)
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'feature_importance.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  特征重要性图已保存至: {save_path}")

    # 打印 GiGs 总排名
    print(f"  >>> GiGs Embedding (128D) 总重要性 = {gigs_total:.4f}, "
          f"Top-5: {feat_imp_df['Feature'].head(5).tolist()}")


# ============================================================
# 主函数
# ============================================================

def main():
    print("=" * 60)
    print(" MiRAGE + GiGs: DDCD 五折交叉验证 & 独立测试评估")
    print("=" * 60)
    print(f" 训练集: {TRAIN_CSV}")
    print(f" 测试集: {TEST_CSV}")
    print(f" GiGs  : {GIGS_PKL}")
    print(f" CV折数: {N_FOLDS}")
    print("=" * 60)

    # ----------------------------------------------------------
    # Step 1: 加载 GiGs 预训练数据
    # ----------------------------------------------------------
    print(f"\n[Step 1] 加载 GiGs 预训练数据...")
    gigs_data = None
    if os.path.exists(GIGS_PKL):
        with open(GIGS_PKL, "rb") as f:
            gigs_data = pickle.load(f)
        print(f"  GiGs 数据加载成功! (X: {gigs_data['X'].shape}, Y: {gigs_data['Y'].shape})")
        print(f"  Drugs: {len(gigs_data['drug_to_idx'])}, Diseases: {len(gigs_data['disease_to_idx'])}")
    else:
        print(f"  [WARN] 找不到 GiGs 文件: {GIGS_PKL}, 将不使用图特征!")

    # ----------------------------------------------------------
    # Step 2: 加载训练集
    # ----------------------------------------------------------
    print(f"\n[Step 2] 加载训练集 (用于五折 CV)...")
    X_train, y_train, train_feats = load_and_inject_gigs(TRAIN_CSV, gigs_data, ALL_CANDIDATE_FEATURES)
    if X_train is None:
        print("[FATAL] 训练集加载失败，退出。")
        return

    # ----------------------------------------------------------
    # Step 3: 五折交叉验证
    # ----------------------------------------------------------
    print(f"\n[Step 3] 执行 {N_FOLDS}-Fold 交叉验证...")
    cv_df, cv_predictions, cv_summary = run_cross_validation(X_train, y_train, train_feats, N_FOLDS)

    # 保存五折结果
    cv_csv_path = os.path.join(RESULTS_DIR, 'cv_5fold_metrics.csv')
    cv_df.to_csv(cv_csv_path, index=False)
    print(f"\n  五折 CV 指标表已保存至: {cv_csv_path}")

    cv_preds_path = os.path.join(RESULTS_DIR, 'cv_5fold_predictions.csv')
    cv_predictions.to_csv(cv_preds_path, index=False)
    print(f"  五折 CV 预测详情已保存至: {cv_preds_path}")

    # 绘制五折指标图
    plot_cv_results(cv_df)

    # ----------------------------------------------------------
    # Step 4: 加载独立测试集 (可选)
    # ----------------------------------------------------------
    print(f"\n[Step 4] 加载独立测试集...")
    X_test, y_test, test_feats = load_and_inject_gigs(TEST_CSV, gigs_data, ALL_CANDIDATE_FEATURES)

    # [关键修复] 从五折 CV 获取平均阈值，不在测试集上调优
    cv_avg_threshold = cv_df['Threshold'].mean()
    print(f"\n  [INFO] 五折 CV 平均阈值: {cv_avg_threshold:.4f} (将用于最终测试评估)")

    final_model = None
    if X_test is not None:
        # ----------------------------------------------------------
        # Step 5: 全量训练 + 独立测试
        # ----------------------------------------------------------
        print(f"\n[Step 5] 全量训练 & 独立测试评估...")
        final_model, y_prob_test, y_pred_test, best_thresh, test_metrics = \
            final_evaluate_on_test(X_train, y_train, X_test, y_test,
                                   cv_threshold=cv_avg_threshold)

        # 混淆矩阵
        plot_confusion_matrix(y_test, y_pred_test, best_thresh)

        # 特征重要性
        plot_feature_importance(final_model, X_train.columns)

        # 保存最终测试预测
        test_df_raw = pd.read_csv(TEST_CSV)
        results_df = pd.DataFrame({
            'DrugID': test_df_raw[COL_DRUG_ID],
            'DiseaseID': test_df_raw[COL_DISEASE_ID],
            'True_Label': y_test.values,
            'Probability': y_prob_test,
            'Predicted_Label': y_pred_test,
        })
        full_res_path = os.path.join(RESULTS_DIR, 'final_test_predictions.csv')
        results_df.to_csv(full_res_path, index=False)
        print(f"\n  全量测试预测已保存至: {full_res_path}")

        # --- Top 候选药物 (Case Study) ---
        print(f"\n[Step 6] 提取 Top 候选药物 (Novel Candidates)...")
        novel_candidates = results_df[
            (results_df['True_Label'] == 0) &
            (results_df['Probability'] > best_thresh)
        ].sort_values(by='Probability', ascending=False)

        top_candidates_path = os.path.join(RESULTS_DIR, 'top_novel_candidates.csv')
        novel_candidates.head(100).to_csv(top_candidates_path, index=False)
        print(f"  潜在新药候选 (Top 100) 已保存至: {top_candidates_path}")
        print(f"\n  Top 5 候选者预览:")
        print(novel_candidates[['DrugID', 'DiseaseID', 'Probability']].head(5))

    # ----------------------------------------------------------
    # Step 7: 最终汇总
    # ----------------------------------------------------------
    print(f"\n{'='*60}")
    print(f" 实验完成! 结果汇总")
    print(f"{'='*60}")
    print(f"\n  >>> 五折交叉验证 (Mean ± Std) <<<")
    for m, v in cv_summary.items():
        print(f"    {m:<12s}: {v}")

    if X_test is not None:
        print(f"\n  >>> 独立测试集 <<<")
        for k, v in test_metrics.items():
            if isinstance(v, float):
                print(f"    {k:<12s}: {v:.4f}")
            else:
                print(f"    {k:<12s}: {v}")

    print(f"\n 所有结果已保存至: {RESULTS_DIR}/")
    print(f"{'='*60}")

    # ----------------------------------------------------------
    # Step 7: 4 变体扩展实验 (1D/128D × RF/XGBoost)
    #   论文方法 (RF) 作 baseline, 你的改进 (128D+XGB) 作主贡献
    # ----------------------------------------------------------
    print(f"\n[Step 7] 4 变体扩展实验 (1D/128D × RF/XGBoost)...")

    if gigs_data is not None:
        # 注入两种 GiGs 表示
        X_train_4v = X_train.copy()  # 已经注入了 128D embed (因为 GIGS_INJECT_MODE='embed')
        # 同时算 1D dot 版本的列
        from eval_helpers_4variants import get_gigs_features, run_4_variants

        dot_feats, dot_cols = get_gigs_features(
            pd.read_csv(TRAIN_CSV).fillna(0.0)[['drugID', 'diseaseID']],
            gigs_data, mode='dot'
        )
        X_dot = X_train_4v.copy()
        for i, col in enumerate(dot_cols):
            X_dot[col] = dot_feats[:, i]
        # X_emb = X_train_4v (已有 128D)
        X_emb = X_train_4v

        summary_4v, all_metrics, lifts = run_4_variants(
            X_dot, X_emb, y_train, dataset_name="DDCD"
        )
        summary_4v.to_csv(os.path.join(RESULTS_DIR, '4variants_summary.csv'), index=False)
        # 保存每变体每折详情
        for name, m in all_metrics.items():
            safe = name.replace(' ', '_').replace('(', '').replace(')', '').replace('+', '_')
            m.to_csv(os.path.join(RESULTS_DIR, f'4v_{safe}.csv'), index=False)
        print(f"\n  4 变体结果已保存至: {RESULTS_DIR}/4v_*.csv")

    print(f"\n{'='*60}")
    print(f" 全部完成!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
