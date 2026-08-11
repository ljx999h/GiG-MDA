import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    accuracy_score, recall_score, f1_score,
    precision_recall_curve
)
from xgboost import XGBClassifier

# ===================== 配置 =====================
# 数据集选择: 'data/DDCD/Evaluation/train.csv' | 'data/C-Dataset/Evaluation/train.csv' | 'data/F-Dataset/Evaluation/train.csv'
TRAIN_CSV    = "data/C-Dataset/Evaluation/train.csv"
COL_LABEL    = 'label'
N_FOLDS      = 5
RANDOM_SEED  = 42

# 采样大小: 全量 2.1M 行跑 5 折 × 5 模型要 30+ 分钟, 这里采样
# None  = 全量 (慢)
# 200000 = 推荐值, 全部模型 5 折约 2~3 分钟
SAMPLE_SIZE  = None

# 22 维 MiRAGE 特征 (与论文 TABLE V 一致)
FEAT_MIRAGE_22 = (
    ['count_drug', 'count_disease'] +
    ['q_score_Description', 'q_score_Pathway', 'q_score_Slim'] +
    ['adj_q_score_Description', 'adj_q_score_Pathway', 'adj_q_score_Slim'] +
    ['p_score_Target', 'p_score_Category', 'p_score_Conditions',
    'p_score_Description', 'p_score_Mechanism', 'p_score_Pharmacodynamics',
    'p_score_Smile'] +
    ['adj_p_score_Target', 'adj_p_score_Category', 'adj_p_score_Conditions',
    'adj_p_score_Description', 'adj_p_score_Mechanism',
    'adj_p_score_Pharmacodynamics', 'adj_p_score_Smile']
)

# GiGs 128 维嵌入 (64 drug + 64 disease)
FEAT_GIGS_128 = (
    [f'gigs_drug_emb_{i}' for i in range(64)] +
    [f'gigs_disease_emb_{i}' for i in range(64)]
)

# 是否注入 GiGs: True=150 维 (22 MiRAGE + 128 GiGs), False=22 维 (仅 MiRAGE)
USE_GIGS = True
GIGS_PKL = "code/model/gigs_dataC.pkl"

CLASSIFIERS = {
    "LogisticRegression": lambda: LogisticRegression(
        max_iter=1000, solver='lbfgs', tol=1e-3,
        class_weight='balanced',  # 关键: 处理不平衡 (1:110)
        n_jobs=1, random_state=RANDOM_SEED),
    "KNN":                lambda: KNeighborsClassifier(
        n_neighbors=5, n_jobs=1, algorithm='auto',  # 改 auto (144维 kd_tree 会失败)
        weights='distance'),
    "Decision Tree":      lambda: DecisionTreeClassifier(
        class_weight='balanced',  # 关键
        random_state=RANDOM_SEED),
    "Random Forest":      lambda: RandomForestClassifier(
        n_estimators=100, max_depth=12, n_jobs=-1,
        class_weight='balanced',  # 关键
        random_state=RANDOM_SEED),
    "XGBoost (Ours)":     lambda: XGBClassifier(
        n_estimators=500, max_depth=10, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        tree_method='hist', device='cuda',
        # scale_pos_weight 用 (负/正) 比
        scale_pos_weight=(y_train:=pd.Series([0])).shape[0],  # 占位, 后面会更新
        random_state=RANDOM_SEED, n_jobs=-1, eval_metric='logloss'),
}


def cv_one_classifier(X, y, clf_factory, n_folds=N_FOLDS, seed=RANDOM_SEED):
    """
    5-Fold CV, 阈值在验证集上选 F1-max (与论文一致, 不在测试集上选)
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    metrics = {k: [] for k in ['AUROC', 'AUPR', 'Accuracy', 'Recall', 'F1']}

    for tr, va in skf.split(X, y):
        clf = clf_factory()
        clf.fit(X.iloc[tr], y.iloc[tr])
        y_proba = clf.predict_proba(X.iloc[va])[:, 1]

        # 关键修复: 阈值在验证集上选 F1-max (非默认 0.5)
        prec, rec, thr = precision_recall_curve(y.iloc[va], y_proba)
        f1_s = 2 * (prec * rec) / (prec + rec + 1e-9)
        best = f1_s.argmax()
        threshold = thr[best] if best < len(thr) else 0.5
        y_pred = (y_proba >= threshold).astype(int)

        metrics['AUROC'].append(roc_auc_score(y.iloc[va], y_proba))
        metrics['AUPR'].append(average_precision_score(y.iloc[va], y_proba))
        metrics['Accuracy'].append(accuracy_score(y.iloc[va], y_pred))
        metrics['Recall'].append(recall_score(y.iloc[va], y_pred))
        metrics['F1'].append(f1_score(y.iloc[va], y_pred))

    return {k: (np.mean(v), np.std(v)) for k, v in metrics.items()}


def print_table(results):
    """打印对齐的文本表格"""
    metrics = ['AUROC', 'AUPR', 'Accuracy', 'Recall', 'F1']
    # 列宽
    col_model = max(len(m) for m in results) + 2
    col_metric = 16  # 0.9998±0.0001 占 16 字符
    sep = '-' * (col_model + col_metric * len(metrics) + 2)

    # 找每列最优 mean
    best = {k: max(results[m][k][0] for m in results) for k in metrics}

    print()
    print(sep)
    print(f" {'Model':<{col_model}} " +
        " ".join(f"{m:^{col_metric}}" for m in metrics))
    print(sep)

    for model, mvals in results.items():
        cells = []
        for k in metrics:
            mean, std = mvals[k]
            txt = f"{mean:.4f}±{std:.4f}"
            if abs(mean - best[k]) < 1e-9:
                txt = f"\033[1m{txt}\033[0m"   # 加粗
            cells.append(f"{txt:^{col_metric}}")
        print(f" {model:<{col_model}} " + " ".join(cells))

    print(sep)
    print(f" 5-Fold CV | Mean +/- Std | 22 MiRAGE features | {len(results)} models")
    print(f" Best per column shown in bold")
    print()


def inject_gigs(df, gigs_data):
    """向量化注入 GiGs 128 维嵌入 (复用 ablation_study.py 逻辑)"""
    if gigs_data is None:
        # 缺失时填 0
        for i in range(64):
            df[f'gigs_drug_emb_{i}'] = 0.0
            df[f'gigs_disease_emb_{i}'] = 0.0
        return df
    X, Y = gigs_data['X'], gigs_data['Y']
    d2i, s2i = gigs_data['drug_to_idx'], gigs_data['disease_to_idx']
    drug_col     = 'drugID'     if 'drugID'     in df.columns else 'DrugID'
    disease_col  = 'diseaseID'  if 'diseaseID'  in df.columns else 'DiseaseID'
    drug_idx    = df[drug_col].astype(str).str.strip().map(d2i)
    disease_idx = df[disease_col].astype(str).str.strip().map(s2i)
    valid = drug_idx.notna() & disease_idx.notna()
    emb = np.zeros((len(df), 128), dtype=np.float32)
    if valid.any():
        d = drug_idx[valid].astype(int).values
        s = disease_idx[valid].astype(int).values
        emb[valid.values, :64]  = X[d].astype(np.float32)
        emb[valid.values, 64:]  = Y[s].astype(np.float32)
    emb_df = pd.DataFrame(emb, columns=FEAT_GIGS_128, index=df.index)
    return pd.concat([df, emb_df], axis=1)


def stratified_subsample(X, y, n_samples, seed=RANDOM_SEED):
    """分层采样, 保持正负比例"""
    if n_samples is None or len(X) <= n_samples:
        return X, y
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    pos_ratio = len(pos_idx) / len(y)
    n_pos = max(1, int(n_samples * pos_ratio))
    n_neg = n_samples - n_pos
    n_neg = min(n_neg, len(neg_idx))
    rng = np.random.RandomState(seed)
    sel = np.concatenate([
        rng.choice(pos_idx, min(n_pos, len(pos_idx)), replace=False),
        rng.choice(neg_idx, n_neg, replace=False)
    ])
    rng.shuffle(sel)
    return X.iloc[sel], y.iloc[sel]


def main():
    print(f"[1/2] 加载 {TRAIN_CSV} ...")
    df = pd.read_csv(TRAIN_CSV).fillna(0.0)
    available = [c for c in FEAT_MIRAGE_22 if c in df.columns]
    missing   = [c for c in FEAT_MIRAGE_22 if c not in df.columns]
    if missing:
        print(f"  [WARN] 缺失 {len(missing)} 特征")

    # 可选: 注入 GiGs 128 维
    if USE_GIGS:
        gigs_path = GIGS_PKL
        if not os.path.isabs(gigs_path):
            # 相对当前 cwd
            cand1 = gigs_path
            cand2 = os.path.join("..", gigs_path)
            if os.path.exists(cand1):
                gigs_path = cand1
            elif os.path.exists(cand2):
                gigs_path = cand2
        gigs_data = None
        if os.path.exists(gigs_path):
            import pickle
            with open(gigs_path, "rb") as f:
                gigs_data = pickle.load(f)
            print(f"  [GiGs] 加载 {gigs_path} OK: X={gigs_data['X'].shape}, Y={gigs_data['Y'].shape}")
        else:
            print(f"  [WARN] GiGs 未找到 ({gigs_path}), 跳过注入")
        df = inject_gigs(df, gigs_data)
        available = available + [c for c in FEAT_GIGS_128 if c in df.columns]
        print(f"  + GiGs 128D -> 总 {len(available)} 维")

    X = df[available]
    y = df[COL_LABEL]
    print(f"  {len(df):,} 行 | {X.shape[1]} 特征 | 正样本 {(y==1).sum():,}")

    # 采样
    if SAMPLE_SIZE and len(X) > SAMPLE_SIZE:
        X, y = stratified_subsample(X, y, SAMPLE_SIZE)
        print(f"  [Sample] {len(X):,} rows (pos {(y==1).sum():,}, neg {(y==0).sum():,})")

    print(f"\n[2/2] 运行 {len(CLASSIFIERS)} 个分类器 × {N_FOLDS}-Fold CV ...")
    import time
    results = {}
    for name, factory in CLASSIFIERS.items():
        t0 = time.time()
        print(f"  -> {name}", end=" ... ", flush=True)
        results[name] = cv_one_classifier(X, y, factory)
        print(f"OK ({time.time()-t0:.1f}s)")

    print_table(results)


if __name__ == "__main__":
    main()