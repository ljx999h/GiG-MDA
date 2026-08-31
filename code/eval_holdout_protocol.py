"""
R2 阶段2: 方案B 评估协议 (固定独立测试)

流程:
  1. 训练域 (P_train 正 + out-of-fold 可靠负样本) 构建训练集 (Full Model: MiRAGE + GiGs)
  2. 训练域内 5-fold CV 选阈值 (参考值, 非最终报告)
  3. 全量训练 XGBoost (Full Model)
  4. 封存的 test (P_test + U_test, pair-disjoint) 只评估一次:
       AUROC / AUPRC + 95% bootstrap CI + F1 / Accuracy / Recall / Precision
  5. 输出 results/R2/{DS}_holdout_results.csv / .json

依赖 (已由 data_split + negative_mining_oof + pretrain_gigs_split 生成):
  - data/{DS}/Splits/train_positives.csv, train_negatives.csv
  - data/{DS}/Splits/gigs_split_{DS}.pkl
  - code/results/MiRAGE_score_{DS}.csv
"""
import argparse
import json
import os
import pickle
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    f1_score, recall_score, precision_score, precision_recall_curve
)
import xgboost as xgb

DATASETS = {
    'C': {
        'manifest': 'data/C-Dataset/Splits/split_manifest.csv',
        'score': 'code/results/MiRAGE_score_C.csv',
        'gigs': 'data/C-Dataset/Splits/gigs_split_C.pkl',
        'mapping': 'data/C-Dataset/Mapping/mapping_C.csv',
        'train_pos': 'data/C-Dataset/Splits/train_positives.csv',
        'train_neg': 'data/C-Dataset/Splits/train_negatives.csv',
        'out': 'results/R2/C_holdout',
    },
    'F': {
        'manifest': 'data/F-Dataset/Splits/split_manifest.csv',
        'score': 'code/results/MiRAGE_score_F.csv',
        'gigs': 'data/F-Dataset/Splits/gigs_split_F.pkl',
        'mapping': 'data/F-Dataset/Mapping/mapping_F.csv',
        'train_pos': 'data/F-Dataset/Splits/train_positives.csv',
        'train_neg': 'data/F-Dataset/Splits/train_negatives.csv',
        'out': 'results/R2/F_holdout',
    },
    'DDCD': {
        'manifest': 'data/DDCD/Splits/split_manifest.csv',
        'score': 'code/results/MiRAGE_score_DDCD.csv',
        'gigs': 'data/DDCD/Splits/gigs_split_DDCD.pkl',
        'mapping': 'data/DDCD/Mapping/mapping.csv',
        'train_pos': 'data/DDCD/Splits/train_positives.csv',
        'train_neg': 'data/DDCD/Splits/train_negatives.csv',
        'out': 'results/R2/DDCD_holdout',
    },
}

XGB_N = 500
XGB_D = 10
LR = 0.05


def bootstrap_ci(y_true, y_prob, n_boot=1000, seed=42, metric='aupr'):
    """bootstrap 95% CI."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    scores = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        yt, yp = y_true[idx], y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        if metric == 'auroc':
            scores.append(roc_auc_score(yt, yp))
        else:
            scores.append(average_precision_score(yt, yp))
    scores = np.array(scores)
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def inject_gigs(df, gigs_data):
    """注入 128 维 GiGs 嵌入 (字符串 ID 对齐)."""
    X_emb, Y_emb = gigs_data['X'], gigs_data['Y']
    d2i, s2i = gigs_data['drug_to_idx'], gigs_data['disease_to_idx']
    k = X_emb.shape[1]
    d_idx = df['drugID'].astype(str).str.strip().map(d2i)
    s_idx = df['diseaseID'].astype(str).str.strip().map(s2i)
    valid = d_idx.notna() & s_idx.notna()
    emb = np.zeros((len(df), 2 * k), dtype=np.float32)
    if valid.any():
        emb[valid.values, :k] = X_emb[d_idx[valid].astype(int).values]
        emb[valid.values, k:] = Y_emb[s_idx[valid].astype(int).values]
    cols = [f'gigs_drug_emb_{i}' for i in range(k)] + \
           [f'gigs_disease_emb_{i}' for i in range(k)]
    return pd.concat([df.reset_index(drop=True),
                      pd.DataFrame(emb, columns=cols)], axis=1), cols


def load_score(ds, cfg):
    df = pd.read_csv(cfg['score']).fillna(0.0)
    df['drugID'] = df['drugID'].astype(str).str.strip()
    df['diseaseID'] = df['diseaseID'].astype(str).str.strip()
    if 'label' in df.columns:
        df = df.drop(columns=['label'])
    # C: MiRAGE_score drugID 是整数索引, 映射回 DB 字符串
    if ds == 'C':
        mapping = pd.read_csv(cfg['mapping'])
        if 'Unnamed: 0' in mapping.columns:
            mapping = mapping.drop(columns=['Unnamed: 0'])
        drugbanks = sorted(mapping.iloc[:, 0].astype(str).str.strip().unique())
        idx_to_db = {i: db for i, db in enumerate(drugbanks)}
        df['drugID'] = df['drugID'].map(lambda x: idx_to_db.get(int(x), x))
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, choices=['C', 'F', 'DDCD'])
    parser.add_argument('--n-boot', type=int, default=1000)
    parser.add_argument('--random-state', type=int, default=42)
    args = parser.parse_args()
    cfg = DATASETS[args.dataset]

    print(f"\n{'='*60}\n 方案B 评估: {args.dataset}\n{'='*60}")

    # 1. 训练集 (P_train 正 + 可靠负)
    pos = pd.read_csv(cfg['train_pos']).fillna(0.0)
    neg = pd.read_csv(cfg['train_neg']).fillna(0.0)
    train = pd.concat([pos, neg], ignore_index=True)
    train['drugID'] = train['drugID'].astype(str).str.strip()
    train['diseaseID'] = train['diseaseID'].astype(str).str.strip()
    print(f"训练集: {len(train):,} (正{(train['label']==1).sum():,} 负{(train['label']==0).sum():,})")

    # 2. 测试候选 (manifest test 部分) + MiRAGE 特征
    mf = pd.read_csv(cfg['manifest'])
    mf['drugID'] = mf['drugID'].astype(str).str.strip()
    mf['diseaseID'] = mf['diseaseID'].astype(str).str.strip()
    test_mf = mf[mf['split'] == 'test'][['drugID', 'diseaseID', 'label']].copy()
    score = load_score(args.dataset, cfg)
    key = ['drugID', 'diseaseID']
    test = test_mf.merge(score, on=key, how='inner')
    print(f"测试集: {len(test):,} (正{(test['label']==1).sum():,} 负{(test['label']==0).sum():,})")

    # 3. 注入 GiGs
    with open(cfg['gigs'], 'rb') as f:
        gigs = pickle.load(f)
    train, gigs_cols = inject_gigs(train, gigs)
    test, _ = inject_gigs(test, gigs)

    # 4. 特征列
    feat_cols = [c for c in score.columns if c not in key] + gigs_cols
    Xtr = train[feat_cols].values.astype(np.float32)
    ytr = train['label'].values
    Xte = test[feat_cols].values.astype(np.float32)
    yte = test['label'].values
    spw = (ytr == 0).sum() / max(1, (ytr == 1).sum())
    print(f"特征维度: {Xtr.shape[1]} ({len(feat_cols)-len(gigs_cols)} MiRAGE + {len(gigs_cols)} GiGs)")

    # 5. 训练域内 5-fold CV 选阈值 (参考, 非最终报告)
    print("  训练域内 5-fold CV 选阈值 (参考)...")
    skf = StratifiedKFold(5, shuffle=True, random_state=args.random_state)
    threshs = []
    for tr, va in skf.split(Xtr, ytr):
        clf = xgb.XGBClassifier(n_estimators=XGB_N, max_depth=XGB_D, learning_rate=LR,
                                subsample=0.8, colsample_bytree=0.8,
                                scale_pos_weight=(ytr[tr] == 0).sum() / max(1, (ytr[tr] == 1).sum()),
                                tree_method='hist', device='cuda',
                                random_state=args.random_state, verbosity=0)
        clf.fit(Xtr[tr], ytr[tr])
        yv = clf.predict_proba(Xtr[va])[:, 1]
        p, r, t = precision_recall_curve(ytr[va], yv)
        f1s = 2 * p * r / (p + r + 1e-9)
        best = f1s.argmax()
        threshs.append(t[best] if best < len(t) else 0.5)
    avg_thresh = float(np.mean(threshs))
    print(f"  平均阈值: {avg_thresh:.4f}")

    # 6. 全量训练
    print("  全量训练 XGBoost (Full Model)...")
    clf = xgb.XGBClassifier(n_estimators=XGB_N, max_depth=XGB_D, learning_rate=LR,
                            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
                            tree_method='hist', device='cuda',
                            random_state=args.random_state, verbosity=0)
    clf.fit(Xtr, ytr)

    # 7. test 评估 (只一次)
    yprob = clf.predict_proba(Xte)[:, 1]
    auroc = roc_auc_score(yte, yprob)
    aupr = average_precision_score(yte, yprob)
    ypred = (yprob >= avg_thresh).astype(int)
    f1 = f1_score(yte, ypred)
    acc = accuracy_score(yte, ypred)
    rec = recall_score(yte, ypred)
    prec = precision_score(yte, ypred, zero_division=0)

    auroc_lo, auroc_hi = bootstrap_ci(yte, yprob, args.n_boot, args.random_state, 'auroc')
    aupr_lo, aupr_hi = bootstrap_ci(yte, yprob, args.n_boot, args.random_state, 'aupr')

    print(f"\n  >>> test 结果 (方案B, pair-disjoint):")
    print(f"  AUROC = {auroc:.5f} (95% CI [{auroc_lo:.5f}, {auroc_hi:.5f}])")
    print(f"  AUPR  = {aupr:.5f} (95% CI [{aupr_lo:.5f}, {aupr_hi:.5f}])")
    print(f"  F1={f1:.5f}  Acc={acc:.5f}  Recall={rec:.5f}  Precision={prec:.5f}  Thresh={avg_thresh:.4f}")

    # 8. 保存
    os.makedirs(cfg['out'], exist_ok=True)
    result = {
        'dataset': args.dataset,
        'n_train': int(len(train)), 'n_test': int(len(test)),
        'n_pos_test': int((yte == 1).sum()), 'n_neg_test': int((yte == 0).sum()),
        'AUROC': round(auroc, 6), 'AUROC_ci': [round(auroc_lo, 6), round(auroc_hi, 6)],
        'AUPR': round(aupr, 6), 'AUPR_ci': [round(aupr_lo, 6), round(aupr_hi, 6)],
        'F1': round(f1, 6), 'Accuracy': round(acc, 6),
        'Recall': round(rec, 6), 'Precision': round(prec, 6),
        'Threshold': round(avg_thresh, 4),
        'n_boot': args.n_boot,
    }
    with open(os.path.join(cfg['out'], 'result.json'), 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    pd.DataFrame([result]).to_csv(os.path.join(cfg['out'], 'result.csv'), index=False)
    print(f"\n  ✅ 结果已保存: {cfg['out']}/")


if __name__ == '__main__':
    main()
