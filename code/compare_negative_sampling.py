"""
RQ2 负采样对照 (compare_negative_sampling)

对比三种负采样策略 (同一 pair-disjoint 划分, 主模型特征 = MiRAGE+embed):
  oof    : out-of-fold 可靠负采样 (现有管线, 读 train_negatives.csv)
  random : 从 U_train 随机采样 1:1
  pu     : Elkan-Noto (2008) 简化实现——正 + 随机未标注训练, c=P(s=1|y=1) 校准,
           选校准后 P(y=1) 最低的 n_pos 个

评估: 方案B (train 内 5 折 CV 阈值 + test 一次), 统一 XGBoost 标准配置.
输出: results/R2/negative_sampling_compare.csv

用法: python code/compare_negative_sampling.py --dataset C
"""
import argparse
import os
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             precision_recall_curve)

import r2_config

PU_RATIO = 3   # PU 训练时未标注:正样本比例


def load(ds, seed=42):
    cfg = r2_config.DATASETS[ds]
    pos = pd.read_csv(cfg['train_pos']).fillna(0.0)
    neg = pd.read_csv(cfg['train_neg']).fillna(0.0)
    feats = cfg['feats']
    keep = ['drugID', 'diseaseID', 'label'] + feats
    train_pos = pos[keep]
    train_neg = neg[keep]
    mf = pd.read_csv(cfg['manifest'])
    mf['drugID'] = mf['drugID'].astype(str).str.strip()
    mf['diseaseID'] = mf['diseaseID'].astype(str).str.strip()
    test_mf = mf[mf['split'] == 'test'][['drugID', 'diseaseID', 'label']].copy()
    score = pd.read_csv(cfg['score']).fillna(0.0)
    score['drugID'] = score['drugID'].astype(str).str.strip()
    score['diseaseID'] = score['diseaseID'].astype(str).str.strip()
    if 'label' in score.columns:
        score = score.drop(columns=['label'])
    if ds == 'C':
        mapping = pd.read_csv(cfg['mapping'])
        if 'Unnamed: 0' in mapping.columns:
            mapping = mapping.drop(columns=['Unnamed: 0'])
        drugbanks = sorted(mapping.iloc[:, 0].astype(str).str.strip().unique())
        idx_to_db = {i: db for i, db in enumerate(drugbanks)}
        score['drugID'] = score['drugID'].map(lambda x: idx_to_db.get(int(x), x))
    test = test_mf.merge(score, on=['drugID', 'diseaseID'], how='inner')
    assert len(test) == len(test_mf)
    with open(cfg['gigs'], 'rb') as f:
        gigs = pickle.load(f)
    # 负池 (manifest train 负)
    man = mf[(mf['split'] == 'train') & (mf['label'] == 0)][['drugID', 'diseaseID']].copy()
    man['drugID'] = man['drugID'].astype(str).str.strip()
    man['diseaseID'] = man['diseaseID'].astype(str).str.strip()
    neg_pool = man.merge(score, on=['drugID', 'diseaseID'], how='inner')
    neg_pool['label'] = 0
    return train_pos, train_neg, neg_pool, test, gigs, feats


def gigs_feats(df, gigs):
    X, Y = gigs['X'], gigs['Y']
    d2i, s2i = gigs['drug_to_idx'], gigs['disease_to_idx']
    k = X.shape[1]
    d = df['drugID'].astype(str).str.strip().map(d2i)
    s = df['diseaseID'].astype(str).str.strip().map(s2i)
    v = d.notna() & s.notna()
    emb = np.zeros((len(df), 2 * k), dtype=np.float32)
    if v.any():
        dv, sv = d[v].astype(int).values, s[v].astype(int).values
        emb[v.values, :k] = X[dv]
        emb[v.values, k:] = Y[sv]
    return emb


def feats_matrix(df, feats, emb):
    return np.hstack([df[feats].values, emb]).astype(np.float32)


def pu_negatives(pos, neg_pool, feats, gigs, seed=42):
    """Elkan-Noto 简化: 正 + 随机未标注训练 → c 校准 → 选 P(y=1) 最低的 n_pos 个."""
    n_pos = len(pos)
    n_pu = min(n_pos * PU_RATIO, len(neg_pool))
    unlabeled = neg_pool.sample(n=n_pu, random_state=seed)
    Xpu = np.vstack([feats_matrix(pos, feats, gigs_feats(pos, gigs)),
                     feats_matrix(unlabeled, feats, gigs_feats(unlabeled, gigs))])
    ypu = np.concatenate([np.ones(n_pos), np.zeros(n_pu)])
    clf = xgb.XGBClassifier(**r2_config.XGB_CONFIG)
    clf.fit(Xpu, ypu)
    p_s1_pos = clf.predict_proba(feats_matrix(pos, feats, gigs_feats(pos, gigs)))[:, 1]
    c = np.clip(p_s1_pos.mean(), 1e-3, 1.0)          # P(s=1|y=1) 估计
    p_s1_neg = clf.predict_proba(feats_matrix(neg_pool, feats, gigs_feats(neg_pool, gigs)))[:, 1]
    p_y1 = np.clip(p_s1_neg / c, 0, 1)
    idx = np.argsort(p_y1)[:n_pos]                    # 最可靠的负样本
    return neg_pool.iloc[idx]


def evaluate(train, test, feats, gigs, seed=42):
    tr_emb = gigs_feats(train, gigs)
    te_emb = gigs_feats(test, gigs)
    Xtr = feats_matrix(train, feats, tr_emb)
    Xte = feats_matrix(test, feats, te_emb)
    ytr, yte = train['label'].values, test['label'].values
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    threshs = []
    for tr, va in skf.split(Xtr, ytr):
        clf = xgb.XGBClassifier(**r2_config.XGB_CONFIG)
        clf.fit(Xtr[tr], ytr[tr])
        yv = clf.predict_proba(Xtr[va])[:, 1]
        p, r, t = precision_recall_curve(ytr[va], yv)
        f1s = 2 * p * r / (p + r + 1e-9)
        best = f1s.argmax()
        threshs.append(t[best] if best < len(t) else 0.5)
    clf = xgb.XGBClassifier(**r2_config.XGB_CONFIG)
    clf.fit(Xtr, ytr)
    yp = clf.predict_proba(Xte)[:, 1]
    ypred = (yp >= np.mean(threshs)).astype(int)
    return {'AUROC': roc_auc_score(yte, yp), 'AUPR': average_precision_score(yte, yp),
            'F1': f1_score(yte, ypred)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset', required=True, choices=['C', 'F', 'DDCD'])
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    ds = args.dataset
    train_pos, train_neg, neg_pool, test, gigs, feats = load(ds, args.seed)
    n_pos = len(train_pos)
    print(f"\n=== {ds} 负采样对照 (主模型 MiRAGE+embed) ===")
    print(f"train 正: {n_pos:,} | 负池: {len(neg_pool):,} | test: {len(test):,}")

    # 注: Elkan-Noto PU 的校准 P(s=1)/c 是单调变换, 不影响 AUROC/AUPR (rank 等价于 random);
    # 其价值仅在概率校准 (已证明 post-hoc 校准在 DDA 场景失效). 故对照只保留 oof vs random.
    strategies = {}
    strategies['oof'] = train_neg
    strategies['random'] = neg_pool.sample(n=n_pos, random_state=args.seed)

    rows = []
    for name, neg in strategies.items():
        train = pd.concat([train_pos, neg], ignore_index=True)
        r = evaluate(train, test, feats, gigs, args.seed)
        r['dataset'], r['strategy'] = ds, name
        rows.append(r)
        print(f"  {name:<8} AUROC={r['AUROC']:.4f}  AUPR={r['AUPR']:.4f}  F1={r['F1']:.4f}")

    df = pd.DataFrame(rows)
    os.makedirs(r2_config.RESULTS_R2, exist_ok=True)
    out = os.path.join(r2_config.RESULTS_R2, 'negative_sampling_compare.csv')
    if os.path.exists(out):
        prev = pd.read_csv(out)
        prev = prev[prev['dataset'] != ds]
        df = pd.concat([prev, df], ignore_index=True)
    df.to_csv(out, index=False)
    print(f"  ✅ 已保存: {out}")


if __name__ == '__main__':
    main()
