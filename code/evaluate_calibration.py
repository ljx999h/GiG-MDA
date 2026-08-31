"""
方案B: 校准 + Top-K 决策指标 (evaluate_calibration)

对指定数据集 × 模型, 在方案B 协议下:
  1. train 内 5 折 CV 的 OOF 概率 → 拟合校准器 (Platt / Isotonic)  [train 内, 不碰 test]
  2. 全量训练 → 封存 test 一次评估
  3. 报告: ECE/Brier (校准前/后) + AUPR + AUPR@k / Recall@k (top-k 决策指标)

支撑论文 "calibrated prioritization" 定位 (诊断报告 P1-5).
用法:
  python code/evaluate_calibration.py --dataset C --model GiG-MDA
  python code/evaluate_calibration.py --dataset C --model all
"""
import argparse
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             precision_recall_curve)
import xgboost as xgb

import r2_config

MODEL_CFG = {
    'GiG-MDA':   dict(variant='embed'),
    'XGBoost-dot': dict(variant='dot'),
    'XGBoost-mirage': dict(variant='mirage'),
    'LR':        dict(variant='lr'),
}


def load_data(ds, cfg):
    pos = pd.read_csv(cfg['train_pos']).fillna(0.0)
    neg = pd.read_csv(cfg['train_neg']).fillna(0.0)
    feats = cfg['feats']
    keep = ['drugID', 'diseaseID', 'label'] + feats
    train = pd.concat([pos[keep], neg[keep]], ignore_index=True)
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
    assert len(test) == len(test_mf), 'test 特征缺失'
    with open(cfg['gigs'], 'rb') as f:
        gigs = pickle.load(f)
    return train, test, gigs, feats


def gigs_feats(df, gigs):
    X, Y = gigs['X'], gigs['Y']
    d2i, s2i = gigs['drug_to_idx'], gigs['disease_to_idx']
    k = X.shape[1]
    d = df['drugID'].astype(str).str.strip().map(d2i)
    s = df['diseaseID'].astype(str).str.strip().map(s2i)
    v = d.notna() & s.notna()
    dot = np.zeros(len(df)); emb = np.zeros((len(df), 2 * k))
    if v.any():
        dv, sv = d[v].astype(int).values, s[v].astype(int).values
        dot[v.values] = np.sum(X[dv] * Y[sv], axis=1)
        emb[v.values, :k] = X[dv]; emb[v.values, k:] = Y[sv]
    return dot, emb


def make_xgb():
    return xgb.XGBClassifier(**r2_config.XGB_CONFIG)


def ece(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(y_prob, bins[1:-1])
    e = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        conf = y_prob[m].mean()
        acc = y_true[m].mean()
        e += (m.sum() / len(y_true)) * abs(conf - acc)
    return e


def topk_metrics(y_true, y_prob, k):
    """前 k 的 Precision 与 Recall: Precision@k = 前 k 中正样本数 / k;
    Recall@k = 前 k 中正样本数 / 测试正样本总数."""
    order = np.argsort(-y_prob)[:k]
    top_true = y_true[order]
    prec = top_true.sum() / max(1, k)
    rec = top_true.sum() / max(1, y_true.sum())
    return prec, rec


def topk_report(y_true, y_prob, ks=(10, 50, 100)):
    out = {}
    for k in ks:
        p, r = topk_metrics(y_true, y_prob, k)
        out[f'Precision@{k}'] = p
        out[f'Recall@{k}'] = r
    return out


def eval_model(train, test, gigs, feats, variant, seed=42):
    tr_dot, tr_emb = gigs_feats(train, gigs)
    te_dot, te_emb = gigs_feats(test, gigs)
    ytr, yte = train['label'].values, test['label'].values

    if variant == 'mirage':
        Xtr, Xte = train[feats].values.astype(np.float32), test[feats].values.astype(np.float32)
    elif variant == 'dot':
        Xtr = np.hstack([train[feats].values, tr_dot.reshape(-1, 1)]).astype(np.float32)
        Xte = np.hstack([test[feats].values, te_dot.reshape(-1, 1)]).astype(np.float32)
    elif variant == 'embed':
        Xtr = np.hstack([train[feats].values, tr_emb]).astype(np.float32)
        Xte = np.hstack([test[feats].values, te_emb]).astype(np.float32)
    else:  # lr
        from sklearn.preprocessing import StandardScaler
        Xtr = np.hstack([train[feats].values, tr_dot.reshape(-1, 1)]).astype(np.float32)
        Xte = np.hstack([test[feats].values, te_dot.reshape(-1, 1)]).astype(np.float32)
        sc = StandardScaler().fit(Xtr)
        Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)

    # train 内 OOF 概率 → 校准器
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    oof = np.zeros(len(ytr))
    for tr, va in skf.split(Xtr, ytr):
        if variant == 'lr':
            clf = LogisticRegression(max_iter=1000, C=1.0, solver='liblinear', random_state=42)
        else:
            clf = make_xgb()
        clf.fit(Xtr[tr], ytr[tr])
        oof[va] = clf.predict_proba(Xtr[va])[:, 1]

    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(oof, ytr)
    plat = LogisticRegression(max_iter=1000)
    plat.fit(oof.reshape(-1, 1), ytr)

    # 全量训练 → test
    if variant == 'lr':
        clf = LogisticRegression(max_iter=1000, C=1.0, solver='liblinear', random_state=42)
    else:
        clf = make_xgb()
    clf.fit(Xtr, ytr)
    yp = clf.predict_proba(Xte)[:, 1]

    res = {
        'dataset': None, 'model': None, 'AUROC': roc_auc_score(yte, yp),
        'AUPR': average_precision_score(yte, yp),
        'ECE_raw': ece(yte, yp),
        'ECE_iso': ece(yte, iso.predict(yp)),
        'ECE_platt': ece(yte, plat.predict_proba(yp.reshape(-1, 1))[:, 1]),
        'Brier_raw': float(np.mean((yp - yte) ** 2)),
        'Brier_iso': float(np.mean((iso.predict(yp) - yte) ** 2)),
        'Brier_platt': float(np.mean((plat.predict_proba(yp.reshape(-1, 1))[:, 1] - yte) ** 2)),
    }
    res.update(topk_report(yte, yp))
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset', required=True, choices=['C', 'F', 'DDCD'])
    ap.add_argument('--model', default='GiG-MDA',
                    help='GiG-MDA | XGBoost-dot | XGBoost-mirage | LR | all')
    ap.add_argument('--seed', type=int, default=r2_config.SEED)
    args = ap.parse_args()
    cfg = r2_config.DATASETS[args.dataset]

    train, test, gigs, feats = load_data(args.dataset, cfg)
    models = list(MODEL_CFG) if args.model == 'all' else [args.model]
    rows = []
    for m in models:
        r = eval_model(train, test, gigs, feats, MODEL_CFG[m]['variant'], args.seed)
        r['dataset'], r['model'] = args.dataset, m
        rows.append(r)
        print(f"{m:<14} AUROC={r['AUROC']:.4f} AUPR={r['AUPR']:.4f} "
              f"ECE: raw={r['ECE_raw']:.4f} iso={r['ECE_iso']:.4f} platt={r['ECE_platt']:.4f} "
              f"| P@10={r['Precision@10']:.4f} R@50={r['Recall@50']:.4f}")

    df = pd.DataFrame(rows)
    os.makedirs(r2_config.RESULTS_R2, exist_ok=True)
    out = os.path.join(r2_config.RESULTS_R2, f'calibration_{args.dataset}.csv')
    df.to_csv(out, index=False)
    print(f"  ✅ 已保存: {out}")


if __name__ == '__main__':
    main()
