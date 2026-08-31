# -*- coding: utf-8 -*-
"""
干净协议消融: 同一评估配置与负样本, 只变特征构建的邻居源.
mapping80 (泄漏源: 测试关联进入训练侧特征构建) vs r2train (安全源).
用法: python code/ablate_neighbor_source.py C code/results/score_C_mapping80.csv 42
"""
import argparse
import pandas as pd
import numpy as np
import xgboost as xgb
import sys
sys.path.insert(0, 'code')
import r2_config
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_curve


def evaluate(Xtr, ytr, Xte, yte, seed=42, xgb_seed=None):
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    cfg = dict(r2_config.XGB_CONFIG)
    if xgb_seed is not None:
        cfg['random_state'] = xgb_seed
    threshs = []
    for tr, va in skf.split(Xtr, ytr):
        clf = xgb.XGBClassifier(**cfg)
        clf.fit(Xtr[tr], ytr[tr])
        yv = clf.predict_proba(Xtr[va])[:, 1]
        p, r, t = precision_recall_curve(ytr[va], yv)
        f1s = 2 * p * r / (p + r + 1e-9)
        best = f1s.argmax()
        threshs.append(t[best] if best < len(t) else 0.5)
    clf = xgb.XGBClassifier(**cfg)
    clf.fit(Xtr, ytr)
    yp = clf.predict_proba(Xte)[:, 1]
    return roc_auc_score(yte, yp), average_precision_score(yte, yp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dataset', choices=['C', 'F', 'DDCD'])
    ap.add_argument('score_path')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--xgb-seed', type=int, default=None,
                    help='XGBoost 训练随机种子 (默认用统一配置)')
    args = ap.parse_args()
    ds = args.dataset

    if ds == 'DDCD':
        root = 'data/DDCD/Splits'
    else:
        root = f'data/{ds}-Dataset/Splits'
    feats = r2_config.DATASETS[ds]['feats']

    pos = pd.read_csv(f'{root}/train_positives.csv')[['drugID', 'diseaseID', 'label']]
    neg = pd.read_csv(f'{root}/train_negatives.csv')[['drugID', 'diseaseID', 'label']]
    train = pd.concat([pos, neg], ignore_index=True)
    mf = pd.read_csv(f'{root}/split_manifest.csv')
    test = mf[mf['split'] == 'test'][['drugID', 'diseaseID', 'label']].copy()

    score = pd.read_csv(args.score_path).fillna(0.0)
    score['drugID'] = score['drugID'].astype(str).str.strip()
    score['diseaseID'] = score['diseaseID'].astype(str).str.strip()
    if 'label' in score.columns:
        score = score.drop(columns=['label'])
    if ds == 'C':
        mapping = pd.read_csv(f'data/C-Dataset/Mapping/mapping_C.csv')
        if 'Unnamed: 0' in mapping.columns:
            mapping = mapping.drop(columns=['Unnamed: 0'])
        drugbanks = sorted(mapping.iloc[:, 0].astype(str).str.strip().unique())
        idx_to_db = {i: db for i, db in enumerate(drugbanks)}
        score['drugID'] = score['drugID'].map(lambda x: idx_to_db.get(int(x), x))

    for df_ in (train, test):
        df_['drugID'] = df_['drugID'].astype(str).str.strip()
        df_['diseaseID'] = df_['diseaseID'].astype(str).str.strip()

    train = train.merge(score[['drugID', 'diseaseID'] + feats],
                        on=['drugID', 'diseaseID'], how='left').fillna(0.0)
    test = test.merge(score[['drugID', 'diseaseID'] + feats],
                      on=['drugID', 'diseaseID'], how='left').fillna(0.0)
    assert len(test) == mf[mf['split'] == 'test'].shape[0], 'test 特征缺失'

    auroc, aupr = evaluate(train[feats].values, train['label'].values,
                           test[feats].values, test['label'].values,
                           seed=args.seed, xgb_seed=args.xgb_seed)
    print(f'{ds} s{args.seed} neighbor-source={args.score_path.split("/")[-1]}: '
          f'AUROC={auroc:.4f} AUPR={aupr:.4f}')


if __name__ == '__main__':
    main()
