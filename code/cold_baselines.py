# -*- coding: utf-8 -*-
"""冷启动经典基线 (LR / RF) 在 cold-drug test 上的 AUPR.
与 tab:cold 的 base 模型使用同一 MiRAGE 特征集, 用于判断简单分类器是否也能达到
增广通道的水平. 配置与主表基线一致 (LR: scaler+liblinear C=1; RF: 100 棵 depth 12).
用法: python code/cold_baselines.py --dataset C --seeds 42 7 123 2024
输出: results/R2/cold_baselines.csv (dataset, seed, model, AUPR)
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, 'code')
from cold_eval import load

RANDOM_SEED = 42


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=['C', 'F', 'DDCD'])
    ap.add_argument('--seeds', nargs='+', type=int, default=[42, 7, 123, 2024])
    args = ap.parse_args()
    rows = []
    for seed in args.seeds:
        train, test, gigs, feats, base = load(args.dataset, seed, 'cold-drug')
        ytr, yte = train['label'].values, test['label'].values
        Xtr = train[base].values.astype(np.float64)
        Xte = test[base].values.astype(np.float64)
        # LR (scaler + liblinear)
        sc = StandardScaler().fit(Xtr)
        lr = LogisticRegression(max_iter=1000, C=1.0, solver='liblinear',
                                random_state=RANDOM_SEED)
        lr.fit(sc.transform(Xtr), ytr)
        p_lr = lr.predict_proba(sc.transform(Xte))[:, 1]
        # RF
        rf = RandomForestClassifier(n_estimators=100, max_depth=12, n_jobs=-1,
                                    random_state=RANDOM_SEED)
        rf.fit(Xtr, ytr)
        p_rf = rf.predict_proba(Xte)[:, 1]
        for name, p in [('LR', p_lr), ('RF', p_rf)]:
            rows.append({'dataset': args.dataset, 'seed': seed, 'model': name,
                         'AUROC': round(roc_auc_score(yte, p), 4),
                         'AUPR': round(average_precision_score(yte, p), 4)})
            print(f'{args.dataset} s{seed} {name}: AUROC={roc_auc_score(yte, p):.4f} '
                  f'AUPR={average_precision_score(yte, p):.4f}', flush=True)
    out = 'results/R2/cold_baselines.csv'
    df = pd.DataFrame(rows)
    if os.path.exists(out):
        prev = pd.read_csv(out)
        key = ['dataset', 'seed', 'model']
        prev = prev.merge(df[key].drop_duplicates(), on=key, how='left', indicator=True)
        prev = prev[prev['_merge'] == 'left_only'].drop(columns=['_merge'])
        df = pd.concat([prev, df], ignore_index=True)
    df = df.sort_values(['dataset', 'seed', 'model']).reset_index(drop=True)
    df.to_csv(out, index=False)
    print('saved', out)


if __name__ == '__main__':
    main()
