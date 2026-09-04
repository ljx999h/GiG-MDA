# -*- coding: utf-8 -*-
"""
正则化诊断 (Discussion 机制段的数据源)
=======================================
动机: 3.7 的 drug-level 随机对照显示恒定的 per-drug 随机向量 ≈ MolEmb32.
进一步诊断表明该增益在共享 XGBoost 配置 (500 trees x depth 10) 下对小训练集
主要来自"加列即正则化": 对纯 base 特征做列采样 (colsample_bytree) 即可复现
同量级 AUPR 提升; 而在大训练集 (DDCD) 上同一正则反而略有害 (base 未过拟合).

对每个 cold-drug (dataset, seed) 组合输出:
  base_default:  base 特征 + 统一配置 (500/d10)
  base_colsample_03: base 特征 + colsample_bytree=0.3
  base_depth4:   base 特征 + max_depth=4
结果: results/R2/regularization_diagnostic.csv

用法: python code/regularization_diagnostic.py --dataset C --seeds 42 7 123 2024
      (DDCD 同理; 时间较长可用 --quick 只跑 42)
"""
import argparse
import os

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score

import sys
sys.path.append('code')
import r2_config
from cold_eval import load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=['C', 'F', 'DDCD'])
    ap.add_argument('--seeds', nargs='+', type=int, default=[42])
    ap.add_argument('--out', default='results/R2/regularization_diagnostic.csv')
    args = ap.parse_args()
    base_cfg = dict(r2_config.XGB_CONFIG)
    rows = []
    for seed in args.seeds:
        train, test, gigs, feats, base = load(args.dataset, seed, 'cold-drug')
        ytr, yte = train['label'].values, test['label'].values
        Xtr = train[base].values.astype(np.float32)
        Xte = test[base].values.astype(np.float32)
        for name, over in [('base_default', {}),
                           ('base_colsample_03', {'colsample_bytree': 0.3}),
                           ('base_depth4', {'max_depth': 4})]:
            cfg = dict(base_cfg)
            cfg.update(over)
            clf = xgb.XGBClassifier(**cfg)
            clf.fit(Xtr, ytr)
            aupr = average_precision_score(yte, clf.predict_proba(Xte)[:, 1])
            rows.append({'dataset': args.dataset, 'seed': seed, 'config': name,
                         'AUPR': aupr})
            print(f"{args.dataset} s{seed} {name:22s} AUPR {aupr:.4f}", flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    df.to_csv(args.out, index=False)
    print('saved:', args.out)


if __name__ == '__main__':
    main()
