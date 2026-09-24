# -*- coding: utf-8 -*-
"""
导出冷启动评估的逐对预测（候选对标识 + 真实标签 + 预测分数），
并用独立实现重新计算 AUROC / AP，与稿件归档值对照。

用途：回应"复跑数值差异需在原实验环境中确认"这一核查项。
输出（默认 results/pred_dump/）：
  {dataset}_s{seed}_{variant}_pred.csv     每变体：drugID,diseaseID,label,score
  {dataset}_s{seed}_summary.csv            每变体的 AUROC/AP（本脚本独立计算）
  {dataset}_s{seed}_compare.csv            与归档值对照

用法：
  python code/dump_cold_predictions.py --dataset C --seed 42
  python code/dump_cold_predictions.py --dataset C --seed 42 --out-dir results/pred_dump_cpu
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, 'code')
import r2_config
from cold_eval import load, load_mol_emb, gigs_feats, MOL_FEATS

# 稿件中的四个主要 XGBoost 配置（变体名与 cold_eval.py 一致）
VARIANTS = ['MiRAGE', 'MiRAGE+MolEmb32', 'MiRAGE+embed', 'MiRAGE+embed+MolEmb32']
# 稿件归档值（cold_start_results.csv）
ARCHIVE = {
    ('C', 42): {'MiRAGE': 0.3003, 'MiRAGE+MolEmb32': 0.3005,
                'MiRAGE+embed': 0.3046, 'MiRAGE+embed+MolEmb32': 0.3034},
}


def ap_independent(y, s):
    """不依赖 sklearn 的 AP 实现（逐步法），用于交叉核对。"""
    order = np.argsort(-np.asarray(s))
    y = np.asarray(y)[order]
    tp = np.cumsum(y)
    prec = tp / np.arange(1, len(y) + 1)
    n_pos = y.sum()
    if n_pos == 0:
        return float('nan')
    # 仅在正样本位置累加（与 average_precision_score 定义一致）
    return float((prec * y).sum() / n_pos)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=['C', 'F', 'DDCD'])
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--mode', default='cold-drug')
    ap.add_argument('--out-dir', default='results/pred_dump')
    args = ap.parse_args()
    ds, seed = args.dataset, args.seed
    os.makedirs(args.out_dir, exist_ok=True)

    print(f'=== {ds} seed {seed} | device={r2_config.XGB_CONFIG.get("device")} ===')
    train, test, gigs, feats, base = load(ds, seed, args.mode)
    ytr, yte = train['label'].values, test['label'].values
    tr_dot, tr_emb = gigs_feats(train, gigs)
    te_dot, te_emb = gigs_feats(test, gigs)

    # MolEmb32（PCA 仅在去重训练药物上拟合）
    from sklearn.decomposition import PCA
    emb_map = load_mol_emb(ds)
    pca = PCA(n_components=32, random_state=42)
    tr_raw = np.array([emb_map.get(d, np.zeros(768)) for d in train['drugID']], dtype=np.float32)
    te_raw = np.array([emb_map.get(d, np.zeros(768)) for d in test['drugID']], dtype=np.float32)
    uniq_d = pd.unique(train['drugID'].astype(str).str.strip())
    uniq_raw = np.array([emb_map.get(d, np.zeros(768)) for d in uniq_d], dtype=np.float32)
    pca.fit(uniq_raw)
    tr_pca32 = pca.transform(tr_raw).astype(np.float32)
    te_pca32 = pca.transform(te_raw).astype(np.float32)

    X = {
        'MiRAGE': (train[base].values, test[base].values),
        'MiRAGE+MolEmb32': (np.hstack([train[base].values, tr_pca32]),
                            np.hstack([test[base].values, te_pca32])),
        'MiRAGE+embed': (np.hstack([train[base].values, tr_emb]),
                         np.hstack([test[base].values, te_emb])),
        'MiRAGE+embed+MolEmb32': (np.hstack([train[base].values, tr_emb, tr_pca32]),
                                  np.hstack([test[base].values, te_emb, te_pca32])),
    }

    rows = []
    for name in VARIANTS:
        Xtr, Xte = X[name]
        Xtr = Xtr.astype(np.float32); Xte = Xte.astype(np.float32)
        clf = xgb.XGBClassifier(**r2_config.XGB_CONFIG)
        clf.fit(Xtr, ytr)
        score = clf.predict_proba(Xte)[:, 1]

        out = pd.DataFrame({
            'drugID': test['drugID'].astype(str).str.strip().values,
            'diseaseID': test['diseaseID'].astype(str).str.strip().values,
            'label': yte,
            'score': score,
        })
        f = os.path.join(args.out_dir, f'{ds}_s{seed}_{name.replace("+", "_")}_pred.csv')
        out.to_csv(f, index=False)

        auroc = roc_auc_score(yte, score)
        ap_sk = average_precision_score(yte, score)
        ap_mine = ap_independent(yte, score)
        arch = ARCHIVE.get((ds, seed), {}).get(name)
        rows.append({
            'dataset': ds, 'seed': seed, 'variant': name, 'dim': Xtr.shape[1],
            'AUROC': round(auroc, 4), 'AUPR_sklearn': round(ap_sk, 4),
            'AUPR_independent': round(ap_mine, 4),
            'archive_AUPR': arch,
            'delta_vs_archive': round(ap_sk - arch, 4) if arch else None,
        })
        print(f'  {name:<24} dim={Xtr.shape[1]:>4}  AUROC={auroc:.4f}  '
              f'AP={ap_sk:.4f} (独立实现 {ap_mine:.4f})'
              + (f'  归档={arch:.4f} Δ={ap_sk-arch:+.4f}' if arch else ''))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out_dir, f'{ds}_s{seed}_summary.csv'), index=False)
    print(f'\n已写出预测与汇总 → {args.out_dir}/')


if __name__ == '__main__':
    main()
