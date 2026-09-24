# -*- coding: utf-8 -*-
"""同一分类器 (LR) 下的表征对照: base / +embed / +MolEmb32 / +both, cold-drug test AUPR.
回答 "增益来自表征还是分类器" 的问题 (cold 设置, 4 seeds).
用法: python code/cold_lr_rep.py --dataset C
输出: results/R2/cold_lr_rep.csv
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, 'code')
from cold_eval import load, load_mol_emb
from compare_pretrain_ablation import mol_feats, fit_transform
from build_results_ledger import gigs_features

RANDOM_SEED = 42


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=['C', 'F', 'DDCD'])
    ap.add_argument('--seeds', nargs='+', type=int, default=[42, 7, 123, 2024])
    args = ap.parse_args()
    ds = args.dataset
    emb_map = load_mol_emb(ds)
    rows = []
    for seed in args.seeds:
        train, test, gigs, feats, base = load(ds, seed, 'cold-drug')
        ytr, yte = train['label'].values, test['label'].values
        Xb_tr = train[base].values.astype(np.float32)
        Xb_te = test[base].values.astype(np.float32)
        _, tr_emb = gigs_features(train, gigs)
        _, te_emb = gigs_features(test, gigs)
        tr_mol = mol_feats(train, emb_map, seed, shuffle=False)
        te_mol = mol_feats(test, emb_map, seed, shuffle=False)
        uniq_d = pd.unique(train['drugID'].astype(str).str.strip())
        uniq_df = pd.DataFrame({'drugID': uniq_d})
        tr_m32, te_m32 = fit_transform(
            tr_mol, te_mol, seed,
            uniq_raw=mol_feats(uniq_df, emb_map, seed, shuffle=False))
        variants = {
            'LR-base': (Xb_tr, Xb_te),
            'LR+embed': (np.hstack([Xb_tr, tr_emb]), np.hstack([Xb_te, te_emb])),
            'LR+MolEmb32': (np.hstack([Xb_tr, tr_m32]), np.hstack([Xb_te, te_m32])),
            'LR+both': (np.hstack([Xb_tr, tr_emb, tr_m32]),
                        np.hstack([Xb_te, te_emb, te_m32])),
        }
        for name, (Xtr, Xte) in variants.items():
            sc = StandardScaler().fit(Xtr)
            lr = LogisticRegression(max_iter=1000, C=1.0, solver='liblinear',
                                    random_state=RANDOM_SEED)
            lr.fit(sc.transform(Xtr), ytr)
            p = lr.predict_proba(sc.transform(Xte))[:, 1]
            rows.append({'dataset': ds, 'seed': seed, 'variant': name,
                         'AUROC': round(roc_auc_score(yte, p), 4),
                         'AUPR': round(average_precision_score(yte, p), 4)})
            print(f'{ds} s{seed} {name:12s} AUROC={roc_auc_score(yte, p):.4f} '
                  f'AUPR={average_precision_score(yte, p):.4f}', flush=True)
    out = 'results/R2/cold_lr_rep.csv'
    df = pd.DataFrame(rows)
    if os.path.exists(out):
        prev = pd.read_csv(out)
        key = ['dataset', 'seed', 'variant']
        prev = prev.merge(df[key].drop_duplicates(), on=key, how='left', indicator=True)
        prev = prev[prev['_merge'] == 'left_only'].drop(columns=['_merge'])
        df = pd.concat([prev, df], ignore_index=True)
    df = df.sort_values(['dataset', 'seed', 'variant']).reset_index(drop=True)
    df.to_csv(out, index=False)
    print('saved', out)


if __name__ == '__main__':
    main()
