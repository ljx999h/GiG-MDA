# -*- coding: utf-8 -*-
"""
cold-drug top-k 决策指标: Base / MolEmb32 / GRMF-embed / ECFP32 在
cold-drug test 上的 P@10 / P@50 / R@100 (逐 seed).
用法: python code/cold_topk.py --dataset C --seeds 42 7 123 2024
"""
import argparse
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, 'code')
sys.path.insert(0, 'code')
from cold_eval import load, evaluate, load_mol_emb
from compare_pretrain_ablation import mol_feats, ecfp_feats, fit_transform


def topk(scores, y, k=10):
    idx = np.argsort(-scores)
    top = y[idx[:k]]
    p = top.sum() / k
    r = top.sum() / max(y.sum(), 1)
    return p, r


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
        from build_results_ledger import gigs_features
        tr_dot, tr_emb = gigs_features(train, gigs)
        te_dot, te_emb = gigs_features(test, gigs)
        tr_mol = mol_feats(train, emb_map, seed, shuffle=False)
        te_mol = mol_feats(test, emb_map, seed, shuffle=False)
        uniq_d = pd.unique(train['drugID'].astype(str).str.strip())
        uniq_df = pd.DataFrame({'drugID': uniq_d})
        tr_m32, te_m32 = fit_transform(tr_mol, te_mol, seed,
                                       uniq_raw=mol_feats(uniq_df, emb_map, seed, shuffle=False))
        tr_ecfp = ecfp_feats(ds, train, seed)
        te_ecfp = ecfp_feats(ds, test, seed)
        tr_e32, te_e32 = fit_transform(tr_ecfp, te_ecfp, seed)
        variants = {
            'Base': (train[base].values, test[base].values),
            'MolEmb32': (np.hstack([train[base].values, tr_m32]),
                         np.hstack([test[base].values, te_m32])),
            'GRMF': (np.hstack([train[base].values, tr_emb]),
                     np.hstack([test[base].values, te_emb])),
            'ECFP32': (np.hstack([train[base].values, tr_e32]),
                       np.hstack([test[base].values, te_e32])),
        }
        for name, (Xtr, Xte) in variants.items():
            import xgboost as xgb
            import r2_config
            clf = xgb.XGBClassifier(**r2_config.XGB_CONFIG)
            clf.fit(Xtr.astype(np.float32), ytr)
            scores = clf.predict_proba(Xte.astype(np.float32))[:, 1]
            p10, r10 = topk(scores, yte, 10)
            p50, r50 = topk(scores, yte, 50)
            p100, r100 = topk(scores, yte, 100)
            rows.append({'dataset': ds, 'seed': seed, 'variant': name,
                         'P@10': p10, 'P@50': p50, 'R@100': r100})
            print(f'{ds} s{seed} {name:10s} P@10={p10:.3f} P@50={p50:.3f} R@100={r100:.4f}')
    df = pd.DataFrame(rows)
    out = f'results/R2/cold_topk_{ds}.csv'
    df.to_csv(out, index=False)
    print('saved', out)


if __name__ == '__main__':
    main()
