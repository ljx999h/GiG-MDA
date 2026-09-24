# -*- coding: utf-8 -*-
"""泄漏型协议下的完整方法评估 (mapping80 特征源; 与 §3.1 消融同一协议).
对 base 与三通道 (leak-base + GRMF 嵌入 + MolEmb32) 报告 AUROC/AUPR/F1.
用法: python code/leakprone_three_channel.py --dataset C --seed 42
输出: results/R2/leakprone_three_channel.csv
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, 'code')
import r2_config
import build_results_ledger as L

SCORE = {
    'C': 'code/results/score_C_mapping80.csv',
    'F': 'code/results/score_F_mapping80.csv',
    'DDCD': 'code/results/score_DDCD_mapping80.csv',
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=['C', 'F', 'DDCD'])
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--score', default=None, help='覆盖特征文件 (默认 mapping80)')
    ap.add_argument('--tag', default='leak', help='变体名前缀 (输出标识用)')
    args = ap.parse_args()
    ds, seed = args.dataset, args.seed
    cfg = r2_config.DATASETS[ds]

    pos = pd.read_csv(cfg['train_pos']).fillna(0.0)
    neg = pd.read_csv(cfg['train_neg']).fillna(0.0)
    feat_cols = [c for c in cfg['feats'] if c in pos.columns]
    keep = ['drugID', 'diseaseID', 'label'] + feat_cols
    train = pd.concat([pos[keep], neg[keep]], ignore_index=True)
    for c in ['drugID', 'diseaseID']:
        train[c] = train[c].astype(str).str.strip()
    mf = pd.read_csv(cfg['manifest'])
    mf['drugID'] = mf['drugID'].astype(str).str.strip()
    mf['diseaseID'] = mf['diseaseID'].astype(str).str.strip()
    test_mf = mf[mf['split'] == 'test'][['drugID', 'diseaseID', 'label']].copy()

    # 泄漏型特征 (mapping80)
    score = pd.read_csv(args.score or SCORE[ds]).fillna(0.0)
    score['drugID'] = score['drugID'].astype(str).str.strip()
    score['diseaseID'] = score['diseaseID'].astype(str).str.strip()
    if 'label' in score.columns:
        score = score.drop(columns=['label'])
    if ds == 'C':
        mapping = pd.read_csv('data/C-Dataset/Mapping/mapping_C.csv')
        if 'Unnamed: 0' in mapping.columns:
            mapping = mapping.drop(columns=['Unnamed: 0'])
        drugbanks = sorted(mapping.iloc[:, 0].astype(str).str.strip().unique())
        idx_to_db = {i: db for i, db in enumerate(drugbanks)}
        score['drugID'] = score['drugID'].map(lambda x: idx_to_db.get(int(x), x))

    key = ['drugID', 'diseaseID']
    test = test_mf.merge(score[key + feat_cols], on=key, how='inner')
    assert len(test) == len(test_mf), f'test 特征缺失 {len(test_mf)-len(test)}'
    train = train[key + ['label']].merge(score[key + feat_cols], on=key, how='left').fillna(0.0)

    gigs = L.load_gigs(cfg)
    _, tr_emb = L.gigs_features(train, gigs)
    _, te_emb = L.gigs_features(test, gigs)
    tr_mol, te_mol = L.mol_emb_features(ds, train, test, n_comp=32, seed=seed)

    ytr = train['label'].values
    yte = test['label'].values
    Xb_tr = train[feat_cols].values.astype(np.float32)
    Xb_te = test[feat_cols].values.astype(np.float32)
    variants = {
        f'{args.tag}-base': (Xb_tr, Xb_te),
        f'{args.tag}-three-channel': (np.hstack([Xb_tr, tr_emb, tr_mol]),
                                      np.hstack([Xb_te, te_emb, te_mol])),
    }
    rows = []
    for name, (Xtr, Xte) in variants.items():
        r = L.evaluate_holdout(Xtr, ytr, Xte, yte, L.make_unified_xgb,
                               seed=seed, n_boot=1000)
        rows.append({'dataset': ds, 'seed': seed, 'variant': name,
                     'AUROC': round(r['AUROC'], 4), 'AUPR': round(r['AUPR'], 4),
                     'F1': round(r['F1'], 4), 'feature_dim': Xtr.shape[1]})
        print(f"{ds} s{seed} {name:20s} AUROC={r['AUROC']:.4f} "
              f"AUPR={r['AUPR']:.4f} F1={r['F1']:.4f}", flush=True)

    out = 'results/R2/leakprone_three_channel.csv'
    df = pd.DataFrame(rows)
    if os.path.exists(out):
        prev = pd.read_csv(out)
        k = ['dataset', 'seed', 'variant']
        prev = prev.merge(df[k].drop_duplicates(), on=k, how='left', indicator=True)
        prev = prev[prev['_merge'] == 'left_only'].drop(columns=['_merge'])
        df = pd.concat([prev, df], ignore_index=True)
    df = df.sort_values(['dataset', 'seed', 'variant']).reset_index(drop=True)
    df.to_csv(out, index=False)
    print('saved', out)


if __name__ == '__main__':
    main()
