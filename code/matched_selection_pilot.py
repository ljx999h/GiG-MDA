# -*- coding: utf-8 -*-
"""
D 组 pilot: matched model selection (per-variant inner-CV, 同预算)
对 cold-drug (dataset, seed), 每个变体 (MiRAGE / MolEmb32 / Gaussian32 /
dot / embed / both) 在 train fold 内 3-fold CV 于小网格选配置
(max_depth x min_child_weight x colsample_bytree; 验证 AUPR),
以选中配置在完整 train 训练, outer test 评估一次.
输出: results/R2/matched_selection_pilot.csv
用法: python code/matched_selection_pilot.py --dataset C --seeds 42 2024
      python code/matched_selection_pilot.py --dataset DDCD --seeds 42
"""
import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import average_precision_score
from sklearn.decomposition import PCA

sys.path.insert(0, 'code')
import r2_config
from cold_eval import load, load_mol_emb

GRID = list(itertools.product([6, 10], [1, 5], [0.5, 1.0]))


def make_variants(ds, train, test, seed, base_feats, gigs):
    emb_map = load_mol_emb(ds)
    Xb_tr = train[base_feats].values.astype(np.float32)
    Xb_te = test[base_feats].values.astype(np.float32)
    uniq_d = list(pd.unique(train['drugID'].astype(str).str.strip()))
    out = {'MiRAGE': (Xb_tr, Xb_te)}

    def rows_map(df):
        d2i, s2i = gigs['drug_to_idx'], gigs['disease_to_idx']
        d = df['drugID'].astype(str).str.strip().map(d2i)
        s = df['diseaseID'].astype(str).str.strip().map(s2i)
        v = d.notna() & s.notna()
        k = gigs['X'].shape[1]
        dot = np.zeros(len(df), dtype=np.float32)
        emb = np.zeros((len(df), 2 * k), dtype=np.float32)
        if v.any():
            dv = d[v].astype(int).values
            sv = s[v].astype(int).values
            dot[v.values] = np.sum(gigs['X'][dv] * gigs['Y'][sv], axis=1)
            emb[v.values, :k] = gigs['X'][dv]
            emb[v.values, k:] = gigs['Y'][sv]
        return dot, emb

    tr_dot, tr_emb = rows_map(train)
    te_dot, te_emb = rows_map(test)
    out['dot'] = (np.hstack([Xb_tr, tr_dot[:, None]]), np.hstack([Xb_te, te_dot[:, None]]))
    out['embed'] = (np.hstack([Xb_tr, tr_emb]), np.hstack([Xb_te, te_emb]))

    # MolEmb32
    pca = PCA(n_components=32, random_state=42)
    raw_u = np.array([emb_map.get(d, np.zeros(768)) for d in uniq_d], dtype=np.float32)
    pca.fit(raw_u)
    z = np.zeros(768, dtype=np.float32)
    tr_e = pca.transform(np.array([emb_map.get(d, z) for d in train['drugID']])).astype(np.float32)
    te_e = pca.transform(np.array([emb_map.get(d, z) for d in test['drugID']])).astype(np.float32)
    out['MolEmb32'] = (np.hstack([Xb_tr, tr_e]), np.hstack([Xb_te, te_e]))
    out['both'] = (np.hstack([Xb_tr, tr_emb, tr_e]), np.hstack([Xb_te, te_emb, te_e]))

    # Gaussian32 (drug-level 单次)
    var = raw_u.var(0) + 1e-9
    rng = np.random.RandomState(seed * 100000)
    all_d = list(dict.fromkeys(list(uniq_d) +
                               list(pd.unique(test['drugID'].astype(str).str.strip()))))
    gmap = dict(zip(all_d, rng.normal(0, np.sqrt(var),
                                      size=(len(all_d), var.shape[0])).astype(np.float32)))
    pca_g = PCA(n_components=32, random_state=42)
    pca_g.fit(np.array([gmap[d] for d in uniq_d]))
    d0 = np.zeros(32, dtype=np.float32)
    tr_g = pca_g.transform(np.array([gmap.get(d, d0) for d in train['drugID']])).astype(np.float32)
    te_g = pca_g.transform(np.array([gmap.get(d, d0) for d in test['drugID']])).astype(np.float32)
    out['Gaussian32'] = (np.hstack([Xb_tr, tr_g]), np.hstack([Xb_te, te_g]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=['C', 'F', 'DDCD'])
    ap.add_argument('--seeds', nargs='+', type=int, default=[42])
    args = ap.parse_args()
    ds = args.dataset
    rows = []
    for seed in args.seeds:
        train, test, gigs, feats, base = load(ds, seed, 'cold-drug')
        ytr, yte = train['label'].values, test['label'].values
        variants = make_variants(ds, train, test, seed, base, gigs)
        print(f'\n=== {ds} s{seed} ===', flush=True)
        for name, (Xtr, Xte) in variants.items():
            best = None
            kf = KFold(3, shuffle=True, random_state=seed)
            for md, mcw, cs in GRID:
                cv = []
                for tr_i, va_i in kf.split(Xtr):
                    cfg = dict(r2_config.XGB_CONFIG)
                    cfg.update(max_depth=md, min_child_weight=mcw, colsample_bytree=cs)
                    clf = xgb.XGBClassifier(**cfg)
                    clf.fit(Xtr[tr_i], ytr[tr_i])
                    yv = clf.predict_proba(Xtr[va_i])[:, 1]
                    cv.append(average_precision_score(ytr[va_i], yv))
                m = float(np.mean(cv))
                if best is None or m > best[0]:
                    best = (m, (md, mcw, cs))
            md, mcw, cs = best[1]
            cfg = dict(r2_config.XGB_CONFIG)
            cfg.update(max_depth=md, min_child_weight=mcw, colsample_bytree=cs)
            clf = xgb.XGBClassifier(**cfg)
            clf.fit(Xtr, ytr)
            ta = average_precision_score(yte, clf.predict_proba(Xte)[:, 1])
            rows.append({'dataset': ds, 'seed': seed, 'variant': name,
                         'cv_aupr': round(best[0], 4), 'cfg': f'{md}/{mcw}/{cs}',
                         'test_aupr_matched': round(ta, 4)})
            print(f'  {name:12s} cv={best[0]:.4f} cfg={md}/{mcw}/{cs} '
                  f'test={ta:.4f}', flush=True)
    out = 'results/R2/matched_selection_pilot.csv'
    df = pd.DataFrame(rows)
    if os.path.exists(out):
        df = pd.concat([pd.read_csv(out), df], ignore_index=True)
    df.to_csv(out, index=False)
    print('saved', out)


if __name__ == '__main__':
    main()
