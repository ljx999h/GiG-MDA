# -*- coding: utf-8 -*-
"""验证 tab:ablation 的 MiRAGE+dot cold lift (C -42.3%, DDCD -4.1%, mean over 4 seeds).
对 C/DDCD x seeds {42,7,123,2024}: 训练 base 与 base+dot, 算 cold AUPR 与相对 lift.
输出: results/cold_dot_results.csv (dataset, seed, base_aupr, dot_aupr, lift)
"""
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score

sys.path.insert(0, 'code')
import r2_config
from cold_eval import load, gigs_feats

rows = []
for ds in ['C', 'DDCD']:
    for seed in [42, 7, 123, 2024]:
        train, test, gigs, feats, base = load(ds, seed, 'cold-drug')
        tr_dot, _ = gigs_feats(train, gigs)
        te_dot, _ = gigs_feats(test, gigs)
        ytr, yte = train['label'].values, test['label'].values
        for name, Xtr, Xte in [
            ('base', train[base].values.astype(np.float32),
             test[base].values.astype(np.float32)),
            ('dot', np.hstack([train[base].values, tr_dot[:, None]]).astype(np.float32),
             np.hstack([test[base].values, te_dot[:, None]]).astype(np.float32)),
        ]:
            clf = xgb.XGBClassifier(**r2_config.XGB_CONFIG)
            clf.fit(Xtr, ytr)
            aupr = average_precision_score(yte, clf.predict_proba(Xte)[:, 1])
            rows.append({'dataset': ds, 'seed': seed, 'variant': name, 'aupr': aupr})
        print(f'{ds}/{seed} done', flush=True)

df = pd.DataFrame(rows)
df.to_csv('results/cold_dot_results.csv', index=False)
piv = df.pivot_table(index=['dataset', 'seed'], columns='variant', values='aupr').reset_index()
piv['lift_pct'] = (piv['dot'] - piv['base']) / piv['base'] * 100
print('\nper-seed lift:')
print(piv[['dataset', 'seed', 'lift_pct']].to_string(index=False))
print('\nmean lift per dataset:')
print(piv.groupby('dataset')['lift_pct'].mean().to_string())
