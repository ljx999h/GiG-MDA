# -*- coding: utf-8 -*-
"""3.7 Source-level leakage check 数据重算 (P0-1 后): C cold 4 seeds,
移除 DrugBank Conditions/Category 两模态 (--exclude), 用现 canonical 负样本,
评估 base 与 MolEmb32 的 test AUPR."""
import os
import pickle
import subprocess
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score
from sklearn.decomposition import PCA

sys.path.insert(0, 'code')
import r2_config
from cold_eval import load, load_mol_emb

D = 'C-Dataset'
seeds = [42, 7, 123, 2024]
results = []
for sd in seeds:
    mf = 'data/C-Dataset/Splits/split_manifest_cold.csv' if sd == 42 \
        else f'data/C-Dataset/Splits/split_manifest_cold_s{sd}.csv'
    score_new = f'code/results/MiRAGE_score_C_cold_exclmod_s{sd}.csv'
    if not os.path.exists(score_new):
        r = subprocess.run([sys.executable, 'code/build_mirage_features.py',
                            '--dataset', 'C', '--neighbor-source', 'r2train',
                            '--exclude', 'Conditions,Category',
                            '--manifest', mf, '--out', score_new],
                           capture_output=True, text=True, encoding='utf-8',
                           errors='replace')
        assert r.returncode == 0, r.stdout + r.stderr
    score = pd.read_csv(score_new).fillna(0.0)
    # C: int drugID -> DB
    full = pd.read_csv('data/C-Dataset/Mapping/mapping_C.csv')
    full = full[[c for c in full.columns if not str(c).startswith('Unnamed')]]
    full = full.iloc[:, :2]
    full.columns = ['drugID', 'diseaseID']
    i2d = {i: d for i, d in enumerate(
        sorted(full['drugID'].astype(str).str.strip().unique()))}
    score['drugID'] = score['drugID'].map(i2d)
    score['drugID'] = score['drugID'].astype(str).str.strip()
    score['diseaseID'] = score['diseaseID'].astype(str).str.strip()
    if 'label' in score.columns:
        score = score.drop(columns=['label'])
    mfest = pd.read_csv(mf)
    mfest['drugID'] = mfest['drugID'].astype(str).str.strip()
    mfest['diseaseID'] = mfest['diseaseID'].astype(str).str.strip()
    test = mfest[mfest['split'] == 'test'][['drugID', 'diseaseID', 'label']] \
        .merge(score, on=['drugID', 'diseaseID'], how='inner')
    assert len(test) == test['label'].notna().sum()
    # 特征列: 18 基线列剔除模态 (去掉 p/adj Conditions,Category)
    feats = [c for c in r2_config.DATASETS['C']['feats']
             if 'Conditions' not in c and 'Category' not in c]
    dirn = 'data/C-Dataset/Splits/Cold' if sd == 42 \
        else f'data/C-Dataset/Splits/Cold_s{sd}'
    pos = pd.read_csv(f'{dirn}/train_positives.csv').fillna(0.0)
    neg = pd.read_csv(f'{dirn}/train_negatives.csv').fillna(0.0)
    for df_ in (pos, neg):
        for c in ['drugID', 'diseaseID']:
            df_[c] = df_[c].astype(str).str.strip()
    train = pd.concat([pos, neg], ignore_index=True)
    ytr, yte = train['label'].values, test['label'].values
    Xb_tr = train[feats].to_numpy(dtype=np.float32)
    Xb_te = test[feats].to_numpy(dtype=np.float32)
    emb_map = load_mol_emb('C')
    uniq_d = list(pd.unique(train['drugID'].astype(str).str.strip()))
    pca = PCA(n_components=32, random_state=42)
    pca.fit(np.array([emb_map.get(d, np.zeros(768)) for d in uniq_d], dtype=np.float32))
    z = np.zeros(768, dtype=np.float32)
    tr_e = pca.transform(np.array([emb_map.get(d, z) for d in train['drugID']])).astype(np.float32)
    te_e = pca.transform(np.array([emb_map.get(d, z) for d in test['drugID']])).astype(np.float32)

    def ap(Xtr, Xte):
        clf = xgb.XGBClassifier(**r2_config.XGB_CONFIG)
        clf.fit(Xtr, ytr)
        return average_precision_score(yte, clf.predict_proba(Xte)[:, 1])

    base = ap(Xb_tr, Xb_te)
    mol = ap(np.hstack([Xb_tr, tr_e]), np.hstack([Xb_te, te_e]))
    results.append({'seed': sd, 'base_nomod': base, 'mol_nomod': mol,
                    'lift_nomod': (mol / base - 1) * 100})
    print(f'C s{sd}: base_nomod={base:.4f} mol_nomod={mol:.4f} '
          f'lift_nomod={(mol/base-1)*100:+.1f}%', flush=True)

out = pd.DataFrame(results)
out.to_csv('results/R2/modality_ablation_cold_new.csv', index=False)
print('saved results/R2/modality_ablation_cold_new.csv')
