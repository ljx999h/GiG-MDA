# -*- coding: utf-8 -*-
"""
案例研究 (路线1): 冷药物排名 + 文献支持率富集
在 DDCD cold split (seed 42) 上训练 MiRAGE+MolEmb32 冷模型 (--variant mol, 默认)
或 base MiRAGE 模型 (--variant base, 对照), 对全部冷药物候选对排序. 输出:
  results/case_study_cold_top30_{variant}.csv   top-30 对 (含药物/疾病名)
  results/case_study_cold_random30_{variant}.csv  随机 30 对 (同候选池, 同协议)
  results/case_study_cold_summary_{variant}.json  富集统计
证据检索由 case_study_crossref.py 执行 (Crossref works API).
"""
import argparse
import json
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.decomposition import PCA

sys.path.insert(0, 'code')
import r2_config
from cold_eval import load, load_mol_emb

DS, SEED, N = 'DDCD', 42, 30
OUT = 'results'

ap = argparse.ArgumentParser()
ap.add_argument('--variant', default='mol', choices=['base', 'mol'])
args = ap.parse_args()
VARIANT = args.variant

print(f'[{DS} seed {SEED}] loading cold-split data ...')
train, test, gigs, feats, base = load(DS, SEED, 'cold-drug')

if VARIANT == 'mol':
    print(f'[{DS}] building MolEmb32 (PCA-32 on unique training drugs) ...')
    emb_map = load_mol_emb(DS)
    pca = PCA(n_components=32, random_state=42)
    uniq_d = pd.unique(train['drugID'].astype(str).str.strip())
    uniq_raw = np.array([emb_map.get(d, np.zeros(768)) for d in uniq_d], dtype=np.float32)
    pca.fit(uniq_raw)
    tr_raw = np.array([emb_map.get(d, np.zeros(768)) for d in train['drugID']], dtype=np.float32)
    te_raw = np.array([emb_map.get(d, np.zeros(768)) for d in test['drugID']], dtype=np.float32)
    tr_pca = pca.transform(tr_raw).astype(np.float32)
    te_pca = pca.transform(te_raw).astype(np.float32)
    Xtr = np.hstack([train[base].values, tr_pca]).astype(np.float32)
    Xte = np.hstack([test[base].values, te_pca]).astype(np.float32)
    NAME = 'MiRAGE+MolEmb32'
else:
    Xtr = train[base].values.astype(np.float32)
    Xte = test[base].values.astype(np.float32)
    NAME = 'MiRAGE (base)'

ytr = train['label'].values
print(f'[{DS}] training {NAME} on {len(Xtr):,} pairs x {Xtr.shape[1]} feats ...')
clf = xgb.XGBClassifier(**r2_config.XGB_CONFIG)
clf.fit(Xtr, ytr)

print(f'[{DS}] scoring {len(Xte):,} cold candidates ...')
yp = clf.predict_proba(Xte)[:, 1]

test = test.copy()
test['score'] = yp
test = test.sort_values('score', ascending=False).reset_index(drop=True)

top = test.head(N).copy()
rng = np.random.RandomState(0)
rand = test.sample(N, random_state=0).copy()

drugs = pd.read_csv('data/DDCD/Features/drugsInfo.csv',
                    usecols=['DrugID', 'DrugName']).astype(str)
diseases = pd.read_csv('data/DDCD/Features/diseasesInfo.csv',
                       usecols=['DiseaseID', 'DiseaseName']).astype(str)
for df_ in (top, rand):
    df_['drugID'] = df_['drugID'].astype(str).str.strip()
    df_['diseaseID'] = df_['diseaseID'].astype(str).str.strip()

top = top.merge(drugs, left_on='drugID', right_on='DrugID', how='left')
top = top.merge(diseases, left_on='diseaseID', right_on='DiseaseID', how='left')
rand = rand.merge(drugs, left_on='drugID', right_on='DrugID', how='left')
rand = rand.merge(diseases, left_on='diseaseID', right_on='DiseaseID', how='left')

cols = ['drugID', 'DrugName', 'diseaseID', 'DiseaseName', 'label', 'score']
top[cols].to_csv(f'{OUT}/case_study_cold_top30_{VARIANT}.csv', index=False)
rand[cols].to_csv(f'{OUT}/case_study_cold_random30_{VARIANT}.csv', index=False)
test[['drugID', 'diseaseID', 'score']].to_csv(
    f'{OUT}/case_study_full_ranking_{VARIANT}.csv', index=False)

p_top = top['label'].mean()
p_rand = rand['label'].mean()
n_pos_test = int(test['label'].sum())
summary = {
    'dataset': DS, 'seed': SEED, 'n': N, 'variant': VARIANT,
    'precision_at_N_top': float(p_top),
    'precision_at_N_random': float(p_rand),
    'positive_rate_test': float(test['label'].mean()),
    'n_positives_test': n_pos_test,
    'top_hits': int(top['label'].sum()),
    'random_hits': int(rand['label'].sum()),
}
with open(f'{OUT}/case_study_cold_summary_{VARIANT}.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
print('\n--- top-30 ---')
for i, r in top.iterrows():
    print(f'{top.index.get_loc(i)+1:2d}. {r["DrugName"]} | {r["DiseaseName"]} | label={r["label"]} | score={r["score"]:.4f}')
print('\n--- random-30 ---')
for i, r in rand.iterrows():
    print(f'{rand.index.get_loc(i)+1:2d}. {r["DrugName"]} | {r["DiseaseName"]} | label={r["label"]} | score={r["score"]:.4f}')
