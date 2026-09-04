# -*- coding: utf-8 -*-
"""
案例研究 (导师三表方案): 部署模型 novel-prediction 文献比对
============================================================
论文 3.9 节案例分析. 三个维度:
  T1 全局:   所有 disease-drug 对按预测分排序, 去掉全部已知关联 → top-15 文献比对
  T2 固定药物 (默认 Dexamethasone, DDCD therapeutic 关联最多的处方药)
             → 该药 × 所有疾病, 去掉已知 → top-15 文献比对
  T3 固定疾病 (默认 Stomach Neoplasms 胃癌) → 去掉已知药物 → top-15 文献比对

模型口径 = 部署模型 (协议验证后的最终部署):
  - 已知集 = --mapping 指定的映射文件 (默认全量 42,200; therapeutic 子集传
    data/DDCD/Mapping/mapping_therapeutic.csv → 构建器需以同文件跑 --neighbor-source full)
  - 变体: MiRAGE + MolEmb32; XGBoost 标准配置; 已知 + 等量均匀随机负样本
  - 打分: 网格内全部 label=0 候选
输出 (results/): case_study_deploy_top15_{global,drug,gastric}.csv 等
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
from cold_eval import load_mol_emb

ap = argparse.ArgumentParser()
ap.add_argument('--feat-file', default='code/results/MiRAGE_score_DDCD_deploy.csv')
ap.add_argument('--drug', default='DB01234', help='固定药物 DrugBank ID (默认 Dexamethasone)')
ap.add_argument('--disease', default='MESH:D013274', help='固定疾病 MeSH ID (默认 Stomach Neoplasms)')
args = ap.parse_args()
FEAT_FILE, DRUG, DISEASE = args.feat_file, args.drug, args.disease
OUT = 'results'
RANDOM_STATE = 0

drugs = pd.read_csv('data/DDCD/Features/drugsInfo.csv',
                    usecols=['DrugID', 'DrugName']).astype(str)
diseases = pd.read_csv('data/DDCD/Features/diseasesInfo.csv',
                       usecols=['DiseaseID', 'DiseaseName']).astype(str)
DRUG_NAME = drugs.loc[drugs['DrugID'] == DRUG, 'DrugName'].iloc[0]
DISEASE_NAME = diseases.loc[diseases['DiseaseID'] == DISEASE, 'DiseaseName'].iloc[0]
print(f'固定药物: {DRUG} {DRUG_NAME} | 固定疾病: {DISEASE} {DISEASE_NAME}')

print(f'\n加载部署特征: {FEAT_FILE} ...')
df = pd.read_csv(FEAT_FILE)
df['drugID'] = df['drugID'].astype(str).str.strip()
df['diseaseID'] = df['diseaseID'].astype(str).str.strip()
feats = r2_config.DATASETS['DDCD']['feats']
print(f'  行数 {len(df):,} | 特征 {len(feats)} | 已知对 {int(df.label.sum()):,} | novel {(df.label == 0).sum():,}')

# ---- MolEmb32: PCA-32 fit 在全部 drug 上 (部署场景所有药物都有训练关联) ----
emb_map = load_mol_emb('DDCD')
all_drug_ids = pd.unique(df['drugID'])
pca = PCA(n_components=32, random_state=42)
raw_all = np.array([emb_map.get(d, np.zeros(768)) for d in all_drug_ids], dtype=np.float32)
pca.fit(raw_all)
emb_unique = pca.transform(raw_all).astype(np.float32)
emb32_by_id = dict(zip(all_drug_ids, emb_unique))
print(f'  PCA-32 拟合于 {len(all_drug_ids)} 个药物嵌入')

emb32 = np.array([emb32_by_id.get(d, np.zeros(32, dtype=np.float32))
                  for d in df['drugID']], dtype=np.float32)
print('  MolEmb32 映射完成')

base = df[feats].to_numpy(dtype=np.float32)
X = np.hstack([base, emb32]).astype(np.float32)
y = df['label'].to_numpy()

# ---- 训练: 全部已知 + 等量均匀随机 novel 负样本 ----
pos_idx = np.flatnonzero(y == 1)
novel_idx = np.flatnonzero(y == 0)
rng = np.random.RandomState(42)
neg_idx = rng.choice(novel_idx, size=len(pos_idx), replace=False)
tr_idx = np.concatenate([pos_idx, neg_idx])
Xtr, ytr = X[tr_idx], y[tr_idx]
print(f'\n训练: {len(pos_idx):,} 已知 + {len(neg_idx):,} 均匀负样本 = {len(tr_idx):,} 对 x {Xtr.shape[1]} 维')

clf = xgb.XGBClassifier(**r2_config.XGB_CONFIG)
clf.fit(Xtr, ytr)
print('  训练完成')

# ---- 打分全部 novel 候选 (分块) ----
scores = np.zeros(len(novel_idx), dtype=np.float32)
CH = 400_000
for i in range(0, len(novel_idx), CH):
    s = slice(i, min(i + CH, len(novel_idx)))
    scores[s] = clf.predict_proba(X[novel_idx[s]])[:, 1].astype(np.float32)
    print(f'  打分 {i + 1:,} - {min(i + CH, len(novel_idx)):,} / {len(novel_idx):,}')
del X
novel = df.iloc[novel_idx][['drugID', 'diseaseID']].copy().reset_index(drop=True)
novel['score'] = scores
print(f'novel 分数: min {novel.score.min():.4f} | median {novel.score.median():.4f} | max {novel.score.max():.4f}')

novel_sorted = novel.sort_values('score', ascending=False).reset_index(drop=True)
novel_sorted.to_csv(f'{OUT}/case_study_deploy_full_ranking.csv', index=False)


def merge_names(t):
    t = t.merge(drugs, left_on='drugID', right_on='DrugID', how='left')
    t = t.merge(diseases, left_on='diseaseID', right_on='DiseaseID', how='left')
    return t[['drugID', 'DrugName', 'diseaseID', 'DiseaseName', 'score']]


pools = {
    'global': novel_sorted.head(15),
    'drug': novel_sorted[novel_sorted['drugID'] == DRUG].head(15),
    'gastric': novel_sorted[novel_sorted['diseaseID'] == DISEASE].head(15),
}
controls = {
    'global': novel_sorted.sample(15, random_state=RANDOM_STATE),
    'drug': novel_sorted[novel_sorted['drugID'] == DRUG].sample(15, random_state=RANDOM_STATE),
    'gastric': novel_sorted[novel_sorted['diseaseID'] == DISEASE].sample(15, random_state=RANDOM_STATE),
}

summary = {}
for tag, pool in pools.items():
    t = merge_names(pool.copy()).reset_index(drop=True)
    t.to_csv(f'{OUT}/case_study_deploy_top15_{tag}.csv', index=False)
    c = merge_names(controls[tag].copy()).reset_index(drop=True)
    c.to_csv(f'{OUT}/case_study_deploy_random15_{tag}.csv', index=False)
    pool_size = len(novel_sorted) if tag == 'global' else \
        int((novel_sorted['drugID'] == DRUG).sum()) if tag == 'drug' else \
        int((novel_sorted['diseaseID'] == DISEASE).sum())
    summary[tag] = {'pool_size': pool_size,
                    'top15_score_range': [float(t.score.min()), float(t.score.max())]}
    print(f'\n===== top-15 [{tag}] 池={pool_size:,} =====')
    for i, r in t.iterrows():
        print(f'{i + 1:2d}. {r.DrugName:24s} | {r.DiseaseName:45s} | {r.score:.4f}')
    print(f'--- 随机对照 15 [{tag}] ---')
    for i, r in c.iterrows():
        print(f'{i + 1:2d}. {r.DrugName:24s} | {r.DiseaseName:45s} | {r.score:.4f}')

summary['meta'] = {'drug': DRUG, 'drug_name': DRUG_NAME, 'disease': DISEASE,
                   'disease_name': DISEASE_NAME, 'n_pos_train': int(y.sum()),
                   'n_candidates_total': int(len(df)),
                   'n_novel': int(len(novel)), 'model': 'MiRAGE+MolEmb32 deploy'}
with open(f'{OUT}/case_study_deploy_summary.json', 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=1, ensure_ascii=False)
print(f'\n已保存: {OUT}/case_study_deploy_*.csv + summary.json')
