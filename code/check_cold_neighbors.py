# -*- coding: utf-8 -*-
"""
Task: P1-5 F 数据集无增益的实证核查之一:
对每个数据集 (C/F/DDCD) × 4 个冷分割种子, 计算每个冷药物的
结构最近邻 (max Tanimoto to any training drug), 比较稀疏度.
若 F 的冷药物结构邻居显著更稀疏, 可支撑"F 冷药物结构近邻稀疏"
这一可检验假设; 否则排除该解释.
"""
import numpy as np
import pandas as pd

BASE = {'C': 'data/C-Dataset/Splits', 'F': 'data/F-Dataset/Splits', 'DDCD': 'data/DDCD/Splits'}
SIMS = {
    'C': 'data/C-Dataset/SimilarityMatrices/SMILE_similarity_C.csv',
    'F': 'data/F-Dataset/SimilarityMatrices/SMILE_similarity_F.csv',
    'DDCD': 'data/DDCD/SimilarityMatrices/drugSmile_tanimoto',
}
SEEDS = [42, 7, 123, 2024]

for ds in ['C', 'F', 'DDCD']:
    sim = pd.read_csv(SIMS[ds], index_col=0)
    sim.index = sim.index.astype(str).str.strip()
    sim.columns = sim.columns.astype(str).str.strip()
    n_cold, n_missing = 0, 0
    per_seed = []
    for s in SEEDS:
        tag = 'split_manifest_cold.csv' if s == 42 else f'split_manifest_cold_s{s}.csv'
        m = pd.read_csv(f'{BASE[ds]}/{tag}', dtype={'drugID': str})
        m['drugID'] = m['drugID'].str.strip()
        train_d = set(m.loc[m['split'] == 'train', 'drugID'])
        test_d = set(m.loc[m['split'] == 'test', 'drugID'])
        cold = sorted(test_d - train_d)
        n_cold += len(cold)
        smax = []
        for d in cold:
            if d not in sim.index:
                n_missing += 1
                continue
            row = sim.loc[d]
            tr = [c for c in sim.columns if c in train_d]
            smax.append(row[tr].max() if len(tr) else np.nan)
        smax = [x for x in smax if not np.isnan(x)]
        per_seed.append(np.mean(smax) if smax else np.nan)
    print(f'\n=== {ds} ===')
    print(f'  cold drugs total (4 seeds): {n_cold}, missing SMILES in matrix: {n_missing}')
    print(f'  mean max-sim per seed: {np.round(per_seed, 3)}')
    print(f'  overall mean max-sim: {np.nanmean(per_seed):.4f}')
