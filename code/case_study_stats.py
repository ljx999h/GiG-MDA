# -*- coding: utf-8 -*-
"""案例研究统计: base vs mol 头部对照, Fisher 检验, 剔除主导药物, base 排名列"""
import json

import pandas as pd
from scipy.stats import fisher_exact, binomtest

mol = pd.read_csv('results/case_study_cold_top30_mol.csv')
base = pd.read_csv('results/case_study_cold_top30_base.csv')
rnd_mol = pd.read_csv('results/case_study_cold_random30_mol.csv')
rnd_base = pd.read_csv('results/case_study_cold_random30_base.csv')

# --- 随机对照是否同一批对 ---
same = set(zip(rnd_mol.drugID, rnd_mol.diseaseID)) == set(zip(rnd_base.drugID, rnd_base.diseaseID))
print('random-30 same pair set (mol vs base):', same)

# --- 头部 verified 恢复 ---
v_mol, v_base = int(mol.label.sum()), int(base.label.sum())
print(f'\nverified in top-30: mol {v_mol}/30 ({v_mol/30:.1%}) | base {v_base}/30 ({v_base/30:.1%})')
print('pool positive rate: 0.0189')

# --- Fisher / 二项检验 ---
t = fisher_exact([[v_mol, 30 - v_mol], [v_base, 30 - v_base]])
print(f'mol vs base Fisher: OR={t.statistic:.2f} p={t.pvalue:.3f}')
t2 = fisher_exact([[v_mol, 30 - v_mol], [0, 30]])
print(f'mol vs random-30 control Fisher: OR={t2.statistic:.1f} p={t2.pvalue:.2e}')
b = binomtest(v_mol, 30, 0.0189)
print(f'mol vs pool rate binomial: p={b.pvalue:.2e}')

# --- 文献富集 (novel pairs only) ---
ev_mol = pd.read_csv('results/case_study_evidence_v2_top30.csv')
ev_rnd = pd.read_csv('results/case_study_evidence_v2_random30.csv')
nov_mol = ev_mol[ev_mol.label == 0]
d_nov = (nov_mol.evidence == 'direct').sum()
d_rnd = (ev_rnd.evidence == 'direct').sum()
print(f'\nnovel-pair literature direct: mol {d_nov}/{len(nov_mol)} ({d_nov/len(nov_mol):.1%}) '
      f'| control {d_rnd}/30 ({d_rnd/30:.1%})')
t3 = fisher_exact([[d_nov, len(nov_mol) - d_nov], [d_rnd, 30 - d_rnd]])
print(f'Fisher: OR={t3.statistic:.2f} p={t3.pvalue:.3f}')

# --- 剔除主导药物 (mol top-30) ---
abx = {'Ceftriaxone', 'Netilmicin'}
n_cfx = (mol.DrugName == 'Ceftriaxone').sum()
print(f'\nceftriaxone pairs in mol top-30: {n_cfx}')
rem = mol[mol.DrugName != 'Ceftriaxone']
print(f'without ceftriaxone: {int(rem.label.sum())}/{len(rem)} ({rem.label.mean():.1%})')
rem2 = mol[~mol.DrugName.isin(abx)]
print(f'without both antibiotics: {int(rem2.label.sum())}/{len(rem2)} ({rem2.label.mean():.1%})')

# --- base 排名列 (mol top-30 每对在 base 全排序中的位置) ---
base_full = pd.read_csv('results/case_study_full_ranking_base.csv')
base_full = base_full.sort_values('score', ascending=False).reset_index(drop=True)
base_full['drugID'] = base_full['drugID'].astype(str).str.strip()
base_full['diseaseID'] = base_full['diseaseID'].astype(str).str.strip()
rank_map = {tuple(r[['drugID', 'diseaseID']]): i + 1 for i, r in base_full.iterrows()}
mol2 = mol.copy()
mol2['drugID'] = mol2['drugID'].astype(str).str.strip()
mol2['diseaseID'] = mol2['diseaseID'].astype(str).str.strip()
mol2['base_rank'] = [rank_map.get((d, s)) for d, s in zip(mol2.drugID, mol2.diseaseID)]
assert mol2.base_rank.notna().all(), 'some mol top-30 pairs missing from base ranking'
mol2.to_csv('results/case_study_top30_with_base_rank.csv', index=False)
print('\nbase ranks of mol top-30 (min/med/max):',
      int(mol2.base_rank.min()), int(mol2.base_rank.median()), int(mol2.base_rank.max()))
print('mol pairs within base top-100:', int((mol2.base_rank <= 100).sum()), '/ 30')
print('mol pairs within base top-1000:', int((mol2.base_rank <= 1000).sum()), '/ 30')

summary = {
    'verified_top30': {'mol': v_mol, 'base': v_base, 'pool_rate': 0.0189,
                       'fisher_mol_vs_base_p': round(t.pvalue, 4),
                       'fisher_mol_vs_control_p': t2.pvalue,
                       'binomial_vs_pool_p': b.pvalue},
    'novel_literature': {'mol_direct': int(d_nov), 'mol_novel': int(len(nov_mol)),
                         'control_direct': int(d_rnd), 'control_n': 30,
                         'fisher_p': round(t3.pvalue, 4)},
    'concentration': {'ceftriaxone_pairs': int(n_cfx),
                      'verified_without_ceftriaxone': [int(rem.label.sum()), int(len(rem))],
                      'verified_without_antibiotics': [int(rem2.label.sum()), int(len(rem2))]},
}
with open('results/case_study_stats.json', 'w') as f:
    json.dump(summary, f, indent=2)
print('\n-> results/case_study_stats.json')
