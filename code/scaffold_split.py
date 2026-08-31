"""
R2: Bemis-Murcko scaffold-disjoint 划分 (scaffold_split)

真正的化学冷启动: 测试药物与训练药物不共享任何核心骨架.
流程:
  1. 每个药物提取 Bemis-Murcko scaffold (RDKit GetScaffoldForMol);
     无骨架的药物以自身 SMILES 作为独有 scaffold.
  2. 按 scaffold 分组, 随机选 ~20% 的 scaffold 组 → 其全部药物为 test;
     其余药物为 train (scaffold-disjoint 保证).
  3. 输出 split_manifest_scaffold.csv (同 split_manifest 格式).

用法: python code/scaffold_split.py --dataset C --random-state 42
"""
import argparse
import os
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

DATASETS = {
    'C': {'features': 'data/C-Dataset/Features/drug_features_C.csv',
          'mapping': 'data/C-Dataset/Mapping/mapping_C.csv',
          'out': 'data/C-Dataset/Splits/split_manifest_scaffold.csv',
          'id_space': 'C-index'},
    'F': {'features': 'data/F-Dataset/Features/drugs_features_df.csv',
          'mapping': 'data/F-Dataset/Mapping/mapping_F.csv',
          'out': 'data/F-Dataset/Splits/split_manifest_scaffold.csv',
          'id_space': 'string'},
    'DDCD': {'features': 'data/DDCD/Features/drugsInfo.csv',
             'mapping': 'data/DDCD/Mapping/mapping.csv',
             'out': 'data/DDCD/Splits/split_manifest_scaffold.csv',
             'id_space': 'string'},
}


def drug_to_scaffold(ds):
    cfg = DATASETS[ds]
    sm = pd.read_csv(cfg['features'])
    sm['DrugID'] = sm['DrugID'].astype(str).str.strip()
    sm['DrugSmile'] = sm['DrugSmile'].astype(str).str.strip()
    scaf = {}
    for _, r in sm.iterrows():
        m = Chem.MolFromSmiles(r['DrugSmile'])
        if m is None:
            scaf[r['DrugID']] = r['DrugSmile']
        else:
            s = MurckoScaffold.GetScaffoldForMol(m)
            scaf[r['DrugID']] = Chem.MolToSmiles(s) if s else r['DrugSmile']
    return scaf


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset', required=True, choices=['C', 'F', 'DDCD'])
    ap.add_argument('--random-state', type=int, default=42)
    ap.add_argument('--out', default=None, help='覆盖输出路径')
    args = ap.parse_args()
    ds = args.dataset
    cfg = DATASETS[ds]
    if args.out:
        cfg['out'] = args.out

    scaf = drug_to_scaffold(ds)
    df = pd.read_csv(cfg['mapping'])
    df = df[[c for c in df.columns if not str(c).startswith('Unnamed')]]
    df = df.iloc[:, :2]
    df.columns = ['drugID', 'diseaseID']
    df['drugID'] = df['drugID'].astype(str).str.strip()
    df['diseaseID'] = df['diseaseID'].astype(str).str.strip()
    df = df.drop_duplicates()

    drugs = sorted(df['drugID'].unique())
    diseases = sorted(df['diseaseID'].unique())
    scaf_of = {d: scaf.get(d, d) for d in drugs}
    scaf_groups = {}
    for d in drugs:
        scaf_groups.setdefault(scaf_of[d], []).append(d)
    scaf_list = sorted(scaf_groups.keys())
    print(f"药物: {len(drugs)} | 独有 scaffold 数: {len(scaf_list)}")

    rng = np.random.RandomState(args.random_state)
    n_test_scaf = max(1, int(len(scaf_list) * 0.2))
    test_scafs = set(rng.choice(scaf_list, n_test_scaf, replace=False))
    test_drugs = set(d for s in test_scafs for d in scaf_groups[s])
    print(f"测试 scaffold 组: {n_test_scaf}/{len(scaf_list)} | 测试药物: {len(test_drugs)}")

    # 候选空间
    all_pairs = pd.DataFrame({'drugID': drugs}).merge(
        pd.DataFrame({'diseaseID': diseases}), how='cross')
    known = set(zip(df['drugID'], df['diseaseID']))
    all_pairs['label'] = [1 if (d, s) in known else 0
                          for d, s in zip(all_pairs['drugID'], all_pairs['diseaseID'])]
    all_pairs['is_test'] = all_pairs['drugID'].isin(test_drugs)
    tr = all_pairs[~all_pairs['is_test']][['drugID', 'diseaseID', 'label']].copy()
    te = all_pairs[all_pairs['is_test']][['drugID', 'diseaseID', 'label']].copy()
    tr['split'] = 'train'
    te['split'] = 'test'

    # 审计: scaffold-disjoint + pair 零交集
    tr_scafs = set(scaf_of[d] for d in tr['drugID'].unique())
    te_scafs = set(scaf_of[d] for d in te['drugID'].unique())
    overlap = tr_scafs & te_scafs
    assert not overlap, f'scaffold 重叠: {len(overlap)}'
    tr_pairs = set(zip(tr['drugID'], tr['diseaseID']))
    te_pairs = set(zip(te['drugID'], te['diseaseID']))
    assert tr_pairs.isdisjoint(te_pairs), 'pair 重叠'
    print("  [PASS] scaffold-disjoint + pair 零交集")

    man = pd.concat([tr, te], ignore_index=True)
    print(f"train: {len(tr):,} (正{(tr['label']==1).sum():,}) | test: {len(te):,} (正{(te['label']==1).sum():,})")
    os.makedirs(os.path.dirname(cfg['out']), exist_ok=True)
    man[['drugID', 'diseaseID', 'label', 'split']].to_csv(cfg['out'], index=False)
    print(f"  ✅ {cfg['out']}")


if __name__ == '__main__':
    main()
