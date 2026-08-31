"""
R2 冷启动划分 (cold_split)

--mode cold-drug: 随机选 20% 药物为"冷药物", 其所有候选对进 test;
                 train = 热药物 × 全疾病候选对 (正样本 = 热药物已知关联)
输出: data/{DS}/Splits/split_manifest_cold.csv (与 split_manifest.csv 同格式)

关键性质:
  - train/test pair 零交集 (热药物对 vs 冷药物对天然不相交)
  - 冷药物在 train 中零关联 → GBA 邻居空 / GRMF 嵌入退化 → 只有
    SMILES 分子表征仍提供信号 (方案A 冷启动卖点的评估场景)
  - 评估: 方案B (train 内 CV 阈值 + 冷 test 只跑一次)

用法:
  python code/cold_split.py --dataset C --mode cold-drug --cold-ratio 0.2 --random-state 42
"""
import argparse
import os
import pandas as pd

DATASETS = {
    'C': {'mapping': 'data/C-Dataset/Mapping/mapping_C.csv',
          'out': 'data/C-Dataset/Splits/split_manifest_cold.csv',
          'id_mode': 'C-index'},
    'F': {'mapping': 'data/F-Dataset/Mapping/mapping_F.csv',
          'out': 'data/F-Dataset/Splits/split_manifest_cold.csv',
          'id_mode': 'string'},
    'DDCD': {'mapping': 'data/DDCD/Mapping/mapping.csv',
             'out': 'data/DDCD/Splits/split_manifest_cold.csv',
             'id_mode': 'string'},
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset', required=True, choices=['C', 'F', 'DDCD'])
    ap.add_argument('--mode', default='cold-drug', choices=['cold-drug', 'cold-disease'])
    ap.add_argument('--cold-ratio', type=float, default=0.2)
    ap.add_argument('--random-state', type=int, default=42)
    ap.add_argument('--out', default=None, help='覆盖输出路径 (多 seed 用 _s{seed} 后缀)')
    args = ap.parse_args()
    cfg = DATASETS[args.dataset]
    if args.out:
        cfg['out'] = args.out

    df = pd.read_csv(cfg['mapping'])
    df = df[[c for c in df.columns if not str(c).startswith('Unnamed')]]
    df = df.iloc[:, :2]
    df.columns = ['drugID', 'diseaseID']
    df['drugID'] = df['drugID'].astype(str).str.strip()
    df['diseaseID'] = df['diseaseID'].astype(str).str.strip()
    df = df.drop_duplicates()
    pos = df
    print(f"已知关联(正): {len(pos):,}")

    drugs = sorted(pos['drugID'].unique())
    diseases = sorted(pos['diseaseID'].unique())
    import numpy as np
    rng = np.random.RandomState(args.random_state)

    # 候选空间
    all_pairs = pd.DataFrame({'drugID': drugs}).merge(
        pd.DataFrame({'diseaseID': diseases}), how='cross')
    known = set(zip(pos['drugID'], pos['diseaseID']))
    all_pairs['label'] = [1 if (d, s) in known else 0
                          for d, s in zip(all_pairs['drugID'], all_pairs['diseaseID'])]

    if args.mode == 'cold-drug':
        n_cold = max(1, int(len(drugs) * args.cold_ratio))
        cold_drugs = set(rng.choice(drugs, n_cold, replace=False))
        all_pairs['is_cold'] = all_pairs['drugID'].isin(cold_drugs)
        print(f"冷药物: {len(cold_drugs)}/{len(drugs)} ({len(cold_drugs)/len(drugs):.1%})")
    else:  # cold-disease
        n_cold = max(1, int(len(diseases) * args.cold_ratio))
        cold_diseases = set(rng.choice(diseases, n_cold, replace=False))
        all_pairs['is_cold'] = all_pairs['diseaseID'].isin(cold_diseases)
        print(f"冷疾病: {len(cold_diseases)}/{len(diseases)} ({len(cold_diseases)/len(diseases):.1%})")

    tr = all_pairs[~all_pairs['is_cold']][['drugID', 'diseaseID', 'label']].copy()
    te = all_pairs[all_pairs['is_cold']][['drugID', 'diseaseID', 'label']].copy()
    tr['split'] = 'train'
    te['split'] = 'test'
    man = pd.concat([tr, te], ignore_index=True)

    # 审计: 零交集 + 冷实体在 train 中零关联
    tr_pairs = set(zip(tr['drugID'], tr['diseaseID']))
    te_pairs = set(zip(te['drugID'], te['diseaseID']))
    assert tr_pairs.isdisjoint(te_pairs), 'train/test pair 重叠!'
    if args.mode == 'cold-drug':
        tr_pos = tr[tr['label'] == 1]
        overlap = set(tr_pos['drugID']) & cold_drugs
        assert len(overlap) == 0, f'冷药物在 train 中有关联: {overlap}'
    print(f"  [PASS] pair 零交集; 冷实体在 train 零关联")

    print(f"train: {len(tr):,} (正{(tr['label']==1).sum():,} 负{(tr['label']==0).sum():,})")
    print(f"test : {len(te):,} (正{(te['label']==1).sum():,} 负{(te['label']==0).sum():,})")

    os.makedirs(os.path.dirname(cfg['out']), exist_ok=True)
    man[['drugID', 'diseaseID', 'label', 'split']].to_csv(cfg['out'], index=False)
    print(f"  ✅ {cfg['out']}")


if __name__ == '__main__':
    main()
