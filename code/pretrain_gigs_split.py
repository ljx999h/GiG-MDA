"""
R2 阶段2: 在 P_train 上重新训练 GRMF 嵌入 (fold-local, 方案B)

方案B要求 GRMF 嵌入只用训练关联(P_train)训练, 测试关联完全隔离.
本脚本读 split_manifest 的 train 正样本, 在 P_train 上训练 GiGsMatrixFactorization,
输出 fold-local 嵌入 pkl.

超参数: 与 pretrain_gigs_C/F.py 实际运行一致 (k=64, lambda1=0.1, lambda2=0.01, lambda3=0.1, max_iter=150)
ID: 统一为字符串 (drugID, diseaseID), 与 manifest / MiRAGE_score 对齐.

输出: data/{DS}/Splits/gigs_split_{DS}.pkl
  含 X, Y, drug_to_idx, disease_to_idx
"""
import argparse
import os
import pickle
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gigs_model import GiGsMatrixFactorization

DATASETS = {
    'C': {'manifest': 'data/C-Dataset/Splits/split_manifest.csv',
          'out': 'data/C-Dataset/Splits/gigs_split_C.pkl'},
    'F': {'manifest': 'data/F-Dataset/Splits/split_manifest.csv',
          'out': 'data/F-Dataset/Splits/gigs_split_F.pkl'},
    'DDCD': {'manifest': 'data/DDCD/Splits/split_manifest.csv',
             'out': 'data/DDCD/Splits/gigs_split_DDCD.pkl'},
}

K_DIM = 64
L1, L2, L3 = 0.1, 0.01, 0.1
MAX_ITER = 150


def gipk(interaction_matrix, gamma=1.0):
    """GIPK 相似度 (与 pretrain_gigs_C/F.py 一致)."""
    dist_sq = (np.sum(interaction_matrix**2, axis=1).reshape(-1, 1) +
               np.sum(interaction_matrix**2, axis=1) -
               2 * np.dot(interaction_matrix, interaction_matrix.T))
    bandwidth = np.mean(dist_sq)
    if bandwidth == 0:
        bandwidth = 1.0
    return np.exp(-(gamma / bandwidth) * dist_sq)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, choices=['C', 'F', 'DDCD'])
    parser.add_argument('--random-state', type=int, default=42,
                        help='X/Y 初始化种子 (多种子实验用, 与划分/OOF 种子一致)')
    parser.add_argument('--manifest', default=None,
                        help='覆盖 split_manifest 路径 (冷启动划分用)')
    parser.add_argument('--out', default=None, help='覆盖嵌入输出路径 (冷启动用)')
    parser.add_argument('--k', type=int, default=K_DIM)
    parser.add_argument('--l1', type=float, default=L1)
    parser.add_argument('--l2', type=float, default=L2)
    parser.add_argument('--l3', type=float, default=L3)
    args = parser.parse_args()
    cfg = DATASETS[args.dataset]
    if args.manifest:
        cfg['manifest'] = args.manifest
    if args.out:
        cfg['out'] = args.out

    print(f"\n{'='*60}\n GRMF 在 P_train 上重训: {args.dataset} (seed={args.random_state})\n{'='*60}")

    mf = pd.read_csv(cfg['manifest'])
    mf['drugID'] = mf['drugID'].astype(str).str.strip()
    mf['diseaseID'] = mf['diseaseID'].astype(str).str.strip()
    pos_train = mf[(mf['split'] == 'train') & (mf['label'] == 1)]
    print(f"P_train 正关联: {len(pos_train):,}")

    # 实体: 用全候选空间 (train+test 所有实体), 保证 test 实体有嵌入定义
    drugs = sorted(mf['drugID'].unique())
    diseases = sorted(mf['diseaseID'].unique())
    drug_to_idx = {d: i for i, d in enumerate(drugs)}
    disease_to_idx = {d: i for i, d in enumerate(diseases)}
    m, n = len(drugs), len(diseases)
    print(f"实体(全候选): {m} drugs × {n} diseases")

    # 关联矩阵 A (只用 P_train)
    A = np.zeros((m, n))
    for _, row in pos_train.iterrows():
        A[drug_to_idx[row['drugID']], disease_to_idx[row['diseaseID']]] = 1
    print(f"A 密度: {A.sum()/(m*n):.5%}")

    # GIPK 相似度
    print("  计算 GIPK...")
    Sd = gipk(A)
    Sv = gipk(A.T)

    # 训练 GRMF
    print(f"  训练 GiGsMatrixFactorization (k={args.k}, λ=({args.l1},{args.l2},{args.l3}), max_iter={MAX_ITER}, seed={args.random_state})...")
    gigs = GiGsMatrixFactorization(k=args.k, lambda1=args.l1, lambda2=args.l2,
                                   lambda3=args.l3, max_iter=MAX_ITER)
    X, Y = gigs.fit(A, Sd, Sv, random_state=args.random_state)

    # 保存 (字符串 ID key, 与 manifest 对齐)
    data = {
        'X': X, 'Y': Y,
        'drug_to_idx': drug_to_idx,
        'disease_to_idx': disease_to_idx,
    }
    os.makedirs(os.path.dirname(cfg['out']), exist_ok=True)
    with open(cfg['out'], 'wb') as f:
        pickle.dump(data, f)
    print(f"  ✅ 嵌入已保存: {cfg['out']}")
    print(f"  X: {X.shape}, Y: {Y.shape}")


if __name__ == '__main__':
    main()
