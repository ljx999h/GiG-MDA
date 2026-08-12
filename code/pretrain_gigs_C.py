"""
C-Dataset GiGs 预训练 (无泄漏版)
 数据: 663 drugs × 409 diseases, 用 mapping80_C.csv 训练
 输出: code/model/gigs_dataC.pkl
"""
import pandas as pd
import numpy as np
import torch
import os
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from gigs_model import GiGsMatrixFactorization

# 跳过 BERT 依赖 (C-Dataset 不需要文本特征, 复用 DDCD 的 BERT cache)
# 我们直接基于整数 ID + GIPK 训练 GiGs

# 配置
MAPPING_FILE = "data/C-Dataset/Mapping/mapping80_C.csv"
OUTPUT_GIGS_DATA = "code/model/gigs_dataC.pkl"


def compute_gipk(interaction_matrix, gamma=1.0):
    dist_sq = (np.sum(interaction_matrix**2, axis=1).reshape(-1, 1) +
               np.sum(interaction_matrix**2, axis=1) -
               2 * np.dot(interaction_matrix, interaction_matrix.T))
    bandwidth = np.mean(dist_sq)
    if bandwidth == 0: bandwidth = 1.0
    return np.exp(-(gamma / bandwidth) * dist_sq)


def main():
    print("=" * 60)
    print(" GiGs 预训练 (C-Dataset, 无泄漏版)")
    print(f" 训练映射: {MAPPING_FILE}")
    print("=" * 60)

    if not os.path.exists(MAPPING_FILE):
        print(f"[FATAL] 找不到 {MAPPING_FILE}")
        return

    df = pd.read_csv(MAPPING_FILE)
    df.columns = ['drug', 'disease']
    df['drug'] = df['drug'].astype(int)
    df['disease'] = df['disease'].astype(int)

    all_drugs = sorted(df['drug'].unique().tolist())
    all_diseases = sorted(df['disease'].unique().tolist())

    drug_to_idx = {d: i for i, d in enumerate(all_drugs)}
    disease_to_idx = {d: i for i, d in enumerate(all_diseases)}
    m, n = len(all_drugs), len(all_diseases)
    print(f"  规模: {m} drugs × {n} diseases  |  关联: {len(df):,}")

    # 构建关联矩阵 A (仅用 mapping80)
    A = np.zeros((m, n))
    for _, row in df.iterrows():
        A[drug_to_idx[row['drug']], disease_to_idx[row['disease']]] = 1
    print(f"  A 矩阵密度: {A.sum()/(m*n)*100:.4f}%")

    # 相似度矩阵 (无文本特征, 仅 GIPK)
    print("  计算 GIPK 相似度...")
    Sd = compute_gipk(A)             # 药物侧
    Sv = compute_gipk(A.T)           # 疾病侧

    # 训练 GiGs
    print("  训练 GiGs 矩阵分解...")
    gigs = GiGsMatrixFactorization(
        k=64, lambda1=0.1, lambda2=0.01, lambda3=0.1, max_iter=150
    )
    X, Y = gigs.fit(A, Sd, Sv)

    if np.allclose(X, 0):
        print("  [WARN] X 全 0, 请调参")

    # 保存
    data_to_save = {
        "X": X, "Y": Y,
        "drug_to_idx": drug_to_idx,
        "disease_to_idx": disease_to_idx
    }
    os.makedirs(os.path.dirname(OUTPUT_GIGS_DATA), exist_ok=True)
    with open(OUTPUT_GIGS_DATA, "wb") as f:
        pickle.dump(data_to_save, f)
    print(f"\n  ✅ 完成! {OUTPUT_GIGS_DATA}")
    print(f"  X: {X.shape}, Y: {Y.shape}")


if __name__ == "__main__":
    main()
