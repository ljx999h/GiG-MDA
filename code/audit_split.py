"""
R2 阶段1: 划分审计 (audit_split)

对 split_manifest.csv 做自动审计:
  1. pair 重叠检查: train vs test 必须零交集 (失败即 exit 非0)
  2. 实体重叠统计: drug/disease 在 train/test 的共享程度
  3. 正负分布统计
  4. 文件哈希 (可复现性)

用法: python audit_split.py --dataset C
"""
import argparse
import hashlib
import os
import sys
import pandas as pd

DATASETS = {
    'C':   {'manifest': 'data/C-Dataset/Splits/split_manifest.csv'},
    'F':   {'manifest': 'data/F-Dataset/Splits/split_manifest.csv'},
    'DDCD': {'manifest': 'data/DDCD/Splits/split_manifest.csv'},
}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, choices=['C', 'F', 'DDCD'])
    args = parser.parse_args()

    manifest_path = DATASETS[args.dataset]['manifest']
    if not os.path.exists(manifest_path):
        print(f"[FAIL] 找不到 {manifest_path}, 请先运行 data_split.py")
        sys.exit(1)

    print(f"\n{'='*60}\n 划分审计: {args.dataset}\n{'='*60}")
    df = pd.read_csv(manifest_path)
    df['drugID'] = df['drugID'].astype(str).str.strip()
    df['diseaseID'] = df['diseaseID'].astype(str).str.strip()

    # 1. 基础统计
    train = df[df['split'] == 'train']
    test = df[df['split'] == 'test']
    print(f"总候选: {len(df):,} | train: {len(train):,} | test: {len(test):,}")
    print(f"正样本: train {len(train[train['label']==1]):,} / test {len(test[test['label']==1]):,}")

    # 2. pair 重叠检查 (核心)
    def pair_set(d):
        return set(zip(d['drugID'], d['diseaseID']))
    train_pairs = pair_set(train)
    test_pairs = pair_set(test)
    overlap = train_pairs & test_pairs
    if overlap:
        print(f"[FAIL] ❌ train/test 有 {len(overlap):,} 个重叠 pair!")
        print("  示例:", list(overlap)[:5])
        sys.exit(1)
    print(f"[PASS] ✅ train/test pair 零交集")

    # 3. 实体重叠统计 (信息性, 非失败条件)
    train_drugs = set(train['drugID']); test_drugs = set(test['drugID'])
    train_dis = set(train['diseaseID']); test_dis = set(test['diseaseID'])
    drug_shared = len(train_drugs & test_drugs) / max(1, len(train_drugs | test_drugs))
    dis_shared = len(train_dis & test_dis) / max(1, len(train_dis | test_dis))
    print(f"[INFO] drug 实体重叠率: {drug_shared:.1%} ({len(train_drugs & test_drugs)}/{len(train_drugs | test_drugs)})")
    print(f"[INFO] disease 实体重叠率: {dis_shared:.1%} ({len(train_dis & test_dis)}/{len(train_dis | test_dis)})")

    # 4. 文件哈希
    h = sha256_file(manifest_path)
    print(f"[INFO] manifest SHA256: {h[:16]}...")

    # 5. 正负完整性
    assert len(train_pairs) + len(test_pairs) == len(pair_set(df)), "[FAIL] 划分有重复/遗漏"
    print(f"[PASS] ✅ 划分完整性 (train+test = 全部候选)")
    print(f"\n审计通过 ✓")


if __name__ == '__main__':
    main()
