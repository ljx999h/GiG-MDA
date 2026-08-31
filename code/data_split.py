"""
R2 阶段1: pair-disjoint 数据划分 (方案B前提)

对每个数据集, 把全候选对空间 (drug × disease) 划分为 train/test:
  - 正样本 (已知关联): P_train / P_test   (pair-disjoint)
  - 负样本 (未标注对):  U_train / U_test   (pair-disjoint)
保证 train 与 test 的 (drug, disease) pair 零交集, 这是 held-out 评价独立性的前提.

输出: data/{DS}/Splits/split_manifest.csv
  列: drugID, diseaseID, label(1=已知关联/0=未标注), split(train/test)
  同时输出 data/{DS}/Splits/split_summary.json (统计)
"""
import argparse
import json
import os
import pandas as pd
from sklearn.model_selection import train_test_split

# 数据集配置
DATASETS = {
    'C': {
        'mapping': 'data/C-Dataset/Mapping/mapping_C.csv',
        'out_dir': 'data/C-Dataset/Splits',
        'note': 'C-Dataset (drug=DrugBank str, disease=int)',
    },
    'F': {
        'mapping': 'data/F-Dataset/Mapping/mapping_F.csv',
        'out_dir': 'data/F-Dataset/Splits',
        'note': 'F-Dataset (drug=DrugBank str, disease=OMIM str)',
    },
    'DDCD': {
        'mapping': 'data/DDCD/Mapping/mapping.csv',
        'out_dir': 'data/DDCD/Splits',
        'note': 'DDCD (drug=DrugBank str, disease=MESH str)',
    },
}


def clean_mapping(df):
    """清理映射, 统一列名为 drugID/diseaseID, 字符串化 ID."""
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    if len(df.columns) >= 2:
        df = df.iloc[:, :2]
        df.columns = ['drugID', 'diseaseID']
    df['drugID'] = df['drugID'].astype(str).str.strip()
    df['diseaseID'] = df['diseaseID'].astype(str).str.strip()
    # 去重
    df = df.drop_duplicates(subset=['drugID', 'diseaseID'])
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, choices=['C', 'F', 'DDCD'])
    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument('--random-state', type=int, default=42)
    args = parser.parse_args()

    cfg = DATASETS[args.dataset]
    print(f"\n{'='*60}\n {cfg['note']}\n{'='*60}")

    # 1. 读已知关联
    mapping = clean_mapping(pd.read_csv(cfg['mapping']))
    drugs = sorted(mapping['drugID'].unique())
    diseases = sorted(mapping['diseaseID'].unique())
    n_pos = len(mapping)
    n_cand = len(drugs) * len(diseases)
    print(f"已知关联(正): {n_pos:,} | 候选对空间: {len(drugs):,} × {len(diseases):,} = {n_cand:,}")

    # 2. 全候选对空间
    print("  生成候选对空间 (笛卡尔积)...")
    all_pairs = pd.DataFrame({'drugID': drugs}).merge(
        pd.DataFrame({'diseaseID': diseases}), how='cross'
    )

    # 3. 标记正负
    known = set(zip(mapping['drugID'], mapping['diseaseID']))
    all_pairs['label'] = [
        1 if (d, s) in known else 0
        for d, s in zip(all_pairs['drugID'], all_pairs['diseaseID'])
    ]
    pos = all_pairs[all_pairs['label'] == 1].copy()
    neg = all_pairs[all_pairs['label'] == 0].copy()
    print(f"  正: {len(pos):,} | 负: {len(neg):,}")

    # 4. pair-disjoint 划分 (train/test)
    print("  划分 train/test (pair-disjoint)...")
    pos_train, pos_test = train_test_split(
        pos, test_size=args.test_size, random_state=args.random_state
    )
    neg_train, neg_test = train_test_split(
        neg, test_size=args.test_size, random_state=args.random_state
    )

    # 5. 零交集断言
    def pair_set(df):
        return set(zip(df['drugID'], df['diseaseID']))

    pt, pte = pair_set(pos_train), pair_set(pos_test)
    nt, nte = pair_set(neg_train), pair_set(neg_test)
    assert pt.isdisjoint(pte), "❌ 正样本 train/test pair 重叠!"
    assert nt.isdisjoint(nte), "❌ 负样本 train/test pair 重叠!"
    assert pt.isdisjoint(nte) and pte.isdisjoint(nt), "❌ 正负跨集重叠!"
    print("  ✅ 正/负 train/test 全部 pair-disjoint")

    # 6. 打标并输出
    for df, split in [(pos_train, 'train'), (pos_test, 'test'),
                      (neg_train, 'train'), (neg_test, 'test')]:
        df['split'] = split
    manifest = pd.concat([pos_train, pos_test, neg_train, neg_test], ignore_index=True)
    manifest = manifest[['drugID', 'diseaseID', 'label', 'split']]

    os.makedirs(cfg['out_dir'], exist_ok=True)
    out_manifest = os.path.join(cfg['out_dir'], 'split_manifest.csv')
    manifest.to_csv(out_manifest, index=False)
    print(f"  ✅ 划分清单已保存: {out_manifest} ({len(manifest):,} 行)")

    # 7. 汇总
    summary = {
        'dataset': args.dataset,
        'test_size': args.test_size,
        'random_state': args.random_state,
        'n_drugs': len(drugs),
        'n_diseases': len(diseases),
        'n_candidates': n_cand,
        'n_pos_train': len(pos_train), 'n_pos_test': len(pos_test),
        'n_neg_train': len(neg_train), 'n_neg_test': len(neg_test),
        'train_test_pair_disjoint': True,
    }
    out_summary = os.path.join(cfg['out_dir'], 'split_summary.json')
    with open(out_summary, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  ✅ 汇总已保存: {out_summary}")
    print(f"\n  train: 正{len(pos_train):,} 负{len(neg_train):,} | "
          f"test: 正{len(pos_test):,} 负{len(neg_test):,}")


if __name__ == '__main__':
    main()
