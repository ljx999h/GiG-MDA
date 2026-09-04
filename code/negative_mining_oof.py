"""
R2 阶段1: out-of-fold 可靠负样本挖掘 (P0-2 修复)

替代旧的 paper_negative_sampling (每候选最多1次预测 + 默认0被当一致判负).
新设计 (cross-fitting / out-of-fold):
  - 把 train 负池 U_train 分成 K 份
  - 对每份 i: 用 (P_train 正样本 + 其他 K-1 份负池) 训练弱分类器 M_i
  - M_i 对**整个 U_train** 预测 → 每个候选被 K 个模型真实预测
  - coverage_count = K (无默认0, 必须全部被真实预测)
  - num_preds = 被预测为正的次数 ∈ [0, K]
  - 可靠负样本 = num_preds == 0 (K 个模型一致判负) 且 coverage == K

输出:
  data/{DS}/Splits/train_positives.csv   (P_train 正样本 + MiRAGE 特征)
  data/{DS}/Splits/train_negatives.csv   (选中的可靠负样本 + coverage + num_preds)
  data/{DS}/Splits/neg_mining_report.json
"""
import argparse
import json
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier

DATASETS = {
    'C': {
        'mapping': 'data/C-Dataset/Mapping/mapping_C.csv',
        'manifest': 'data/C-Dataset/Splits/split_manifest.csv',
        'score': 'code/results/MiRAGE_score_C.csv',
        'out_dir': 'data/C-Dataset/Splits',
    },
    'F': {
        'mapping': 'data/F-Dataset/Mapping/mapping_F.csv',
        'manifest': 'data/F-Dataset/Splits/split_manifest.csv',
        'score': 'code/results/MiRAGE_score_F.csv',
        'out_dir': 'data/F-Dataset/Splits',
    },
    'DDCD': {
        'mapping': 'data/DDCD/Mapping/mapping.csv',
        'manifest': 'data/DDCD/Splits/split_manifest.csv',
        'score': 'code/results/MiRAGE_score_DDCD.csv',
        'out_dir': 'data/DDCD/Splits',
    },
}


def load_score(ds, cfg):
    """读 MiRAGE_score, 统一 pair key 为 (drugID, diseaseID) 字符串.
    只保留特征 (label 来自 manifest, 避免 merge 冲突)."""
    df = pd.read_csv(cfg['score']).fillna(0.0)
    df['drugID'] = df['drugID'].astype(str).str.strip()
    df['diseaseID'] = df['diseaseID'].astype(str).str.strip()
    if 'label' in df.columns:
        df = df.drop(columns=['label'])

    # C 特殊: MiRAGE_score 的 drugID 是整数索引, 需映射回 DB 字符串
    if ds == 'C':
        mapping = pd.read_csv(cfg['mapping'])
        if 'Unnamed: 0' in mapping.columns:
            mapping = mapping.drop(columns=['Unnamed: 0'])
        drugbanks = sorted(mapping.iloc[:, 0].astype(str).str.strip().unique())
        idx_to_db = {i: db for i, db in enumerate(drugbanks)}
        df['drugID'] = df['drugID'].map(lambda x: idx_to_db.get(int(x), x))
    return df


def load_manifest_train(cfg):
    """读 split_manifest, 只取 train 部分, 统一 ID 字符串."""
    mf = pd.read_csv(cfg['manifest'])
    mf['drugID'] = mf['drugID'].astype(str).str.strip()
    mf['diseaseID'] = mf['diseaseID'].astype(str).str.strip()
    train = mf[mf['split'] == 'train'].copy()
    return train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, choices=['C', 'F', 'DDCD'])
    parser.add_argument('--K', type=int, default=5, help='cross-fitting 折数')
    parser.add_argument('--random-state', type=int, default=42)
    parser.add_argument('--max-neg-ratio', type=float, default=None,
                        help='负样本:正样本比例上限, None=与正样本匹配(1:1)')
    parser.add_argument('--score', default=None,
                        help='覆盖 MiRAGE_score 路径 (R2 用 code/results/MiRAGE_score_{DS}_r2.csv)')
    parser.add_argument('--manifest', default=None,
                        help='覆盖 split_manifest 路径 (冷启动划分用)')
    parser.add_argument('--out-dir', default=None,
                        help='覆盖输出目录 (冷启动用, 避免覆盖常规 train_pos/neg)')
    args = parser.parse_args()

    cfg = DATASETS[args.dataset]
    if args.score:
        cfg['score'] = args.score
    if args.manifest:
        cfg['manifest'] = args.manifest
    if args.out_dir:
        cfg['out_dir'] = args.out_dir
    print(f"\n{'='*60}\n out-of-fold 负采样: {args.dataset} (K={args.K})\n{'='*60}")
    print(f"  score 文件: {cfg['score']}")

    # 1. 加载
    score = load_score(args.dataset, cfg)
    manifest_train = load_manifest_train(cfg)
    print(f"MiRAGE_score: {len(score):,} 行 | train manifest: {len(manifest_train):,}")

    # 2. 对齐 train 部分
    key = ['drugID', 'diseaseID']
    aligned = manifest_train.merge(score, on=key, how='inner')
    print(f"对齐后: {len(aligned):,} 行 (应= train候选数)")
    if len(aligned) < len(manifest_train):
        print(f"  [WARN] 部分 manifest 无特征: {len(manifest_train)-len(aligned)} 行丢失")

    # 3. 分离正负
    pos = aligned[aligned['label'] == 1].copy()
    neg_pool = aligned[aligned['label'] == 0].copy()
    print(f"P_train 正: {len(pos):,} | U_train 负池: {len(neg_pool):,}")

    # 特征列 (P0-1 修复): 显式使用该数据集的非-MolFormer GBA 特征,
    # 禁止 MolFormer/预训练分子相似度列进入委员会筛选 (否则负样本受分子信息影响)
    feat_cols = [c for c in score.columns
                 if c not in key + ['label'] and 'MolFormer' not in c]
    banned = [c for c in feat_cols if 'MolEmb' in c or 'MolFormer' in c]
    assert not banned, f'预训练分子列混入委员会特征: {banned}'
    assert feat_cols, '委员会特征列为空'
    print(f'  委员会特征列 ({len(feat_cols)} 列, 无 MolFormer): {feat_cols[:3]} ...')

    # 4. out-of-fold 投票
    kf = KFold(n_splits=args.K, shuffle=True, random_state=args.random_state)
    folds = list(kf.split(neg_pool))
    num_preds = pd.Series(0, index=neg_pool.index)
    coverage = pd.Series(0, index=neg_pool.index)

    for i, (train_idx, _) in enumerate(folds):
        neg_train_fold = neg_pool.iloc[train_idx]
        X_train = pd.concat([pos[feat_cols], neg_train_fold[feat_cols]])
        y_train = pd.concat([pd.Series(1, index=pos.index),
                             pd.Series(0, index=neg_train_fold.index)])
        clf = DecisionTreeClassifier(max_depth=4, min_samples_leaf=15,
                                     random_state=args.random_state + i)
        clf.fit(X_train, y_train)
        # 对整个 U_train 预测 (每个候选被每个模型真实预测)
        y_pred_all = clf.predict(neg_pool[feat_cols])
        coverage += 1
        num_preds.loc[neg_pool.index[y_pred_all == 1]] += 1
        print(f"  Fold {i+1}/{args.K} 完成 | 累计 num_preds>0: {(num_preds>0).sum():,}")

    # 5. 断言: 每个候选被 K 次真实预测
    assert (coverage == args.K).all(), \
        f"[FAIL] 存在候选未获得 K={args.K} 次预测 (coverage 分布: {coverage.value_counts().to_dict()})"
    print(f"  ✅ 所有 {len(neg_pool):,} 个负候选均被 K={args.K} 次真实预测")

    # 6. 可靠负样本
    reliable = neg_pool[(num_preds == 0) & (coverage == args.K)].copy()
    reliable['coverage_count'] = coverage[reliable.index].values
    reliable['num_preds'] = num_preds[reliable.index].values
    print(f"  可靠负样本 (num_preds=0 & coverage={args.K}): {len(reliable):,} "
          f"({len(reliable)/len(neg_pool):.1%} of 负池)")

    # 7. 采样到与正样本匹配
    n_pos = len(pos)
    if args.max_neg_ratio and len(reliable) > n_pos * args.max_neg_ratio:
        reliable = reliable.sample(n=int(n_pos * args.max_neg_ratio),
                                   random_state=args.random_state)
    else:
        reliable = reliable.sample(n=min(n_pos, len(reliable)),
                                   random_state=args.random_state)
    print(f"  最终选中负样本: {len(reliable):,} (1:{len(reliable)/max(1,n_pos):.1f} vs 正)")

    # 8. 保存
    os.makedirs(cfg['out_dir'], exist_ok=True)
    pos_out = os.path.join(cfg['out_dir'], 'train_positives.csv')
    neg_out = os.path.join(cfg['out_dir'], 'train_negatives.csv')
    pos.to_csv(pos_out, index=False)
    reliable.to_csv(neg_out, index=False)
    print(f"  ✅ 正样本: {pos_out} ({len(pos):,})")
    print(f"  ✅ 负样本: {neg_out} ({len(reliable):,})")

    # 9. 报告
    report = {
        'dataset': args.dataset,
        'K': args.K,
        'n_pos_train': int(n_pos),
        'n_neg_pool': int(len(neg_pool)),
        'n_reliable': int((num_preds == 0).sum()),
        'n_selected_neg': int(len(reliable)),
        'coverage_all_K': bool((coverage == args.K).all()),
        'num_preds_dist': num_preds.value_counts().to_dict(),
    }
    rep_out = os.path.join(cfg['out_dir'], 'neg_mining_report.json')
    with open(rep_out, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  ✅ 报告: {rep_out}")


if __name__ == '__main__':
    main()
