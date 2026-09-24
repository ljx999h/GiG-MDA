# -*- coding: utf-8 -*-
"""
汇总冷启动评估结果，生成 results/cold_start_results.csv（稿件表 3 的机器可读来源）。

对每个 (dataset, seed) 组合：
  1) 若缺少训练内构建的输入，按需生成：
       build_mirage_features --neighbor-source r2train --manifest <冷划分> --exclude MolFormer
       pretrain_gigs_split   --manifest <冷划分>
       negative_mining_oof   --manifest <冷划分> --score <特征>
  2) 用 dump_cold_predictions 评估四个主要 XGBoost 配置
  3) 读取其 summary.csv，汇总为 cold_start_results.csv
      列: dataset,seed,base,molemb32,grmf,both

用法：
  python code/aggregate_cold_results.py --datasets C --seeds 42 7          # 子集
  python code/aggregate_cold_results.py --datasets C F DDCD --seeds 42 7 123 2024   # 全部
"""
import argparse
import os
import subprocess
import sys

import pandas as pd

PY = sys.executable
MANIFEST = {
    'C': 'data/C-Dataset/Splits/split_manifest_cold',
    'F': 'data/F-Dataset/Splits/split_manifest_cold',
    'DDCD': 'data/DDCD/Splits/split_manifest_cold',
}
SPLITS_DIR = {
    'C': 'data/C-Dataset/Splits',
    'F': 'data/F-Dataset/Splits',
    'DDCD': 'data/DDCD/Splits',
}
# dump_cold_predictions 的变体名 -> 输出列名
COLMAP = {'MiRAGE': 'base', 'MiRAGE+MolEmb32': 'molemb32',
          'MiRAGE+embed': 'grmf', 'MiRAGE+embed+MolEmb32': 'both'}


def run(cmd):
    print(f"\n>>> {' '.join(map(str, cmd))}", flush=True)
    r = subprocess.run([str(c) for c in cmd])
    if r.returncode != 0:
        raise RuntimeError(f'FAILED: {" ".join(map(str, cmd))}')


def suffix(seed):
    return '' if seed == 42 else f'_s{seed}'


def ensure_inputs(ds, seed):
    """按需生成该 (dataset, seed) 的训练内构建输入。"""
    sfx = suffix(seed)
    manifest = f'{MANIFEST[ds]}{sfx}.csv'
    feats = f'code/results/MiRAGE_score_{ds}_cold{sfx}.csv'
    gigs = f'{SPLITS_DIR[ds]}/gigs_split_{ds}_cold{sfx}.pkl'
    colddir = f'{SPLITS_DIR[ds]}/Cold{sfx}'

    if not os.path.exists(feats):
        run([PY, 'code/build_mirage_features.py', '--dataset', ds,
             '--neighbor-source', 'r2train', '--exclude', 'MolFormer',
             '--manifest', manifest, '--out', feats])
    if not os.path.exists(gigs):
        run([PY, 'code/pretrain_gigs_split.py', '--dataset', ds,
             '--manifest', manifest, '--out', gigs])
    if not os.path.exists(os.path.join(colddir, 'train_positives.csv')):
        run([PY, 'code/negative_mining_oof.py', '--dataset', ds,
             '--manifest', manifest, '--score', feats, '--out-dir', colddir])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=['C', 'F', 'DDCD'])
    ap.add_argument('--seeds', nargs='+', type=int, default=[42, 7, 123, 2024])
    ap.add_argument('--out', default='results/cold_start_results.csv')
    ap.add_argument('--workdir', default='results/pred_dump_aggregate')
    args = ap.parse_args()

    rows = []
    for ds in args.datasets:
        for seed in args.seeds:
            print(f"\n{'#'*66}\n# {ds} | seed {seed}\n{'#'*66}", flush=True)
            ensure_inputs(ds, seed)
            run([PY, 'code/dump_cold_predictions.py', '--dataset', ds,
                 '--seed', str(seed), '--out-dir', args.workdir])
            s = pd.read_csv(f'{args.workdir}/{ds}_s{seed}_summary.csv')
            rec = {'dataset': ds, 'seed': seed}
            for _, r in s.iterrows():
                rec[COLMAP[r['variant']]] = round(float(r['AUPR_sklearn']), 4)
            rows.append(rec)

    order = {'C': 0, 'F': 1, 'DDCD': 2}
    df = pd.DataFrame(rows)[['dataset', 'seed', 'base', 'molemb32', 'grmf', 'both']]
    df = df.sort_values(['dataset', 'seed'], key=lambda c: c.map(order) if c.name == 'dataset' else c)
    df.to_csv(args.out, index=False)
    print(f'\n已写出 {args.out}')
    print(df.to_string(index=False))
    n_ok = int((df.both > df.base).sum())
    print(f'\n三通道 > 基础: {n_ok}/{len(df)} 个冷划分')


if __name__ == '__main__':
    main()
