"""
R2 多种子实验编排 (run_multiseed)

对每个 seed 跑完整管线 (划分→OOF负采样→GRMF重训→账本), 全部产物覆盖式生成:
  data_split.py            --random-state <seed>   (pair-disjoint 划分)
  negative_mining_oof.py   --random-state <seed>   (OOF 可靠负采样, 用 r2 特征)
  pretrain_gigs_split.py   --random-state <seed>   (GRMF 在 P_train 重训, 种子化初始化)
  build_results_ledger.py  --seed <seed> --append  (账本追加该 seed 的行)

统计单位 = seed (独立划分); 同一 seed 内模型间可配对比较.
输出: results/R2/results_manifest.csv (每行含 seed 列)

用法:
  python code/run_multiseed.py --seeds 42 7 123 2024 999 --datasets C F DDCD
  python code/run_multiseed.py --seeds 7 --datasets C        # 快速冒烟
"""
import argparse
import subprocess
import sys
import time

SCORES = {
    'C': 'code/results/MiRAGE_score_C_r2.csv',
    'F': 'code/results/MiRAGE_score_F_r2.csv',
    'DDCD': 'code/results/MiRAGE_score_DDCD_r2.csv',
}


def run(cmd):
    print(f"\n>>> {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise RuntimeError(f"FAILED: {' '.join(cmd)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seeds', nargs='+', type=int, default=[42, 7, 123, 2024, 999])
    ap.add_argument('--datasets', nargs='+', default=['C', 'F', 'DDCD'])
    args = ap.parse_args()

    for seed in args.seeds:
        for ds in args.datasets:
            print(f"\n{'#'*70}\n# SEED {seed} | {ds}\n{'#'*70}", flush=True)
            t0 = time.time()
            run([sys.executable, 'code/data_split.py', '--dataset', ds, '--random-state', str(seed)])
            run([sys.executable, 'code/build_mirage_features.py', '--dataset', ds,
                 '--neighbor-source', 'r2train'])  # per-seed 重建 GBA 特征 (fold-local)
            run([sys.executable, 'code/negative_mining_oof.py', '--dataset', ds,
                 '--random-state', str(seed), '--score', SCORES[ds]])
            run([sys.executable, 'code/pretrain_gigs_split.py', '--dataset', ds,
                 '--random-state', str(seed)])
            run([sys.executable, 'code/build_results_ledger.py', '--dataset', ds,
                 '--seed', str(seed), '--append'])
            print(f"  [seed {seed} | {ds}] 完成, 耗时 {time.time()-t0:.0f}s", flush=True)
    print("\n全部种子完成", flush=True)


if __name__ == '__main__':
    main()
