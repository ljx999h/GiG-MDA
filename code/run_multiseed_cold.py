"""
R2 多 seed 冷启动编排 (run_multiseed_cold)

对每个 cold split 种子跑完整冷启动管线:
  cold_split --random-state <s> --out *_cold_s{s}.csv
  build_mirage_features --manifest *_cold_s{s}.csv --out MiRAGE_score_{DS}_cold_s{s}.csv
  pretrain_gigs_split  --manifest *_cold_s{s}.csv --out gigs_split_{DS}_cold_s{s}.pkl
  negative_mining_oof  --manifest *_cold_s{s}.csv --out-dir Cold_s{s}
  cold_eval --seed <s>  (输出 4 变体 AUPR/AUROC)

用法: python code/run_multiseed_cold.py --seeds 7 123 2024 --datasets C DDCD
"""
import argparse
import subprocess
import sys
import time

COLD_MANIFEST = {
    'C': 'data/C-Dataset/Splits/split_manifest_cold',
    'F': 'data/F-Dataset/Splits/split_manifest_cold',
    'DDCD': 'data/DDCD/Splits/split_manifest_cold',
}


def run(cmd):
    print(f"\n>>> {' '.join(map(str, cmd))}", flush=True)
    r = subprocess.run([str(c) for c in cmd])
    if r.returncode != 0:
        raise RuntimeError(f"FAILED: {' '.join(map(str, cmd))}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seeds', nargs='+', type=int, default=[7, 123, 2024])
    ap.add_argument('--datasets', nargs='+', default=['C', 'DDCD'])
    args = ap.parse_args()

    for ds in args.datasets:
        base_manifest = COLD_MANIFEST[ds]
        for s in args.seeds:
            tag = f's{s}'
            print(f"\n{'#'*70}\n# COLD {ds} | seed {s}\n{'#'*70}", flush=True)
            t0 = time.time()
            run([sys.executable, 'code/cold_split.py', '--dataset', ds,
                 '--random-state', str(s), '--out', f'{base_manifest}_{tag}.csv'])
            run([sys.executable, 'code/build_mirage_features.py', '--dataset', ds,
                 '--neighbor-source', 'r2train',
                 '--manifest', f'{base_manifest}_{tag}.csv',
                 '--out', f'code/results/MiRAGE_score_{ds}_cold_{tag}.csv'])
            run([sys.executable, 'code/pretrain_gigs_split.py', '--dataset', ds,
                 '--manifest', f'{base_manifest}_{tag}.csv',
                 '--out', f'data/{ds}-Dataset/Splits/gigs_split_{ds}_cold_{tag}.pkl'
                          if ds != 'DDCD' else f'data/DDCD/Splits/gigs_split_{ds}_cold_{tag}.pkl'])
            run([sys.executable, 'code/negative_mining_oof.py', '--dataset', ds,
                 '--manifest', f'{base_manifest}_{tag}.csv',
                 '--score', f'code/results/MiRAGE_score_{ds}_cold_{tag}.csv',
                 '--out-dir', f'data/{ds}-Dataset/Splits/Cold_{tag}'
                              if ds != 'DDCD' else f'data/DDCD/Splits/Cold_{tag}'])
            run([sys.executable, 'docx_check/cold_eval.py', '--dataset', ds, '--seed', s])
            print(f"  [COLD {ds} | seed {s}] 完成, 耗时 {time.time()-t0:.0f}s", flush=True)
    print("\n全部冷启动种子完成", flush=True)


if __name__ == '__main__':
    main()
