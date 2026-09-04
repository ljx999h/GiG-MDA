# -*- coding: utf-8 -*-
"""F-Dataset cold-drug 补跑: 4 seeds × (划分→特征→嵌入→负采样→评估)。"""
import subprocess
import time

import sys
PY = sys.executable


def run(cmd, tag):
    full = [PY] + [str(c) for c in cmd]
    print(f">>> {' '.join(full)}", flush=True)
    r = subprocess.run(full)
    if r.returncode != 0:
        print(f"FAILED: {tag}", flush=True)
        return False
    return True


def main():
    for seed in [42, 7, 123, 2024]:
        t0 = time.time()
        tag = f'F_{seed}'
        if seed == 42:
            out_m, out_s = 'data/F-Dataset/Splits/split_manifest_cold.csv', 'code/results/MiRAGE_score_F_cold.csv'
            gigs_out = 'data/F-Dataset/Splits/gigs_split_F_cold.pkl'
            neg_dir = 'data/F-Dataset/Splits/Cold'
        else:
            out_m = f'data/F-Dataset/Splits/split_manifest_cold_s{seed}.csv'
            out_s = f'code/results/MiRAGE_score_F_cold_s{seed}.csv'
            gigs_out = f'data/F-Dataset/Splits/gigs_split_F_cold_s{seed}.pkl'
            neg_dir = f'data/F-Dataset/Splits/Cold_s{seed}'
        ok = run(['code/cold_split.py', '--dataset', 'F', '--mode', 'cold-drug',
                  '--random-state', str(seed), '--out', out_m], f'{tag}_split')
        ok = ok and run(['code/build_mirage_features.py', '--dataset', 'F', '--neighbor-source', 'r2train',
                         '--exclude', 'MolFormer',
                         '--manifest', out_m, '--out', out_s], f'{tag}_feat')
        ok = ok and run(['code/pretrain_gigs_split.py', '--dataset', 'F',
                         '--manifest', out_m, '--out', gigs_out], f'{tag}_gigs')
        ok = ok and run(['code/negative_mining_oof.py', '--dataset', 'F',
                         '--manifest', out_m, '--score', out_s, '--out-dir', neg_dir], f'{tag}_neg')
        ok = ok and run(['code/cold_eval.py', '--dataset', 'F', '--seed', str(seed),
                         '--mode', 'cold-drug'], f'{tag}_eval')
        print(f"  [F cold s{seed}] {'OK' if ok else 'FAIL'} {time.time()-t0:.0f}s", flush=True)
    print('\nF 冷启动完成', flush=True)


if __name__ == '__main__':
    main()
