# -*- coding: utf-8 -*-
"""cold-disease 多 seed 补齐: C s123/s2024, DDCD s7/s123. 每种子: 划分→特征→嵌入→负采样→评估."""
import subprocess
import time

PY = r'C:/Users/30744/anaconda3/python.exe'


def run(cmd, tag):
    full = [PY] + [str(c) for c in cmd]
    print(f">>> {' '.join(full)}", flush=True)
    r = subprocess.run(full)
    if r.returncode != 0:
        print(f"FAILED: {tag}", flush=True)
        return False
    return True


def dis_pipeline(ds, seed):
    t0 = time.time()
    tag = f'{ds}_{seed}_dis'
    if ds == 'DDCD':
        mdir, sdir = 'DDCD', 'DDCD'
    else:
        mdir, sdir = f'{ds}-Dataset', f'{ds}-Dataset'
    ok = run(['code/cold_split.py', '--dataset', ds, '--mode', 'cold-disease',
              '--random-state', str(seed),
              '--out', f'data/{sdir}/Splits/split_manifest_colddis_s{seed}.csv'], f'{tag}_split')
    ok = ok and run(['code/build_mirage_features.py', '--dataset', ds, '--neighbor-source', 'r2train',
                     '--manifest', f'data/{sdir}/Splits/split_manifest_colddis_s{seed}.csv',
                     '--out', f'code/results/MiRAGE_score_{ds}_colddis_s{seed}.csv'], f'{tag}_feat')
    ok = ok and run(['code/pretrain_gigs_split.py', '--dataset', ds,
                     '--manifest', f'data/{sdir}/Splits/split_manifest_colddis_s{seed}.csv',
                     '--out', f'data/{sdir}/Splits/gigs_split_{ds}_colddis_s{seed}.pkl'], f'{tag}_gigs')
    ok = ok and run(['code/negative_mining_oof.py', '--dataset', ds,
                     '--manifest', f'data/{sdir}/Splits/split_manifest_colddis_s{seed}.csv',
                     '--score', f'code/results/MiRAGE_score_{ds}_colddis_s{seed}.csv',
                     '--out-dir', f'data/{sdir}/Splits/Cold_dis_s{seed}'], f'{tag}_neg')
    ok = ok and run(['docx_check/cold_eval.py', '--dataset', ds, '--seed', str(seed),
                     '--mode', 'cold-disease'], f'{tag}_eval')
    print(f"  [{ds} colddis s{seed}] {'OK' if ok else 'FAIL'} {time.time()-t0:.0f}s", flush=True)
    return ok


def main():
    jobs = [('C', 123), ('C', 2024), ('DDCD', 7), ('DDCD', 123)]
    for ds, seed in jobs:
        dis_pipeline(ds, seed)
    print("\ncold-disease 补齐完成", flush=True)


if __name__ == '__main__':
    main()
