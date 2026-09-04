# -*- coding: utf-8 -*-
"""
P0-1 负采样隔离全量 sweep (cold 家族): committee 已排除 MolFormer 列.
对每个存在的 (dataset, mode, seed): 备份旧负样本 -> 重生成 -> cold_eval -> 记录.
输出: results/R2/P01_sweep/p01_cold_summary.csv (+ 每组合 eval/mine log)
"""
import csv
import os
import re
import shutil
import subprocess
import sys

BK = 'results/R2/P01_sweep/backup'
LOG = 'results/R2/P01_sweep'
os.makedirs(BK, exist_ok=True)
os.makedirs(LOG, exist_ok=True)

D2DIR = {'C': 'C-Dataset', 'F': 'F-Dataset', 'DDCD': 'DDCD'}
MODE2TAG = {'cold-drug': 'cold', 'cold-disease': 'colddis', 'scaffold': 'scaffold'}


def resolve(ds, mode, seed):
    d = D2DIR[ds]
    tag = MODE2TAG[mode]
    dirn = {'cold-drug': 'Cold', 'cold-disease': 'Cold_dis',
            'scaffold': 'Cold_scaf'}[mode]
    if seed != 42:
        dirn += f'_s{seed}'
    negdir = f'data/{d}/Splits/{dirn}'
    mf = f'data/{d}/Splits/split_manifest_{tag}.csv'
    score = f'code/results/MiRAGE_score_{ds}_{tag}.csv'
    if seed != 42:
        mf = f'data/{d}/Splits/split_manifest_{tag}_s{seed}.csv'
        score = f'code/results/MiRAGE_score_{ds}_{tag}_s{seed}.csv'
    return negdir, mf, score


def run(cmd):
    r = subprocess.run([sys.executable] + cmd, capture_output=True)
    out = (r.stdout or b'').decode('utf-8', errors='replace')
    err = (r.stderr or b'').decode('utf-8', errors='replace')
    return r.returncode == 0, out + err


def parse_eval(text):
    out = {}
    for line in text.splitlines():
        m = re.match(r'\s*(\S+)\s+(\d+)\s+AUROC=([\d.]+)\s+AUPR=([\d.]+)', line)
        if m:
            out[m.group(1)] = m.group(4)
    return out


rows = []
combos = [('C', 'cold-disease', 7), ('C', 'cold-disease', 2024),
          ('DDCD', 'cold-disease', 7), ('C', 'scaffold', 7),
          ('DDCD', 'scaffold', 7)]

for ds, mode, seed in combos:
    negdir, mf, score = resolve(ds, mode, seed)
    if not os.path.exists(mf) or not os.path.exists(score):
        print(f'skip(no files) {ds} {mode} s{seed}', flush=True)
        continue
    os.makedirs(negdir, exist_ok=True)
    tag = f'{ds}_{mode}_s{seed}'
    bdir = os.path.join(BK, tag)
    os.makedirs(bdir, exist_ok=True)
    for f in ('train_negatives.csv', 'train_positives.csv'):
        src = os.path.join(negdir, f)
        if os.path.exists(src):
            shutil.copyfile(src, os.path.join(bdir, f + '.old'))
    print(f'[{ds} {mode} s{seed}] mining', flush=True)
    ok, log = run(['code/negative_mining_oof.py', '--dataset', ds,
                   '--manifest', mf, '--score', score, '--out-dir', negdir])
    with open(os.path.join(LOG, f'{tag}_mine.log'), 'w', encoding='utf-8') as fh:
        fh.write(log)
    if not ok:
        print(f'  MINE-FAIL {tag}', flush=True)
        continue
    ok, log = run(['code/cold_eval.py', '--dataset', ds, '--seed', str(seed),
                   '--mode', mode])
    with open(os.path.join(LOG, f'{tag}_eval.log'), 'w', encoding='utf-8') as fh:
        fh.write(log)
    vals = parse_eval(log)
    rows.append({'dataset': ds, 'mode': mode, 'seed': seed,
                 'base': vals.get('MiRAGE'),
                 'mol': vals.get('MiRAGE+MolEmb32'),
                 'embed': vals.get('MiRAGE+embed'),
                 'both': vals.get('MiRAGE+embed+MolEmb32')})
    print(f'[{ds} {mode} s{seed}] done base={vals.get("MiRAGE")} '
          f'mol={vals.get("MiRAGE+MolEmb32")}', flush=True)

with open(os.path.join(LOG, 'p01_cold_summary.csv'), 'w', newline='',
          encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=['dataset', 'mode', 'seed',
                                      'base', 'mol', 'embed', 'both'])
    w.writeheader()
    w.writerows(rows)
print('ALL DONE')
