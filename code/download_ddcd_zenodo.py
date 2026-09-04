# -*- coding: utf-8 -*-
"""
下载 Zenodo 上的 DDCD 大文件 (train.csv / test.csv, ~550 MB each)
===============================================================
README 声称的大文件恢复脚本. Zenodo 记录: 10.5281/zenodo.21883418
目标位置: data/DDCD/Evaluation/{train,test}.csv (MiRAGE 原始评估划分文件,
与 data_split.py 生成的 Splits/ 无依赖; 主要供旧版评估文件核对用).

用法: python code/download_ddcd_zenodo.py [--out data/DDCD/Evaluation]
"""
import argparse
import json
import os
import urllib.request

RECORD_ID = '21883418'
API = f'https://zenodo.org/api/records/{RECORD_ID}'
FILES = ['test.csv', 'train.csv']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='data/DDCD/Evaluation')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    with urllib.request.urlopen(API, timeout=60) as r:
        rec = json.load(r)
    links = {f['key']: f['links']['self'] for f in rec.get('files', [])}
    missing = [f for f in FILES if f not in links]
    if missing:
        raise SystemExit(f'Zenodo record lacks files: {missing}')
    for name in FILES:
        dest = os.path.join(args.out, name)
        if os.path.exists(dest):
            print(f'skip (exists): {dest}')
            continue
        print(f'downloading {name} ...')
        urllib.request.urlretrieve(links[name], dest)  # noqa: S310  (HTTPS)
        print(f'saved: {dest}')


if __name__ == '__main__':
    main()
