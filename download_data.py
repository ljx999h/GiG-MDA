#!/usr/bin/env python3
"""
Download the large DDCD dataset files (train.csv, test.csv) from Zenodo.

These two files exceed GitHub's 100 MB per-file limit and are therefore hosted
on Zenodo. Running this script restores the full DDCD evaluation data so the
paper's DDCD experiments can be reproduced.

Usage:
    python download_data.py          # download both files
    python download_data.py --check  # verify files already exist with correct size

After downloading, the layout will be:
    data/DDCD/Evaluation/train.csv   (526 MB)
    data/DDCD/Evaluation/test.csv    (533 MB)
"""
import argparse
import os
import sys
import urllib.request

# ============================================================
#  DDCD 大文件托管于 Zenodo (GitHub 单个文件上限 100 MB)
#  DOI: https://doi.org/10.5281/zenodo.21883418
# ============================================================
ZENODO_FILES = {
    "data/DDCD/Evaluation/train.csv": "https://zenodo.org/records/21883418/files/train.csv?download=1",
    "data/DDCD/Evaluation/test.csv":  "https://zenodo.org/records/21883418/files/test.csv?download=1",
}

# 预期文件大小 (字节), 用于校验下载完整性 (实测自 Zenodo)
EXPECTED_SIZES = {
    "data/DDCD/Evaluation/train.csv": 551_045_869,
    "data/DDCD/Evaluation/test.csv":  558_133_190,
}

HERE = os.path.dirname(os.path.abspath(__file__))


def check_files():
    """检查目标文件是否已存在且大小合理."""
    ok = True
    for rel, expected in EXPECTED_SIZES.items():
        path = os.path.join(HERE, rel)
        if not os.path.exists(path):
            print(f"  [MISSING] {rel}")
            ok = False
        else:
            size = os.path.getsize(path)
            status = "OK" if abs(size - expected) < expected * 0.05 else "SIZE MISMATCH"
            print(f"  [{status}] {rel} ({size/1e6:.1f} MB)")
            if status != "OK":
                ok = False
    return ok


def download_all():
    """下载所有 Zenodo 文件."""
    missing_url = [rel for rel, url in ZENODO_FILES.items()
                   if url.startswith("REPLACE_WITH")]
    if missing_url:
        print("[ERROR] 尚未填写 Zenodo 下载链接。")
        print("  1) 将 data/DDCD/Evaluation/train.csv 和 test.csv 上传到 Zenodo")
        print("  2) 把获得的下载链接填入本脚本顶部的 ZENODO_FILES 字典")
        sys.exit(1)

    for rel, url in ZENODO_FILES.items():
        path = os.path.join(HERE, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path):
            print(f"  [SKIP] 已存在: {rel}")
            continue
        print(f"  [下载] {rel} <- {url[:60]}...")
        try:
            urllib.request.urlretrieve(url, path)
            print(f"  [完成] {os.path.getsize(path)/1e6:.1f} MB")
        except Exception as e:
            print(f"  [ERROR] 下载失败: {e}")
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="只检查文件是否存在, 不下载")
    args = parser.parse_args()

    if args.check:
        ok = check_files()
        sys.exit(0 if ok else 1)
    download_all()
    print("\n校验下载结果:")
    check_files()


if __name__ == "__main__":
    main()
