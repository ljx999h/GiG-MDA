# -*- coding: utf-8 -*-
"""
轻量 2 层 GCN 在统一协议下的评估 (常规 + 冷启动).

设计 (与论文统一协议严格对齐):
- 图: 训练关联二部图 (train positives) + 自环, 对称归一化 D^-1/2 (I+A) D^-1/2
- 节点特征: one-hot (标准 GCN 链接预测设置, 不引入关联历史之外的信息)
- 解码器: [h_d; h_s] 拼接 MLP -> score
- 负样本: 统一协议的可靠负样本 (OOF mining 产物), 1:1 采样
- 评估: pair-disjoint test 全候选对, AUPR/AUROC
用法:
  python code/gcn_eval.py --dataset C --seed 42 --mode regular
  python code/gcn_eval.py --dataset C --seed 7 --mode cold-drug
"""
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score

SEEDS = [42, 7, 123, 2024, 999]
HID = 64
EPOCHS = 60
LR = 0.01
WD = 1e-4
BATCH = 8192


def paths(ds, seed, mode):
    if ds == 'DDCD':
        root = 'data/DDCD/Splits'
    else:
        root = f'data/{ds}-Dataset/Splits'
    if mode == 'regular':
        man = f'{root}/split_manifest.csv'
        tp, tn = f'{root}/train_positives.csv', f'{root}/train_negatives.csv'
    else:  # cold-drug
        tag = 'Cold' if seed == 42 else f'Cold_s{seed}'
        man = f'{root}/split_manifest_cold.csv' if seed == 42 else f'{root}/split_manifest_cold_s{seed}.csv'
        tp, tn = f'{root}/{tag}/train_positives.csv', f'{root}/{tag}/train_negatives.csv'
    return man, tp, tn


def load(ds, seed, mode):
    man, tp, tn = paths(ds, seed, mode)
    mf = pd.read_csv(man)
    pos = pd.read_csv(tp)
    neg = pd.read_csv(tn)
    pos = pos[['drugID', 'diseaseID']].copy()
    neg = neg[['drugID', 'diseaseID']].copy()
    pos['label'] = 1
    neg['label'] = 0
    train = pd.concat([pos, neg], ignore_index=True)
    test = mf[mf['split'] == 'test'][['drugID', 'diseaseID', 'label']].copy()
    for df in (train, test):
        df['drugID'] = df['drugID'].astype(str).str.strip()
        df['diseaseID'] = df['diseaseID'].astype(str).str.strip()
    drugs = sorted(set(train['drugID']) | set(test['drugID']))
    dises = sorted(set(train['diseaseID']) | set(test['diseaseID']))
    d2i = {d: i for i, d in enumerate(drugs)}
    s2i = {s: i for i, s in enumerate(dises)}
    nd, ns = len(drugs), len(dises)
    pos_pairs = train[train['label'] == 1][['drugID', 'diseaseID']]
    B = np.zeros((nd, ns), dtype=np.float32)
    for _, r in pos_pairs.iterrows():
        B[d2i[r['drugID']], s2i[r['diseaseID']]] = 1.0
    A = np.zeros((nd + ns, nd + ns), dtype=np.float32)
    A[:nd, nd:] = B
    A[nd:, :nd] = B.T
    A += np.eye(nd + ns, dtype=np.float32)  # 自环
    deg = A.sum(1)
    Dinv = 1.0 / np.sqrt(deg + 1e-9)
    Ahat = A * Dinv[:, None] * Dinv[None, :]
    # GIPK 节点特征 (fold-local: 仅用训练关联) — 文献深方法常用相似度输入
    row_n2 = (B ** 2).sum(1)
    col_n2 = (B ** 2).sum(0)
    gam_d = B.sum() / max(row_n2.sum(), 1e-9)
    gam_s = B.sum() / max(col_n2.sum(), 1e-9)
    SD = np.exp(-gam_d * (row_n2[:, None] + row_n2[None, :] - 2 * B @ B.T))
    SS = np.exp(-gam_s * (col_n2[:, None] + col_n2[None, :] - 2 * B.T @ B))
    X = np.zeros((nd + ns, nd + ns), dtype=np.float32)
    X[:nd, :nd] = SD
    X[nd:, nd:] = SS
    tr_d = train['drugID'].map(d2i).values.astype(np.int64)
    tr_s = train['diseaseID'].map(s2i).values.astype(np.int64) + nd
    tr_y = train['label'].values.astype(np.float32)
    te_d = test['drugID'].map(d2i).values.astype(np.int64)
    te_s = test['diseaseID'].map(s2i).values.astype(np.int64) + nd
    te_y = test['label'].values.astype(np.float32)
    return Ahat, X, (tr_d, tr_s, tr_y), (te_d, te_s, te_y)


class GCN(nn.Module):
    def __init__(self, n, hid=HID):
        super().__init__()
        self.w1 = nn.Linear(n, hid, bias=False)
        self.w2 = nn.Linear(hid, hid, bias=False)
        self.dec = nn.Sequential(nn.Linear(hid * 2, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, Ahat, X, d, s):
        h = torch.relu(Ahat @ (X @ self.w1.weight.T))
        h = Ahat @ self.w2(h)
        return self.dec(torch.cat([h[d], h[s]], 1)).squeeze(-1)


def evaluate(model, Ahat, X, te):
    d, s, y = te
    model.eval()
    with torch.no_grad():
        h = torch.relu(Ahat @ (X @ model.w1.weight.T))
        h = Ahat @ model.w2(h)
        scores = []
        for i in range(0, len(d), 65536):
            dd, ss = d[i:i + 65536], s[i:i + 65536]
            scores.append(model.dec(torch.cat([h[dd], h[ss]], 1)).squeeze(-1).numpy())
        p = np.concatenate(scores)
    return roc_auc_score(y, p), average_precision_score(y, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=['C', 'F', 'DDCD'])
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--mode', default='regular', choices=['regular', 'cold-drug'])
    ap.add_argument('--epochs', type=int, default=EPOCHS)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    Ahat, X, tr, te = load(args.dataset, args.seed, args.mode)
    n = Ahat.shape[0]
    Ahat_t = torch.tensor(Ahat)
    X_t = torch.tensor(X)
    tr_d, tr_s, tr_y = (torch.tensor(x) for x in tr)
    pos_idx = np.where(tr[2] == 1)[0]
    neg_idx = np.where(tr[2] == 0)[0]

    model = GCN(n)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    lossf = nn.BCEWithLogitsLoss()
    for ep in range(args.epochs):
        model.train()
        rng = np.random.RandomState(args.seed + ep)
        pi = rng.choice(pos_idx, min(len(pos_idx), 16384), replace=False)
        ni = rng.choice(neg_idx, min(len(neg_idx), 16384), replace=False)
        idx = np.concatenate([pi, ni])
        rng.shuffle(idx)
        tot = 0.0
        for i in range(0, len(idx), BATCH):
            b = idx[i:i + BATCH]
            yb = tr_y[b]
            loss = lossf(model(Ahat_t, X_t, tr_d[b], tr_s[b]),
                         torch.tensor(yb))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(b)
    auroc, aupr = evaluate(model, Ahat_t, X_t, te)
    print(f'{args.dataset} s{args.seed} {args.mode}: AUROC={auroc:.4f} AUPR={aupr:.4f}')


if __name__ == '__main__':
    main()
