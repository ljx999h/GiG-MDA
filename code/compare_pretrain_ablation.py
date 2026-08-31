"""
A: 预训练有效性对照 (compare_pretrain_ablation)

回答: MolEmb32 的冷启动收益来自"预训练知识"还是"任何分子特征/维度"?
对照 (同维度、同分类器、同协议):
  MiRAGE (base)          — 无分子通道
  +MolEmb32 (现有)       — MoLFormer 预训练表征 (目标)
  +ECFP32                — Morgan 指纹 (radius=2, 1024 bit) → PCA-32 (简单结构指纹是否已足够)
  +Shuffled32            — MoLFormer 嵌入按药物打乱 → PCA-32 (随机特征是否同样有效 → 检验预训练知识)

评估: 冷启动 (cold-drug 20%, 复用现有 4 个划分), 方案B, 统一 XGBoost.
输出: results/R2/pretrain_ablation.csv

用法: python code/compare_pretrain_ablation.py --dataset C
"""
import argparse
import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

import sys
sys.path.append('docx_check')
from cold_eval import load, evaluate, load_mol_emb

FEAT_PATH = {
    'C': 'data/C-Dataset/Features/drug_features_C.csv',
    'F': 'data/F-Dataset/Features/drugs_features_df.csv',
    'DDCD': 'data/DDCD/Features/drugsInfo.csv',
}


def mol_feats(df, emb_map, seed, shuffle=False, n_comp=32):
    """MoLFormer 嵌入 → (可选 drug-level 打乱) → 返回 pair 行原始嵌入.
    drug-level 语义: 对 unique 药物嵌入集合整体置换 (每个药物一个置换后嵌入,
    该药物的所有行共享), 而非行级打乱."""
    raw = np.array([emb_map.get(d, np.zeros(768)) for d in df['drugID']], dtype=np.float32)
    if shuffle:
        rng = np.random.RandomState(seed)
        uni, inv = np.unique(raw, axis=0, return_inverse=True)
        perm = rng.permutation(len(uni))
        raw = uni[perm[inv]]
    return raw


def ecfp_feats(ds, df, seed, n_comp=32):
    """Morgan 指纹 (radius=2, 1024 bit) → PCA-n_comp."""
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    sm = pd.read_csv(FEAT_PATH[ds])
    smiles_map = dict(zip(sm['DrugID'].astype(str).str.strip(), sm['DrugSmile']))
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    vecs = []
    for d in df['drugID']:
        smi = smiles_map.get(d, '')
        m = Chem.MolFromSmiles(smi) if smi else None
        if m is None:
            vecs.append(np.zeros(1024, dtype=np.float32))
        else:
            vecs.append(np.array(gen.GetFingerprint(m), dtype=np.float32))
    return np.vstack(vecs)


def fit_transform(tr_raw, te_raw, seed, n_comp=32, uniq_raw=None):
    pca = PCA(n_components=n_comp, random_state=42)  # 与 cold_eval 一致: PCA 投影种子固定 42, 而非划分种子
    # unique 训练药物嵌入上拟合 (drugID 级, 与 cold_eval 语义一致), 再映射回 pair 行
    pca.fit(uniq_raw if uniq_raw is not None else np.unique(tr_raw, axis=0))
    return pca.transform(tr_raw).astype(np.float32), pca.transform(te_raw).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset', required=True, choices=['C', 'F', 'DDCD'])
    ap.add_argument('--seeds', nargs='+', type=int, default=[42, 7, 123, 2024])
    args = ap.parse_args()
    ds = args.dataset

    rows = []
    emb_map = load_mol_emb(ds)
    for seed in args.seeds:
        train, test, gigs, feats, base = load(ds, seed, 'cold-drug')
        ytr, yte = train['label'].values, test['label'].values

        tr_mol = mol_feats(train, emb_map, seed, shuffle=False)
        te_mol = mol_feats(test, emb_map, seed, shuffle=False)
        uniq_d = pd.unique(train['drugID'].astype(str).str.strip())
        uniq_df = pd.DataFrame({'drugID': uniq_d})
        tr_mol32, te_mol32 = fit_transform(tr_mol, te_mol, seed,
                                           uniq_raw=mol_feats(uniq_df, emb_map, seed, shuffle=False))

        tr_ecfp = ecfp_feats(ds, train, seed)
        te_ecfp = ecfp_feats(ds, test, seed)
        tr_ecfp32, te_ecfp32 = fit_transform(tr_ecfp, te_ecfp, seed)

        tr_shuf = mol_feats(train, emb_map, seed, shuffle=True)
        te_shuf = mol_feats(test, emb_map, seed, shuffle=True)
        tr_shuf32, te_shuf32 = fit_transform(tr_shuf, te_shuf, seed,
                                             uniq_raw=mol_feats(uniq_df, emb_map, seed, shuffle=True))

        variants = {
            'MiRAGE':        (train[base].values, test[base].values),
            'MiRAGE+MolEmb32': (np.hstack([train[base].values, tr_mol32]),
                                np.hstack([test[base].values, te_mol32])),
            'MiRAGE+ECFP32': (np.hstack([train[base].values, tr_ecfp32]),
                              np.hstack([test[base].values, te_ecfp32])),
            'MiRAGE+Shuffled32': (np.hstack([train[base].values, tr_shuf32]),
                                  np.hstack([test[base].values, te_shuf32])),
        }
        print(f"\n=== {ds} cold-drug s{seed} ===", flush=True)
        for name, (Xtr, Xte) in variants.items():
            Xtr = Xtr.astype(np.float32); Xte = Xte.astype(np.float32)
            a, p = evaluate(Xtr, ytr, Xte, yte)
            rows.append({'dataset': ds, 'seed': seed, 'variant': name, 'AUROC': a, 'AUPR': p})
            print(f"  {name:<20} AUROC={a:.4f} AUPR={p:.4f}", flush=True)

    df = pd.DataFrame(rows)
    os.makedirs('results/R2', exist_ok=True)
    out = 'results/R2/pretrain_ablation.csv'
    df.to_csv(out, index=False)
    print(f"\n✅ 已保存: {out}")
    # 汇总: 各变体相对 base 的 lift
    piv = df.pivot_table(index=['dataset', 'seed'], columns='variant', values='AUPR')
    print("\n=== 相对 MiRAGE 的 AUPR lift ===")
    for v in ['MiRAGE+MolEmb32', 'MiRAGE+ECFP32', 'MiRAGE+Shuffled32']:
        lift = (piv[v] / piv['MiRAGE'] - 1) * 100
        print(f"{v:<22} mean={lift.mean():+.1f}%  per-seed={[f'{x:+.0f}%' for x in lift]}")


if __name__ == '__main__':
    main()
