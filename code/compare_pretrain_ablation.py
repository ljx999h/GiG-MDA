"""
A: 预训练有效性对照 (compare_pretrain_ablation)

回答: MolEmb32 的冷启动收益来自"预训练知识"还是"任何分子特征/维度"?
对照 (同维度、同分类器、同协议):
  MiRAGE (base)          — 无分子通道
  +MolEmb32 (现有)       — MoLFormer 预训练表征 (目标)
  +ECFP32                — Morgan 指纹 (radius=2, 1024 bit) → PCA-32 (简单结构指纹是否已足够)
  +Shuffled32            — MoLFormer 嵌入按药物打乱 → PCA-32 (随机特征是否同样有效 → 检验预训练知识)
  +Gaussian32            — 维度与边际方差匹配的 drug-level 高斯随机向量 (恒定每药向量对照)

评审修复 (2026-09-03, 新评审 P0-1):
  1. Gaussian32 原实现按 (pair) 行采样 → 改为 drug-level: 每个药物一个恒定随机向量,
     该药物的所有行共享 (train/test 通过同一 drugID→向量 字典映射).
  2. Shuffled32 原实现 train/test 两侧各自独立置换 → 改为在全部药物 (train∪test) 上
     一次置换, 两侧共享同一映射.
  3. 随机对照重复 --reps 次 (默认 20), 输出 mean±SD / paired ΔAUPR / 经验 p 值
     (p = (#reps 中对照 ≥ MolEmb32 + 1) / (reps + 1), 保守单侧).
  4. 评估提速: 每 rep 只做一次最终拟合 (evaluate() 的 5 折仅用于选阈值, AUPR/AUROC
     不依赖阈值 → 数值与旧 evaluate() 一致).

评估: 冷启动 (cold-drug 20%, 复用现有 4 个划分), 方案B, 统一 XGBoost.
输出: results/R2/pretrain_ablation.csv  (随机对照每 rep 一行; 静态变体 rep=0)

用法: python code/compare_pretrain_ablation.py --dataset C --reps 20
"""
import argparse
import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, average_precision_score

import sys
sys.path.append('code')
import r2_config
from cold_eval import load, load_mol_emb

FEAT_PATH = {
    'C': 'data/C-Dataset/Features/drug_features_C.csv',
    'F': 'data/F-Dataset/Features/drugs_features_df.csv',
    'DDCD': 'data/DDCD/Features/drugsInfo.csv',
}


def row_embeddings(drug_ids, emb_map):
    """drugID 列表 → 768-d 原始嵌入 (行序 = drug_ids)."""
    return np.array([emb_map.get(d, np.zeros(768)) for d in drug_ids], dtype=np.float32)


def apply_map(df, drug_vec_map, pca):
    """pair 行 → 按 drugID 取恒定向量 → PCA 投影 (MolEmb32/对照统一路径)."""
    d0 = np.zeros(next(iter(drug_vec_map.values())).shape, dtype=np.float32)
    v = np.array([drug_vec_map.get(d, d0) for d in df['drugID']])
    return pca.transform(v).astype(np.float32)


def shuffled_map(union_drugs, emb_map, rng):
    """在全部药物上做一次置换: 药物 d 获得 π(d) 的原始嵌入 (train/test 共享)."""
    base = row_embeddings(union_drugs, emb_map)
    perm = rng.permutation(len(union_drugs))
    return dict(zip(union_drugs, base[perm]))


def gaussian_map(union_drugs, uniq_raw_train, rng):
    """drug-level 高斯: 每药物一个 32-d 恒定向量, 边际方差匹配训练 unique 药物嵌入."""
    var = uniq_raw_train.var(0) + 1e-9
    vecs = rng.normal(0, np.sqrt(var), size=(len(union_drugs), var.shape[0])).astype(np.float32)
    return dict(zip(union_drugs, vecs))


def aupr_auc_fit(Xtr, ytr, Xte, yte):
    clf = xgb.XGBClassifier(**r2_config.XGB_CONFIG)
    clf.fit(Xtr, ytr)
    yp = clf.predict_proba(Xte)[:, 1]
    return roc_auc_score(yte, yp), average_precision_score(yte, yp)


def ecfp_feats(ds, df):
    """Morgan 指纹 (radius=2, 1024 bit), pair 行序."""
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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset', required=True, choices=['C', 'F', 'DDCD'])
    ap.add_argument('--seeds', nargs='+', type=int, default=[42, 7, 123, 2024])
    ap.add_argument('--reps', type=int, default=20, help='随机对照重复次数 (默认 20)')
    ap.add_argument('--out', default='results/R2/pretrain_ablation.csv')
    args = ap.parse_args()
    ds = args.dataset
    REPS = args.reps

    rows = []
    emb_map = load_mol_emb(ds)
    for seed in args.seeds:
        train, test, gigs, feats, base = load(ds, seed, 'cold-drug')
        ytr, yte = train['label'].values, test['label'].values
        Xb_tr = train[base].values.astype(np.float32)
        Xb_te = test[base].values.astype(np.float32)
        tr_drugs = list(pd.unique(train['drugID'].astype(str).str.strip()))
        te_drugs = list(pd.unique(test['drugID'].astype(str).str.strip()))
        union_drugs = list(dict.fromkeys(tr_drugs + te_drugs))
        uniq_raw_train = row_embeddings(tr_drugs, emb_map)

        print(f"\n=== {ds} cold-drug s{seed} ===", flush=True)

        # ---- 静态变体: base / MolEmb32 / ECFP32 (无随机性, 每组合一次) ----
        a0, p0 = aupr_auc_fit(Xb_tr, ytr, Xb_te, yte)
        rows.append({'dataset': ds, 'seed': seed, 'rep': 0, 'variant': 'MiRAGE',
                     'AUROC': a0, 'AUPR': p0})
        print(f'  MiRAGE          AUROC={a0:.4f} AUPR={p0:.4f}', flush=True)

        # MolEmb32: PCA fit 于 train unique 嵌入, train/test 行经同一全药物 map 取向量
        pca_mol = PCA(n_components=32, random_state=42)
        pca_mol.fit(uniq_raw_train)
        full_mol_map = dict(zip(union_drugs, row_embeddings(union_drugs, emb_map)))
        tr_mol = np.hstack([Xb_tr, apply_map(train, full_mol_map, pca_mol)])
        te_mol = np.hstack([Xb_te, apply_map(test, full_mol_map, pca_mol)])
        a, p = aupr_auc_fit(tr_mol, ytr, te_mol, yte)
        rows.append({'dataset': ds, 'seed': seed, 'rep': 0, 'variant': 'MiRAGE+MolEmb32',
                     'AUROC': a, 'AUPR': p})
        print(f'  MiRAGE+MolEmb32 AUROC={a:.4f} AUPR={p:.4f}', flush=True)
        mol_aupr = p

        tr_ecfp = ecfp_feats(ds, train)
        te_ecfp = ecfp_feats(ds, test)
        pca_ecfp = PCA(n_components=32, random_state=42)
        pca_ecfp.fit(np.unique(tr_ecfp, axis=0))
        tr_e = np.hstack([Xb_tr, pca_ecfp.transform(tr_ecfp).astype(np.float32)])
        te_e = np.hstack([Xb_te, pca_ecfp.transform(te_ecfp).astype(np.float32)])
        a, p = aupr_auc_fit(tr_e, ytr, te_e, yte)
        rows.append({'dataset': ds, 'seed': seed, 'rep': 0, 'variant': 'MiRAGE+ECFP32',
                     'AUROC': a, 'AUPR': p})
        print(f'  MiRAGE+ECFP32   AUROC={a:.4f} AUPR={p:.4f}', flush=True)

        # ---- 随机对照: 每 rep 重新抽样 (drug-level, train/test 共享映射) ----
        for rep in range(REPS):
            rng = np.random.RandomState(seed * 100000 + rep)
            sh_map = shuffled_map(union_drugs, emb_map, rng)
            g_map = gaussian_map(union_drugs, uniq_raw_train, rng)
            pca_sh = PCA(n_components=32, random_state=42)
            pca_sh.fit(np.array([sh_map[d] for d in tr_drugs]))
            pca_g = PCA(n_components=32, random_state=42)
            pca_g.fit(np.array([g_map[d] for d in tr_drugs]))
            for name, mp, pca_ in [('MiRAGE+Shuffled32', sh_map, pca_sh),
                                   ('MiRAGE+Gaussian32', g_map, pca_g)]:
                tr_X = np.hstack([Xb_tr, apply_map(train, mp, pca_)])
                te_X = np.hstack([Xb_te, apply_map(test, mp, pca_)])
                a, p = aupr_auc_fit(tr_X, ytr, te_X, yte)
                rows.append({'dataset': ds, 'seed': seed, 'rep': rep,
                             'variant': name, 'AUROC': a, 'AUPR': p})
            if (rep + 1) % 5 == 0:
                print(f'  rep {rep + 1}/{REPS} done', flush=True)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    out = args.out
    df.to_csv(out, index=False)
    print(f"\n✅ 已保存: {out} ({len(df)} rows)")

    # ---- 汇总 ----
    piv = df[df['rep'] == 0].pivot_table(index=['dataset', 'seed'], columns='variant',
                                         values='AUPR')
    for ctrl in ['MiRAGE+Shuffled32', 'MiRAGE+Gaussian32']:
        c = df[df['variant'] == ctrl].pivot_table(index=['dataset', 'seed'],
                                                  columns='rep', values='AUPR')
        piv[ctrl + '_mean'] = c.mean(axis=1)
        piv[ctrl + '_sd'] = c.std(axis=1)
        piv[ctrl + '_lift_mean'] = (piv[ctrl + '_mean'] / piv['MiRAGE'] - 1) * 100
        piv[ctrl + '_p_vs_mol'] = (
            (c.values >= piv['MiRAGE+MolEmb32'].values[:, None]).sum(axis=1) + 1
        ) / (c.shape[1] + 1)
    print('\n=== 汇总 (每 (dataset, seed) × 20 reps) ===')
    print(piv.to_string(float_format=lambda x: f'{x:.4f}'))


if __name__ == '__main__':
    main()
