"""
方案A: MoLFormer 深度分子表征管线 (build_molformer_features)

输入: 三数据集 Features 的 DrugSmile (DrugBank ID + SMILES)
输出:
  code/results/molformer/{DS}_embeddings.npy    (n_drugs × 768, 行序 = 矩阵索引序)
  code/results/molformer/{DS}_similarity.csv    (n_drugs × n_drugs 余弦相似度, 索引=DrugBank ID)
  code/results/molformer/{DS}_report.json       (覆盖/嵌入统计)

嵌入方式: MoLFormer-XL-both-10pct (IBM, Nat. Mach. Intell. 2022), mean-pooling over tokens.
只依赖 SMILES → 冷启动药物天然有信号 (方案A 核心卖点).

用法:
  python code/build_molformer_features.py --dataset C
  python code/build_molformer_features.py --dataset all
"""
import argparse
import json
import os
import numpy as np
import pandas as pd
import torch

FEATURES = {
    'C': ('data/C-Dataset/Features/drug_features_C.csv', 'DrugSmile'),
    'F': ('data/F-Dataset/Features/drugs_features_df.csv', 'DrugSmile'),
    'DDCD': ('data/DDCD/Features/drugsInfo.csv', 'DrugSmile'),
}
OUT_DIR = 'code/results/molformer'
MODEL_NAME = 'ibm-research/MoLFormer-XL-both-10pct'
BATCH = 32


def load_smiles(ds):
    path, smile_col = FEATURES[ds]
    df = pd.read_csv(path)
    df['DrugID'] = df['DrugID'].astype(str).str.strip()
    df[smile_col] = df[smile_col].astype(str).str.strip()
    df = df[df[smile_col].notna() & (df[smile_col] != '')]
    df = df.drop_duplicates(subset=['DrugID'])
    return df[['DrugID', smile_col]].rename(columns={smile_col: 'SMILES'})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset', required=True, choices=['C', 'F', 'DDCD', 'all'])
    args = ap.parse_args()

    os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    print(f"MoLFormer 就绪 (device={device})")

    datasets = ['C', 'F', 'DDCD'] if args.dataset == 'all' else [args.dataset]
    for ds in datasets:
        print(f"\n{'='*60}\n MoLFormer 表征: {ds}\n{'='*60}")
        sm = load_smiles(ds)
        print(f"  药物数: {len(sm):,}")

        embs = []
        for i in range(0, len(sm), BATCH):
            batch_smiles = sm['SMILES'].iloc[i:i + BATCH].tolist()
            enc = tok(batch_smiles, padding=True, truncation=True, max_length=512,
                      return_tensors='pt', return_token_type_ids=False).to(device)
            with torch.no_grad():
                last = model(**enc).last_hidden_state
                mask = enc['attention_mask'].unsqueeze(-1).float()
                emb = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            embs.append(emb.cpu().numpy())
            if (i // BATCH) % 10 == 0:
                print(f"  {min(i + BATCH, len(sm))}/{len(sm)}", flush=True)
        E = np.vstack(embs).astype(np.float32)
        print(f"  嵌入: {E.shape}")

        # 中心化余弦相似度矩阵 (DB 索引, 与现有 drug sim 矩阵对齐)
        # 经验: 原始 MoLFormer 余弦挤在 0.94-0.96, 区分度弱 (拓扑相似对 vs 无关对 差 0.015);
        #       中心化后差 0.117 (8 倍), 信息量显著提升.
        Ec = E - E.mean(axis=0, keepdims=True)
        En = Ec / np.linalg.norm(Ec, axis=1, keepdims=True).clip(min=1e-9)
        S = En @ En.T
        np.fill_diagonal(S, 1.0)
        sim_df = pd.DataFrame(S, index=sm['DrugID'].values, columns=sm['DrugID'].values)

        os.makedirs(OUT_DIR, exist_ok=True)
        np.save(os.path.join(OUT_DIR, f'{ds}_embeddings.npy'), E)
        sim_df.to_csv(os.path.join(OUT_DIR, f'{ds}_similarity.csv'))
        rep = {'dataset': ds, 'n_drugs': len(sm), 'embed_dim': int(E.shape[1]),
               'sim_range': [float(S.min()), float(S.max())],
               'device': device}
        with open(os.path.join(OUT_DIR, f'{ds}_report.json'), 'w', encoding='utf-8') as f:
            json.dump(rep, f, indent=2)
        print(f"  ✅ 相似度矩阵 {S.shape} [{S.min():.4f}, {S.max():.4f}] -> {OUT_DIR}/")


if __name__ == '__main__':
    main()
