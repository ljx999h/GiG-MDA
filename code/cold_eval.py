"""
方案A 冷启动评估 (cold_eval): MolFormer 分子通道在 cold-drug 场景的价值

场景: 20% 药物为冷药物 (train 零关联) → GBA 特征退化 (无邻居) / GRMF 嵌入退化
      只有 MoLFormer 分子表征 (SMILES 衍生) 提供信号.
对比变体 (统一 XGBoost 标准配置 + 方案B):
  MiRAGE(18) / MiRAGE+MolFormer(20) / +embed / +MolFormer+embed

用法: python code/cold_eval.py --dataset C
"""
import argparse
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             precision_recall_curve)

import sys
sys.path.insert(0, 'code')
import r2_config

MOL_FEATS = ['p_score_MolFormer', 'adj_p_score_MolFormer']
CFG = {
    'C': dict(manifest='data/C-Dataset/Splits/split_manifest_cold.csv',
              score='code/results/MiRAGE_score_C_cold.csv',
              gigs='data/C-Dataset/Splits/gigs_split_C_cold.pkl',
              train_pos='data/C-Dataset/Splits/Cold/train_positives.csv',
              train_neg='data/C-Dataset/Splits/Cold/train_negatives.csv',
              mapping='data/C-Dataset/Mapping/mapping_C.csv', id_space='C-index'),
    'F': dict(manifest='data/F-Dataset/Splits/split_manifest_cold.csv',
              score='code/results/MiRAGE_score_F_cold.csv',
              gigs='data/F-Dataset/Splits/gigs_split_F_cold.pkl',
              train_pos='data/F-Dataset/Splits/Cold/train_positives.csv',
              train_neg='data/F-Dataset/Splits/Cold/train_negatives.csv',
              mapping='data/F-Dataset/Mapping/mapping_F.csv', id_space='string'),
    'DDCD': dict(manifest='data/DDCD/Splits/split_manifest_cold.csv',
                 score='code/results/MiRAGE_score_DDCD_cold.csv',
                 gigs='data/DDCD/Splits/gigs_split_DDCD_cold.pkl',
                 train_pos='data/DDCD/Splits/Cold/train_positives.csv',
                 train_neg='data/DDCD/Splits/Cold/train_negatives.csv',
                 mapping='data/DDCD/Mapping/mapping.csv', id_space='string'),
}


def load(ds, seed=42, mode='cold-drug'):
    """加载冷启动数据. mode: cold-drug (默认) / cold-disease. seed: 划分种子."""
    cfg = dict(CFG[ds])
    _BASE = {'cold-drug': ('cold', ''), 'cold-disease': ('colddis', 'dis'),
             'scaffold': ('scaffold', 'scaf')}
    base, dis = _BASE[mode]
    tag = f'{base}_s{seed}' if seed != 42 else base
    cfg['manifest'] = cfg['manifest'].replace('_cold', f'_{tag}')
    cfg['score'] = cfg['score'].replace('_cold', f'_{tag}')
    cfg['gigs'] = cfg['gigs'].replace('_cold', f'_{tag}')
    parts = ['Cold']
    if dis:
        parts.append(dis)
    if seed != 42:
        parts.append(f's{seed}')
    dirname = '_'.join(parts)
    cfg['train_pos'] = cfg['train_pos'].replace('Cold', dirname)
    cfg['train_neg'] = cfg['train_neg'].replace('Cold', dirname)

    feats = r2_config.DATASETS[ds]['feats']
    base_feats = [c for c in feats if c not in MOL_FEATS]
    pos = pd.read_csv(cfg['train_pos']).fillna(0.0)
    neg = pd.read_csv(cfg['train_neg']).fillna(0.0)
    keep = ['drugID', 'diseaseID', 'label'] + base_feats
    train = pd.concat([pos[keep], neg[keep]], ignore_index=True)

    mf = pd.read_csv(cfg['manifest'])
    test_mf = mf[mf['split'] == 'test'][['drugID', 'diseaseID', 'label']].copy()
    test_mf['drugID'] = test_mf['drugID'].astype(str).str.strip()
    test_mf['diseaseID'] = test_mf['diseaseID'].astype(str).str.strip()

    score = pd.read_csv(cfg['score']).fillna(0.0)
    score['drugID'] = score['drugID'].astype(str).str.strip()
    score['diseaseID'] = score['diseaseID'].astype(str).str.strip()
    if 'label' in score.columns:
        score = score.drop(columns=['label'])
    if cfg['id_space'] == 'C-index':
        mapping = pd.read_csv(cfg['mapping'])
        if 'Unnamed: 0' in mapping.columns:
            mapping = mapping.drop(columns=['Unnamed: 0'])
        drugbanks = sorted(mapping.iloc[:, 0].astype(str).str.strip().unique())
        idx_to_db = {i: db for i, db in enumerate(drugbanks)}
        score['drugID'] = score['drugID'].map(lambda x: idx_to_db.get(int(x), x))
    test = test_mf.merge(score, on=['drugID', 'diseaseID'], how='inner')
    assert len(test) == len(test_mf), 'test 特征缺失'

    with open(cfg['gigs'], 'rb') as f:
        gigs = pickle.load(f)

    for df_ in (train, test):
        df_['drugID'] = df_['drugID'].astype(str).str.strip()
        df_['diseaseID'] = df_['diseaseID'].astype(str).str.strip()
    # MolFormer 相似度列从 score merge 进 train; 若特征文件已 --exclude MolFormer
    # (P0-1: 委员会负采样与主结果不得含预训练相似度列), 缺失列置零兼容
    if set(MOL_FEATS) <= set(score.columns):
        mol = score[score['drugID'].notna()]
        train = train.merge(mol[['drugID', 'diseaseID'] + MOL_FEATS],
                            on=['drugID', 'diseaseID'], how='left').fillna(0.0)
    else:
        for c in MOL_FEATS:
            train[c] = 0.0
    return train, test, gigs, feats, base_feats


def gigs_feats(df, gigs):
    X, Y = gigs['X'], gigs['Y']
    d2i, s2i = gigs['drug_to_idx'], gigs['disease_to_idx']
    k = X.shape[1]
    d = df['drugID'].astype(str).str.strip().map(d2i)
    s = df['diseaseID'].astype(str).str.strip().map(s2i)
    v = d.notna() & s.notna()
    dot = np.zeros(len(df)); emb = np.zeros((len(df), 2 * k))
    if v.any():
        dv, sv = d[v].astype(int).values, s[v].astype(int).values
        dot[v.values] = np.sum(X[dv] * Y[sv], axis=1)
        emb[v.values, :k] = X[dv]; emb[v.values, k:] = Y[sv]
    return dot, emb


def evaluate(Xtr, ytr, Xte, yte, seed=42):
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    threshs = []
    for tr, va in skf.split(Xtr, ytr):
        clf = xgb.XGBClassifier(**r2_config.XGB_CONFIG)
        clf.fit(Xtr[tr], ytr[tr])
        yv = clf.predict_proba(Xtr[va])[:, 1]
        p, r, t = precision_recall_curve(ytr[va], yv)
        f1s = 2 * p * r / (p + r + 1e-9)
        best = f1s.argmax()
        threshs.append(t[best] if best < len(t) else 0.5)
    clf = xgb.XGBClassifier(**r2_config.XGB_CONFIG)
    clf.fit(Xtr, ytr)
    yp = clf.predict_proba(Xte)[:, 1]
    return roc_auc_score(yte, yp), average_precision_score(yte, yp)


def load_mol_emb(ds):
    """MoLFormer 嵌入 (行序 = features 文件 DrugID 序)."""
    import json
    E = np.load(f'code/results/molformer/{ds}_embeddings.npy')
    feat_path, _ = {'C': ('data/C-Dataset/Features/drug_features_C.csv', 'DrugSmile'),
                    'F': ('data/F-Dataset/Features/drugs_features_df.csv', 'DrugSmile'),
                    'DDCD': ('data/DDCD/Features/drugsInfo.csv', 'DrugSmile')}[ds]
    sm = pd.read_csv(feat_path)
    ids = sm['DrugID'].astype(str).str.strip().values
    return dict(zip(ids, E))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=['C', 'F', 'DDCD'])
    ap.add_argument('--seed', type=int, default=42, help='冷划分种子 (路径带 _s{seed} 后缀)')
    ap.add_argument('--mode', default='cold-drug',
                  choices=['cold-drug', 'cold-disease', 'scaffold'])
    args = ap.parse_args()
    ds = args.dataset
    train, test, gigs, feats, base = load(ds, args.seed, args.mode)
    mol = feats + MOL_FEATS
    tr_dot, tr_emb = gigs_feats(train, gigs)
    te_dot, te_emb = gigs_feats(test, gigs)
    ytr, yte = train['label'].values, test['label'].values

    # MoLFormer 嵌入 (PCA 降维, train 内 fit)
    emb_map = load_mol_emb(ds)
    from sklearn.decomposition import PCA
    for k in (32, 64):
        pca = PCA(n_components=k, random_state=42)
        tr_raw = np.array([emb_map.get(d, np.zeros(768)) for d in train['drugID']], dtype=np.float32)
        te_raw = np.array([emb_map.get(d, np.zeros(768)) for d in test['drugID']], dtype=np.float32)
        # PCA 拟合在 unique 训练药物嵌入上 (每个药物一次), 再映射回 pair 行
        uniq_d = pd.unique(train['drugID'].astype(str).str.strip())
        uniq_raw = np.array([emb_map.get(d, np.zeros(768)) for d in uniq_d], dtype=np.float32)
        pca.fit(uniq_raw)
        globals()[f'tr_pca{k}'] = pca.transform(tr_raw).astype(np.float32)
        globals()[f'te_pca{k}'] = pca.transform(te_raw).astype(np.float32)

    variants = {
        'MiRAGE':             (train[base].values, test[base].values),
        'MiRAGE+dot':         (np.hstack([train[base].values, tr_dot[:, None]]),
                               np.hstack([test[base].values, te_dot[:, None]])),
        'MiRAGE+embed':       (np.hstack([train[base].values, tr_emb]),
                               np.hstack([test[base].values, te_emb])),
        'MiRAGE+MolEmb32':    (np.hstack([train[base].values, tr_pca32]),
                               np.hstack([test[base].values, te_pca32])),
        'MiRAGE+MolEmb64':    (np.hstack([train[base].values, tr_pca64]),
                               np.hstack([test[base].values, te_pca64])),
        'MiRAGE+embed+MolEmb32': (np.hstack([train[base].values, tr_emb, tr_pca32]),
                                  np.hstack([test[base].values, te_emb, te_pca32])),
    }
    print(f"\n=== {ds} 冷启动 (cold-drug 20%) 方案A 验证 ===")
    print(f"冷 test: {len(test):,} 对 (正{(yte==1).sum():,})")
    print(f"{'variant':<24} {'dim':>4} {'AUROC':>7} {'AUPR':>7}")
    for name, (Xtr, Xte) in variants.items():
        Xtr = Xtr.astype(np.float32); Xte = Xte.astype(np.float32)
        a, p = evaluate(Xtr, ytr, Xte, yte)
        print(f"{name:<24} {Xtr.shape[1]:>4} {a:>7.4f} {p:>7.4f}")


if __name__ == '__main__':
    main()
