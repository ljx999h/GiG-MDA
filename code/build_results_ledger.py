"""
R2 统一 runner → 唯一结果账本 (build_results_ledger)

对每个数据集 [C, F, DDCD] × 每个模型, 跑统一方案B:
  train 内 5-fold CV 按 PR 曲线 F1 最大化选阈值 (取平均) → 全量训练 →
  封存 test 只评一次 → 1000 次 bootstrap CI.
所有 XGBoost 家族 (GiG-MDA 三种特征变体 + XGBoost baseline) 共用同一标准配置
(r2_config.XGB_CONFIG: lr=0.1, 无 scale_pos_weight/subsample/colsample).

模型:
  特征变体 (XGBoost): MiRAGE(18/22), MiRAGE+dot(19/23), MiRAGE+embed(146/150)=GiG-MDA
  baselines (各自工厂): LogisticRegression, PREDICT, KNN, RandomForest, XGBoost, LightGBM
    baseline 特征 = MiRAGE + GiGs 1-D dot (与论文叙述一致)

输出: results/R2/results_manifest.csv (长格式, 每行 = dataset × model × seed)

用法:
  python code/build_results_ledger.py                 # 全部数据集
  python code/build_results_ledger.py --dataset C     # 单数据集
"""
import argparse
import json
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    f1_score, recall_score, precision_score, precision_recall_curve
)

import xgboost as xgb

import r2_config
from eval_sota_baselines import (
    make_lr, make_knn, make_rf, make_xgb, make_lgb, make_predict
)
from eval_holdout_protocol import bootstrap_ci

# 统一 XGBoost 配置 (标准配置) — GiG-MDA 变体与 XGBoost baseline 共用
def make_unified_xgb():
    return xgb.XGBClassifier(**r2_config.XGB_CONFIG)


BASELINE_MODELS = [
    ('LogisticRegression', dict(factory=make_lr, needs_scaler=True)),
    ('PREDICT',            dict(factory=None, needs_scaler=False, is_predict=True)),
    ('KNN',                dict(factory=make_knn, needs_scaler=True)),
    ('RandomForest',       dict(factory=make_rf, needs_scaler=False)),
    ('XGBoost',            dict(factory=make_unified_xgb, needs_scaler=False)),
    ('LightGBM',           dict(factory=make_lgb, needs_scaler=False)),
]


def load_score(ds, cfg):
    df = pd.read_csv(cfg['score']).fillna(0.0)
    df['drugID'] = df['drugID'].astype(str).str.strip()
    df['diseaseID'] = df['diseaseID'].astype(str).str.strip()
    if 'label' in df.columns:
        df = df.drop(columns=['label'])
    # C: MiRAGE_score 的 drugID 是整数索引, 映射回 DB 字符串
    if ds == 'C':
        mapping = pd.read_csv(cfg['mapping'])
        if 'Unnamed: 0' in mapping.columns:
            mapping = mapping.drop(columns=['Unnamed: 0'])
        drugbanks = sorted(mapping.iloc[:, 0].astype(str).str.strip().unique())
        idx_to_db = {i: db for i, db in enumerate(drugbanks)}
        df['drugID'] = df['drugID'].map(lambda x: idx_to_db.get(int(x), x))
    return df


def load_gigs(cfg):
    with open(cfg['gigs'], 'rb') as f:
        return pickle.load(f)


def mol_emb_features(ds, train_df, test_df, n_comp=32, seed=42):
    """MoLFormer 嵌入 (768D) → PCA n_comp 维; PCA 只在 train 上拟合 (泄漏安全).
    返回 (train_pca, test_pca) float32."""
    from sklearn.decomposition import PCA
    feat_path = {'C': 'data/C-Dataset/Features/drug_features_C.csv',
                 'F': 'data/F-Dataset/Features/drugs_features_df.csv',
                 'DDCD': 'data/DDCD/Features/drugsInfo.csv'}[ds]
    sm = pd.read_csv(feat_path)
    E = np.load(f'code/results/molformer/{ds}_embeddings.npy')
    emb_map = dict(zip(sm['DrugID'].astype(str).str.strip().values, E))
    tr_raw = np.array([emb_map.get(d, np.zeros(E.shape[1])) for d in train_df['drugID']],
                      dtype=np.float32)
    te_raw = np.array([emb_map.get(d, np.zeros(E.shape[1])) for d in test_df['drugID']],
                      dtype=np.float32)
    pca = PCA(n_components=n_comp, random_state=seed)
    # PCA 拟合在 unique 训练药物嵌入上 (每个药物一次), 再映射回 pair 行
    uniq_d = pd.unique(train_df['drugID'].astype(str).str.strip())
    uniq_raw = np.array([emb_map.get(d, np.zeros(E.shape[1])) for d in uniq_d],
                        dtype=np.float32)
    pca.fit(uniq_raw)
    return pca.transform(tr_raw).astype(np.float32), pca.transform(te_raw).astype(np.float32)


def gigs_features(df, gigs_data):
    """返回 (dot 特征数组, embed 特征数组)."""
    X_emb, Y_emb = gigs_data['X'], gigs_data['Y']
    d2i, s2i = gigs_data['drug_to_idx'], gigs_data['disease_to_idx']
    k = X_emb.shape[1]
    d_idx = df['drugID'].astype(str).str.strip().map(d2i)
    s_idx = df['diseaseID'].astype(str).str.strip().map(s2i)
    valid = d_idx.notna() & s_idx.notna()
    dot = np.zeros(len(df), dtype=np.float32)
    emb = np.zeros((len(df), 2 * k), dtype=np.float32)
    if valid.any():
        dv = d_idx[valid].astype(int).values
        sv = s_idx[valid].astype(int).values
        dot[valid.values] = np.sum(X_emb[dv] * Y_emb[sv], axis=1)
        emb[valid.values, :k] = X_emb[dv]
        emb[valid.values, k:] = Y_emb[sv]
    return dot, emb


def evaluate_holdout(Xtr, ytr, Xte, yte, factory, needs_scaler=False,
                     is_predict=False, mirage_features=None, seed=42, n_boot=1000):
    """统一方案B: train 内 5 折 CV 选阈值 → 全量训练 → test 只评一次 + bootstrap CI."""
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    threshs = []
    for tr, va in skf.split(Xtr, ytr):
        if is_predict:
            clf = make_predict(mirage_features)
            clf.fit(Xtr.iloc[tr], ytr[tr])
            yv = clf.predict_proba(Xtr.iloc[va])[:, 1]
        else:
            clf = factory()
            if needs_scaler:
                scaler = StandardScaler()
                Xtr_s = scaler.fit_transform(Xtr[tr])
                Xva_s = scaler.transform(Xtr[va])
                clf.fit(Xtr_s, ytr[tr])
                yv = clf.predict_proba(Xva_s)[:, 1]
            else:
                clf.fit(Xtr[tr], ytr[tr])
                yv = clf.predict_proba(Xtr[va])[:, 1]
        p, r, t = precision_recall_curve(ytr[va], yv)
        f1s = 2 * p * r / (p + r + 1e-9)
        best = f1s.argmax()
        threshs.append(t[best] if best < len(t) else 0.5)
    avg_thresh = float(np.mean(threshs))

    # 全量训练
    if is_predict:
        clf = make_predict(mirage_features)
        clf.fit(Xtr, ytr)
        yprob = clf.predict_proba(Xte)[:, 1]
    else:
        clf = factory()
        if needs_scaler:
            scaler = StandardScaler()
            Xtr_s = scaler.fit_transform(Xtr)
            Xte_s = scaler.transform(Xte)
            clf.fit(Xtr_s, ytr)
            yprob = clf.predict_proba(Xte_s)[:, 1]
        else:
            clf.fit(Xtr, ytr)
            yprob = clf.predict_proba(Xte)[:, 1]

    auroc = roc_auc_score(yte, yprob)
    aupr = average_precision_score(yte, yprob)
    ypred = (yprob >= avg_thresh).astype(int)
    auroc_lo, auroc_hi = bootstrap_ci(yte, yprob, n_boot, seed, 'auroc')
    aupr_lo, aupr_hi = bootstrap_ci(yte, yprob, n_boot, seed, 'aupr')
    return {
        'AUROC': auroc, 'AUROC_ci_lo': auroc_lo, 'AUROC_ci_hi': auroc_hi,
        'AUPR': aupr, 'AUPR_ci_lo': aupr_lo, 'AUPR_ci_hi': aupr_hi,
        'F1': f1_score(yte, ypred),
        'Accuracy': accuracy_score(yte, ypred),
        'Recall': recall_score(yte, ypred),
        'Precision': precision_score(yte, ypred, zero_division=0),
        'Threshold': avg_thresh,
    }


def run_dataset(ds, cfg, seed, n_boot):
    print(f"\n{'='*60}\n 账本: {ds}\n{'='*60}")

    # 1. 训练集 (P_train 正 + OOF 可靠负, 已嵌入 r2 特征)
    pos = pd.read_csv(cfg['train_pos']).fillna(0.0)
    neg = pd.read_csv(cfg['train_neg']).fillna(0.0)
    feat_cols = [c for c in cfg['feats'] if c in pos.columns]
    assert len(feat_cols) == len(cfg['feats']), f'{ds} 训练特征列缺失'
    keep = ['drugID', 'diseaseID', 'label'] + feat_cols
    train = pd.concat([pos[keep], neg[keep]], ignore_index=True)
    train['drugID'] = train['drugID'].astype(str).str.strip()
    train['diseaseID'] = train['diseaseID'].astype(str).str.strip()
    print(f"  训练集: {len(train):,} (正{(train['label']==1).sum():,} 负{(train['label']==0).sum():,})")

    # 2. 测试集 (manifest test + r2 特征)
    mf = pd.read_csv(cfg['manifest'])
    mf['drugID'] = mf['drugID'].astype(str).str.strip()
    mf['diseaseID'] = mf['diseaseID'].astype(str).str.strip()
    test_mf = mf[mf['split'] == 'test'][['drugID', 'diseaseID', 'label']].copy()
    score = load_score(ds, cfg)
    key = ['drugID', 'diseaseID']
    test = test_mf.merge(score, on=key, how='inner')
    assert len(test) == len(test_mf), \
        f'{ds} test 对缺失特征: {len(test_mf)-len(test):,} (inner join 掉行)'
    print(f"  测试集: {len(test):,} (正{(test['label']==1).sum():,} 负{(test['label']==0).sum():,})")

    # 3. pair-disjoint 审计
    tr_pairs = set(zip(train['drugID'], train['diseaseID']))
    te_pairs = set(zip(test['drugID'], test['diseaseID']))
    assert tr_pairs.isdisjoint(te_pairs), f'{ds} train/test pair 重叠!'
    print(f"  [PASS] train/test pair 零交集")

    # 4. GiGs
    gigs = load_gigs(cfg)
    tr_dot, tr_emb = gigs_features(train, gigs)
    te_dot, te_emb = gigs_features(test, gigs)

    feats = feat_cols
    ytr = train['label'].values
    yte = test['label'].values

    # 5. 特征变体 (统一 XGBoost)
    tr_mol, te_mol = mol_emb_features(ds, train, test, n_comp=32, seed=seed)
    variants = {
        'MiRAGE':       (train[feats].values.astype(np.float32),
                         test[feats].values.astype(np.float32)),
        'MiRAGE+dot':   (np.hstack([train[feats].values, tr_dot.reshape(-1, 1)]).astype(np.float32),
                         np.hstack([test[feats].values, te_dot.reshape(-1, 1)]).astype(np.float32)),
        'MiRAGE+embed': (np.hstack([train[feats].values, tr_emb]).astype(np.float32),
                         np.hstack([test[feats].values, te_emb]).astype(np.float32)),
        'MiRAGE+MolEmb32': (np.hstack([train[feats].values, tr_mol]).astype(np.float32),
                            np.hstack([test[feats].values, te_mol]).astype(np.float32)),
    }
    if getattr(run_dataset, 'quick', False):
        variants = {k: v for k, v in variants.items()
                    if k in ('MiRAGE', 'MiRAGE+MolEmb32')}
    if getattr(run_dataset, 'models', None):
        variants = {k: v for k, v in variants.items() if k in run_dataset.models}
    rows = []
    for name, (Xtr, Xte) in variants.items():
        r = evaluate_holdout(Xtr, ytr, Xte, yte, make_unified_xgb, seed=seed, n_boot=n_boot)
        r['model'] = name
        r['feature_set'] = name
        r['feature_dim'] = Xtr.shape[1]
        rows.append(r)
        print(f"  {name:<14} ({Xtr.shape[1]:>3}维)  AUROC={r['AUROC']:.5f}  AUPR={r['AUPR']:.5f}  F1={r['F1']:.5f}")

    # 6. baselines (MiRAGE + GiGs 1-D dot)
    base_Xtr_df = train[feats].copy()
    base_Xte_df = test[feats].copy()
    base_Xtr_df['score_gigs'] = tr_dot
    base_Xte_df['score_gigs'] = te_dot
    for name, bcfg in BASELINE_MODELS:
        Xtr = base_Xtr_df if bcfg.get('is_predict') else base_Xtr_df.values.astype(np.float32)
        Xte = base_Xte_df if bcfg.get('is_predict') else base_Xte_df.values.astype(np.float32)
        r = evaluate_holdout(Xtr, ytr, Xte, yte,
                             bcfg.get('factory'),
                             needs_scaler=bcfg.get('needs_scaler', False),
                             is_predict=bcfg.get('is_predict', False),
                             mirage_features=feats, seed=seed, n_boot=n_boot)
        r['model'] = name
        r['feature_set'] = 'MiRAGE+dot'
        r['feature_dim'] = Xtr.shape[1]
        rows.append(r)
        print(f"  {name:<18} ({Xtr.shape[1]:>3}维)  AUROC={r['AUROC']:.5f}  AUPR={r['AUPR']:.5f}  F1={r['F1']:.5f}")

    # 7. 组装行
    out = []
    for r in rows:
        out.append({
            'dataset': ds, 'model': r['model'], 'feature_set': r['feature_set'],
            'feature_dim': r['feature_dim'],
            'n_train': len(train), 'n_test': len(test),
            'n_pos_test': int((yte == 1).sum()), 'n_neg_test': int((yte == 0).sum()),
            'protocol': r2_config.PROTOCOL, 'threshold_rule': r2_config.THRESHOLD_RULE,
            'config_id': 'standard-lr0.1', 'seed': seed, 'n_boot': n_boot,
            'feature_source': r2_config.FEATURE_SOURCE, 'grmf_source': r2_config.GRMF_SOURCE,
            'AUROC': round(r['AUROC'], 6), 'AUROC_ci_lo': round(r['AUROC_ci_lo'], 6),
            'AUROC_ci_hi': round(r['AUROC_ci_hi'], 6),
            'AUPR': round(r['AUPR'], 6), 'AUPR_ci_lo': round(r['AUPR_ci_lo'], 6),
            'AUPR_ci_hi': round(r['AUPR_ci_hi'], 6),
            'F1': round(r['F1'], 6), 'Accuracy': round(r['Accuracy'], 6),
            'Recall': round(r['Recall'], 6), 'Precision': round(r['Precision'], 6),
            'Threshold': round(r['Threshold'], 4),
        })
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['C', 'F', 'DDCD'], default=None)
    parser.add_argument('--seed', type=int, default=r2_config.SEED)
    parser.add_argument('--n-boot', type=int, default=r2_config.N_BOOT)
    parser.add_argument('--append', action='store_true', help='追加到现有账本')
    parser.add_argument('--quick', action='store_true',
                        help='只跑 MiRAGE + MiRAGE+MolEmb32 两个变体 (补种子用, 大幅缩短单任务时长)')
    parser.add_argument('--models', nargs='+', default=None,
                        help='只跑指定变体 (如 --models MiRAGE+dot MiRAGE+embed)')
    args = parser.parse_args()
    run_dataset.quick = args.quick
    run_dataset.models = args.models

    datasets = [args.dataset] if args.dataset else ['C', 'F', 'DDCD']
    all_rows = []
    if args.append and os.path.exists(r2_config.MANIFEST_OUT):
        prev = pd.read_csv(r2_config.MANIFEST_OUT)
        if args.quick:
            # quick: 只替换本次追加的模型行, 保留该 seed 的其他模型行
            qmodels = ['MiRAGE', 'MiRAGE+MolEmb32']
            prev = prev[~((prev['dataset'].isin(datasets)) &
                          (prev['seed'] == args.seed) &
                          (prev['model'].isin(qmodels)) &
                          (prev['feature_source'] == r2_config.FEATURE_SOURCE))]
        else:
            prev = prev[~(prev['dataset'].isin(datasets) &
                          (prev['seed'] == args.seed) &
                          (prev['feature_source'] == r2_config.FEATURE_SOURCE))]
        all_rows.extend(prev.to_dict('records'))

    for ds in datasets:
        all_rows.extend(run_dataset(ds, r2_config.DATASETS[ds], args.seed, args.n_boot))

    df = pd.DataFrame(all_rows)
    os.makedirs(r2_config.RESULTS_R2, exist_ok=True)
    if os.path.exists(r2_config.MANIFEST_OUT) and not args.append:
        # 保护: 非 append 覆盖前先备份旧账本
        import shutil
        shutil.copy(r2_config.MANIFEST_OUT, r2_config.MANIFEST_OUT + '.bak')
    df.to_csv(r2_config.MANIFEST_OUT, index=False)
    print(f"\n  ✅ 账本已保存: {r2_config.MANIFEST_OUT} ({len(df):,} 行)")
    print(df[['dataset', 'model', 'feature_dim', 'AUROC', 'AUPR', 'F1']].to_string(index=False))


if __name__ == '__main__':
    main()
