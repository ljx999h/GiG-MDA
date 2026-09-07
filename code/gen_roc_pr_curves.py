# -*- coding: utf-8 -*-
"""
ROC / PR 双子图 (导师建议样式)
场景 A: DDCD cold-drug seed 42  (base / MolEmb32 / embed / both)
场景 B: C-Dataset regular pair-disjoint seed 42  (LR / XGBoost / MiRAGE /
        MolEmb32 / embed / both)
每模型: 训练一次(统一配置) -> test 预测分数 -> ROC/PR 曲线
图例: mean AUC/AP 与 95% bootstrap CI (test 行重采样 1000 次)
输出: results/R3/fig_roc_pr_cold_ddcd.png, results/R3/fig_roc_pr_regular_c.png
数据装配与已发布数字同源 (同一 score/negatives/manifest); 不改任何指标.
"""
import io
import os
import pickle
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_curve, precision_recall_curve, auc,
                             roc_auc_score, average_precision_score)
from sklearn.decomposition import PCA

sys.path.insert(0, 'code')
import r2_config
from cold_eval import load, load_mol_emb

OUT = 'results/R3'
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10.5,
                     'axes.grid': True, 'grid.color': '#d9d9d4',
                     'grid.linewidth': 0.6, 'axes.axisbelow': True,
                     'figure.facecolor': 'white'})
COLORS = ['#c0392b', '#2878b5', '#27ae60', '#e67e22', '#8e44ad',
          '#7f8c8d', '#f39c12', '#1abc9c']


def bootstrap_ci(y, score, metric, n=1000, seed=42):
    rng = np.random.RandomState(seed)
    v = metric(y, score)
    idx = np.arange(len(y))
    vals = []
    for _ in range(n):
        b = rng.choice(idx, size=len(idx), replace=True)
        vals.append(metric(y[b], score[b]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return v, lo, hi


def gigs_vectors(df, gigs):
    d = df['drugID'].astype(str).str.strip().map(gigs['drug_to_idx'])
    s = df['diseaseID'].astype(str).str.strip().map(gigs['disease_to_idx'])
    v = d.notna() & s.notna()
    k = gigs['X'].shape[1]
    dot = np.zeros(len(df), dtype=np.float32)
    emb = np.zeros((len(df), 2 * k), dtype=np.float32)
    if v.any():
        dv = d[v].astype(int).values
        sv = s[v].astype(int).values
        dot[v.values] = np.sum(gigs['X'][dv] * gigs['Y'][sv], axis=1)
        emb[v.values, :k] = gigs['X'][dv]
        emb[v.values, k:] = gigs['Y'][sv]
    return dot, emb


def fit_predict(Xtr, ytr, Xte, kind='xgb'):
    if kind == 'lr':
        # 与 build_results_ledger/evaluate_calibration 的 LR 管线一致 (scaler + liblinear)
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler().fit(Xtr)
        Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
        clf = LogisticRegression(max_iter=1000, C=1.0, solver='liblinear', random_state=42)
        clf.fit(Xtr, ytr)
    else:
        clf = xgb.XGBClassifier(**r2_config.XGB_CONFIG)
        clf.fit(Xtr, ytr)
    return clf.predict_proba(Xte)[:, 1]


def prepare_cold(ds, seed):
    train, test, gigs, feats, base = load(ds, seed, 'cold-drug')
    ytr, yte = train['label'].values, test['label'].values
    Xb_tr = train[base].values.astype(np.float32)
    Xb_te = test[base].values.astype(np.float32)
    emb_map = load_mol_emb(ds)
    uniq_d = list(pd.unique(train['drugID'].astype(str).str.strip()))
    pca = PCA(n_components=32, random_state=42)
    pca.fit(np.array([emb_map.get(d, np.zeros(768)) for d in uniq_d], dtype=np.float32))
    z = np.zeros(768, dtype=np.float32)
    tr_e = pca.transform(np.array([emb_map.get(d, z) for d in train['drugID']])).astype(np.float32)
    te_e = pca.transform(np.array([emb_map.get(d, z) for d in test['drugID']])).astype(np.float32)
    tr_dot, tr_emb = gigs_vectors(train, gigs)
    te_dot, te_emb = gigs_vectors(test, gigs)
    data = {
        'MiRAGE': (Xb_tr, Xb_te),
        'MolEmb32': (np.hstack([Xb_tr, tr_e]), np.hstack([Xb_te, te_e])),
        'embed': (np.hstack([Xb_tr, tr_emb]), np.hstack([Xb_te, te_emb])),
        'both': (np.hstack([Xb_tr, tr_emb, tr_e]),
                 np.hstack([Xb_te, te_emb, te_e])),
    }
    return data, ytr, yte, test


def _int_to_db(ds, D):
    """C 专用: score 整数 drug 索引 -> DB 字符串 (与 build_mirage_features 一致)."""
    full = pd.read_csv(f'data/{D}/Mapping/mapping_{ds}.csv')
    full = full[[c for c in full.columns if not str(c).startswith('Unnamed')]]
    full = full.iloc[:, :2]
    full.columns = ['drugID', 'diseaseID']
    return {i: db for i, db in
            enumerate(sorted(full['drugID'].astype(str).str.strip().unique()))}


def prepare_regular(ds, seed):
    D = {'C': 'C-Dataset', 'F': 'F-Dataset', 'DDCD': 'DDCD'}[ds]
    score = pd.read_csv(r2_config.DATASETS[ds]['score']).fillna(0.0)
    if ds == 'C':
        score['drugID'] = score['drugID'].map(_int_to_db(ds, D))
    score['drugID'] = score['drugID'].astype(str).str.strip()
    score['diseaseID'] = score['diseaseID'].astype(str).str.strip()
    if 'label' in score.columns:
        score = score.drop(columns=['label'])
    feats = r2_config.DATASETS[ds]['feats']
    mf = pd.read_csv(r2_config.DATASETS[ds]['manifest'])
    for c in ['drugID', 'diseaseID']:
        mf[c] = mf[c].astype(str).str.strip()
    test_mf = mf[mf['split'] == 'test'][['drugID', 'diseaseID', 'label']]
    test = test_mf.merge(score, on=['drugID', 'diseaseID'], how='inner')
    assert len(test) == len(test_mf)
    pos = pd.read_csv(f'data/{D}/Splits/train_positives.csv').fillna(0.0)
    neg = pd.read_csv(f'data/{D}/Splits/train_negatives.csv').fillna(0.0)
    for df_ in (pos, neg):
        for c in ['drugID', 'diseaseID']:
            df_[c] = df_[c].astype(str).str.strip()
    train = pd.concat([pos, neg], ignore_index=True)
    ytr = train['label'].values
    yte = test['label'].values
    Xb_tr = train[feats].to_numpy(dtype=np.float32)
    Xb_te = test[feats].to_numpy(dtype=np.float32)
    gigs_p = f'data/{D}/Splits/gigs_split_{ds}.pkl'
    gigs = pickle.load(open(gigs_p, 'rb')) if os.path.exists(gigs_p) else None
    emb_map = load_mol_emb(ds)
    uniq_d = list(pd.unique(train['drugID'].astype(str).str.strip()))
    pca = PCA(n_components=32, random_state=42)
    pca.fit(np.array([emb_map.get(d, np.zeros(768)) for d in uniq_d], dtype=np.float32))
    z = np.zeros(768, dtype=np.float32)
    tr_e = pca.transform(np.array([emb_map.get(d, z) for d in train['drugID']])).astype(np.float32)
    te_e = pca.transform(np.array([emb_map.get(d, z) for d in test['drugID']])).astype(np.float32)
    tr_dot = te_dot = tr_emb = te_emb = None
    if gigs is not None:
        tr_dot, tr_emb = gigs_vectors(train, gigs)
        te_dot, te_emb = gigs_vectors(test, gigs)
    data = {'MiRAGE': (Xb_tr, Xb_te),
            'MolEmb32': (np.hstack([Xb_tr, tr_e]), np.hstack([Xb_te, te_e]))}
    if gigs is not None:
        data['dot'] = (np.hstack([Xb_tr, tr_dot[:, None]]),
                       np.hstack([Xb_te, te_dot[:, None]]))
        data['embed'] = (np.hstack([Xb_tr, tr_emb]), np.hstack([Xb_te, te_emb]))
        data['both'] = (np.hstack([Xb_tr, tr_emb, tr_e]),
                        np.hstack([Xb_te, te_emb, te_e]))
    # LR on MiRAGE+dot
    if gigs is not None:
        data['LR'] = (np.hstack([Xb_tr, tr_dot[:, None]]),
                      np.hstack([Xb_te, te_dot[:, None]]))
    return data, ytr, yte, test


def render(fname, title, scores, yte, order, auroc_fn, aupr_fn):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.6))
    prev = yte.mean()
    for i, name in enumerate(order):
        yp = scores[name]
        a, lo_a, hi_a = bootstrap_ci(yte, yp, auroc_fn)
        p, lo_p, hi_p = bootstrap_ci(yte, yp, aupr_fn)
        fpr, tpr, _ = roc_curve(yte, yp)
        prec, rec, _ = precision_recall_curve(yte, yp)
        col = COLORS[i % len(COLORS)]
        axes[0].plot(fpr, tpr, color=col, lw=1.1)
        axes[1].plot(rec, prec, color=col, lw=1.1)
        axes[0].plot([], [], color=col, lw=1.1,
                     label=f'{name} (AUC = {a:.3f} [{lo_a:.3f}-{hi_a:.3f}]; '
                           f'AP = {p:.3f} [{lo_p:.3f}-{hi_p:.3f}])')
    axes[0].plot([0, 1], [0, 1], color='0.6', ls='--', lw=0.9,
                 label='Random baseline (AUC = 0.5)')
    axes[1].axhline(prev, color='k', ls=':', lw=0.9,
                    label=f'Prevalence ({prev:.3f})')
    axes[0].set_xlabel('False Positive Rate'); axes[0].set_ylabel('True Positive Rate')
    axes[1].set_xlabel('Recall'); axes[1].set_ylabel('Precision')
    axes[0].set_title('(A) ROC curves')
    axes[1].set_title('(B) Precision-Recall curves')
    ytop = max(0.1, float(np.nanmax(prec)) * 1.1)
    axes[1].set_ylim(0, ytop)
    h0, l0 = axes[0].get_legend_handles_labels()
    h1, l1 = axes[1].get_legend_handles_labels()
    fig.legend(h0 + h1, l0 + l1, loc='lower center', bbox_to_anchor=(0.5, 0.012),
               ncol=2, fontsize=10.5, frameon=False, columnspacing=1.4,
               handlelength=1.4)
    fig.suptitle(title, fontsize=13, y=0.995)
    fig.tight_layout(rect=[0, 0.225, 1, 0.93])
    fig.savefig(fname, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print('saved', fname)


def main():
    # 场景 A (图已存在则跳过)
    if not os.path.exists(f'{OUT}/fig_roc_pr_cold_ddcd.png'):
        ds, seed = 'DDCD', 42
        data, ytr, yte, test = prepare_cold(ds, seed)
        scores = {}
        for name, (Xtr, Xte) in data.items():
            scores[name] = fit_predict(Xtr, ytr, Xte)
            print(f'[cold] {name} done', flush=True)
        render(f'{OUT}/fig_roc_pr_cold_ddcd.png',
               'Cold-drug ranking on DDCD (cold split, seed 42): model performance',
               scores, yte, list(data), roc_auc_score, average_precision_score)
    # 场景 B
    ds = 'C'
    data, ytr, yte, test = prepare_regular(ds, 42)
    scores = {}
    order = []
    for name, (Xtr, Xte) in data.items():
        kind = 'lr' if name == 'LR' else 'xgb'
        scores[name] = fit_predict(Xtr, ytr, Xte, kind)
        order.append(name)
        print(f'[regular] {name} done', flush=True)
    render(f'{OUT}/fig_roc_pr_regular_c.png',
           'Regular pair-disjoint ranking on C-Dataset (seed 42): model performance',
           scores, yte, order, roc_auc_score, average_precision_score)


if __name__ == '__main__':
    main()
