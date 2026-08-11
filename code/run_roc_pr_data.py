"""
跑 ROC/PR 曲线数据 + 直接生成论文用图 (matplotlib)
 输出:
   results/figures_data/{dataset}_{method}.csv   — 每折 y_true + y_prob
   results/figures_data/all_roc_pr_summary.csv   — 汇总 AUROC/AUPR
   results/figures_data/{dataset}_roc_pr.png     — ROC+PR 双面板图
"""
import os, pickle, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')                       # 无 GUI, 直接写文件
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, precision_recall_curve, auc, roc_auc_score, average_precision_score
from xgboost import XGBClassifier
try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

RANDOM_SEED = 42
N_FOLDS = 5
RESULTS_DIR = "results/figures_data"
os.makedirs(RESULTS_DIR, exist_ok=True)

DATASETS = {
    'F-Dataset': {'train': 'data/F-Dataset/Evaluation/train.csv',  'gigs': 'code/model/gigs_dataF.pkl'},
    'DDCD':      {'train': 'data/DDCD/Evaluation/train.csv',       'gigs': 'code/model/gigs_dataDDCD.pkl'},
}

FEAT_MIRAGE = ['count_drug','count_disease',
               'q_score_Description','q_score_Pathway','q_score_Slim',
               'p_score_Target','p_score_Category','p_score_Conditions',
               'p_score_Description','p_score_Mechanism','p_score_Pharmacodynamics','p_score_Smile',
               'adj_q_score_Description','adj_q_score_Pathway','adj_q_score_Slim',
               'adj_p_score_Target','adj_p_score_Category','adj_p_score_Conditions',
               'adj_p_score_Description','adj_p_score_Mechanism','adj_p_score_Pharmacodynamics','adj_p_score_Smile']

# 颜色方案 (Nature 系, 论文常用)
COLORS = {
    'LR (1D)':          '#7f7f7f',   # 灰
    'KNN-k5 (1D)':      '#bcbd22',   # 黄绿
    'DT (1D)':          '#d62728',   # 红
    'RF (1D)':          '#ff7f0e',   # 橙
    'LightGBM (1D)':    '#2ca02c',   # 绿
    'XGBoost (1D)':     '#9467bd',   # 紫
    'MiRAGE: 1D+RF':    '#1f77b4',   # 蓝
    'MiRAGE: 128D+RF':  '#17becf',   # 青
    'MiRAGE: 1D+XGB':   '#e377c2',   # 粉
    'MiRAGE: 128D+XGB (Ours)': '#e74c3c',  # 暗红(加粗)
}


def inject_gigs(df, gigs_data):
    if gigs_data is None: return df
    Xm, Ym = gigs_data['X'], gigs_data['Y']
    d2i, s2i = gigs_data['drug_to_idx'], gigs_data['disease_to_idx']
    dc = 'drugID' if 'drugID' in df.columns else 'DrugID'
    sc = 'diseaseID' if 'diseaseID' in df.columns else 'DiseaseID'
    di = df[dc].astype(str).str.strip().map(d2i)
    si = df[sc].astype(str).str.strip().map(s2i)
    v = di.notna() & si.notna()
    emb = np.zeros((len(df), 128), dtype=np.float32)
    if v.any():
        emb[v.values, :64] = Xm[di[v].astype(int).values]
        emb[v.values, 64:] = Ym[si[v].astype(int).values]
    cols = [f'gigs_drug_emb_{i}' for i in range(64)] + [f'gigs_disease_emb_{i}' for i in range(64)]
    return pd.concat([df, pd.DataFrame(emb, columns=cols, index=df.index)], axis=1)


def cv_predictions(X, y, clf_factory):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    yt_all, yp_all, fold_ids = [], [], []
    for fold, (tr, va) in enumerate(skf.split(X, y), 1):
        clf = clf_factory() if callable(clf_factory) else clf_factory
        clf.fit(X.iloc[tr], y.iloc[tr])
        yp = clf.predict_proba(X.iloc[va])[:, 1]
        yt = y.iloc[va].values
        yp_all.append(yp)
        yt_all.append(yt)
        fold_ids.extend([fold] * len(yt))
    return np.concatenate(yt_all), np.concatenate(yp_all), fold_ids


def plot_dataset_wide(ds_name, ds_results):
    """一张横向宽图: 左=ROC 右=PR (适配 LaTeX \\textwidth)"""
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(14, 5.8))

    for mname, (yt, yp, auroc_val) in ds_results.items():
        color = COLORS.get(mname, 'black')
        lw = 2.8 if 'Ours' in mname else 1.3
        alpha = 1.0 if 'Ours' in mname else 0.7
        zorder = 10 if 'Ours' in mname else 1

        # ROC
        fpr, tpr, _ = roc_curve(yt, yp)
        ax_roc.plot(fpr, tpr, color=color, lw=lw, alpha=alpha, zorder=zorder,
                    label=f'{mname} (AUC={auroc_val:.3f})')
        # PR
        prec, rec, _ = precision_recall_curve(yt, yp)
        aupr_val = auc(rec, prec)
        ax_pr.plot(rec, prec, color=color, lw=lw, alpha=alpha, zorder=zorder,
                   label=f'{mname} (AUC={aupr_val:.3f})')

    # ROC 对角线
    ax_roc.plot([0, 1], [0, 1], 'k--', lw=0.8, alpha=0.4)
    ax_roc.set_xlabel('False Positive Rate', fontsize=11)
    ax_roc.set_ylabel('True Positive Rate', fontsize=11)
    ax_roc.set_title(f'{ds_name} — ROC Curves', fontsize=12, fontweight='bold')
    ax_roc.legend(fontsize=6.5, loc='lower right', framealpha=0.9, ncol=2)
    ax_roc.set_xlim(0.0, 1.0)
    ax_roc.set_ylim(0.0, 1.02)

    ax_pr.set_xlabel('Recall', fontsize=11)
    ax_pr.set_ylabel('Precision', fontsize=11)
    ax_pr.set_title(f'{ds_name} — PR Curves', fontsize=12, fontweight='bold')
    ax_pr.legend(fontsize=6.5, loc='lower left', framealpha=0.9, ncol=2)
    ax_pr.set_xlim(0.0, 1.0)
    ax_pr.set_ylim(0.0, 1.02)

    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, f'{ds_name}_roc_pr.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  📊 {save_path}")


def main():
    methods = [
        ('LR (1D)',          '1d',  'lr'),
        ('KNN-k5 (1D)',      '1d',  'knn'),
        ('DT (1D)',          '1d',  'dt'),
        ('RF (1D)',          '1d',  'rf'),
        ('LightGBM (1D)',    '1d',  'lgb'),
        ('XGBoost (1D)',     '1d',  'xgb'),
        ('MiRAGE: 1D+RF',    '1d',  'rf'),
        ('MiRAGE: 128D+RF',  '128d','rf'),
        ('MiRAGE: 1D+XGB',   '1d',  'xgb'),
        ('MiRAGE: 128D+XGB (Ours)', '128d','xgb'),
    ]

    summary_rows = []
    all_ds_results = {}  # {ds_key: {method: (yt, yp, auroc)}}

    for ds_key, ds_cfg in DATASETS.items():
        print(f"\n{'='*60}")
        print(f" {ds_key}")
        print(f"{'='*60}")

        df = pd.read_csv(ds_cfg['train']).fillna(0.0)
        gigs = None
        if os.path.exists(ds_cfg['gigs']):
            with open(ds_cfg['gigs'], 'rb') as f:
                gigs = pickle.load(f)
        df = inject_gigs(df, gigs)

        y = df['label']
        print(f"  {len(df):,} rows | pos {(y==1).sum():,}")

        # 构建特征矩阵
        mirage_cols = [c for c in FEAT_MIRAGE if c in df.columns]
        Xm, Ym = gigs['X'], gigs['Y']
        d2i, s2i = gigs['drug_to_idx'], gigs['disease_to_idx']
        dc = 'drugID' if 'drugID' in df.columns else 'DrugID'
        sc = 'diseaseID' if 'diseaseID' in df.columns else 'DiseaseID'
        di = np.array([d2i.get(str(d).strip(), -1) for d in df[dc]], dtype=np.int32)
        si = np.array([s2i.get(str(s).strip(), -1) for s in df[sc]], dtype=np.int32)
        v = (di >= 0) & (si >= 0)
        dot = np.zeros(len(df), dtype=np.float32)
        if v.any():
            dot[v] = np.sum(Xm[di[v]] * Ym[si[v]], axis=1)
        X_1d = df[mirage_cols].copy()
        X_1d['score_gigs'] = dot
        emb_cols = [f'gigs_drug_emb_{i}' for i in range(64)] + [f'gigs_disease_emb_{i}' for i in range(64)]
        X_128d = df[mirage_cols + [c for c in emb_cols if c in df.columns]]

        ds_results = {}  # {method_name: (y_true, y_prob, auroc)}

        for method_name, feat_type, clf_type in methods:
            t0 = time.time()
            X_use = X_1d if feat_type == '1d' else X_128d

            if clf_type == 'lr':
                clf_factory = lambda: LogisticRegression(max_iter=1000, class_weight='balanced', random_state=RANDOM_SEED)
            elif clf_type == 'knn':
                clf_factory = lambda: KNeighborsClassifier(n_neighbors=5, n_jobs=1, algorithm='auto', weights='distance')
            elif clf_type == 'dt':
                clf_factory = lambda: DecisionTreeClassifier(class_weight='balanced', random_state=RANDOM_SEED)
            elif clf_type == 'lgb' and LGB_AVAILABLE:
                clf_factory = lambda: lgb.LGBMClassifier(n_estimators=500, max_depth=10, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_SEED, verbose=-1)
            elif clf_type == 'rf':
                clf_factory = lambda: RandomForestClassifier(n_estimators=100, max_depth=12, n_jobs=-1, random_state=RANDOM_SEED)
            else:
                clf_factory = lambda: XGBClassifier(n_estimators=500, max_depth=10, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, tree_method='hist', device='cuda', random_state=RANDOM_SEED, verbosity=0)

            print(f"  {method_name} ({X_use.shape[1]} dim)...", end=' ', flush=True)
            yt, yp, fids = cv_predictions(X_use, y, clf_factory)
            auroc = roc_auc_score(yt, yp)
            aupr  = average_precision_score(yt, yp)
            elapsed = time.time() - t0
            ds_results[method_name] = (yt, yp, auroc)

            summary_rows.append({
                'Dataset': ds_key, 'Method': method_name, 'Dim': X_use.shape[1],
                'AUROC': round(auroc, 5), 'AUPR': round(aupr, 5), 'Time(s)': round(elapsed, 0),
            })
            print(f'AUROC={auroc:.4f} AUPR={aupr:.4f} ({elapsed:.0f}s)')

        # 缓存结果供最终画图
        all_ds_results[ds_key] = ds_results

    # 汇总 CSV
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(RESULTS_DIR, 'all_roc_pr_summary.csv'), index=False)

    # 每个数据集画一张独立宽图 (适合 LaTeX \textwidth)
    for ds_key, ds_results in all_ds_results.items():
        plot_dataset_wide(ds_key, ds_results)

    print(f"\n{'='*70}")
    print(f" 全部完成!")
    print(f"  汇总CSV + 2x2 ROC/PR图 → {RESULTS_DIR}/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
