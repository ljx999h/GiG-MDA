"""
生成论文 Figure: ROC + PR curves (3 datasets × 多方法对比)
 基于 5-Fold CV 的聚合预测, 阈值无关的 ROC/PR 曲线
"""
import os, pickle, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from xgboost import XGBClassifier

RANDOM_SEED = 42
N_FOLDS = 5
RESULTS_DIR = "results/figures"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ================================================================
DATASETS = {
    'DDCD':     {'train': 'data/DDCD/Evaluation/train.csv',      'gigs': 'code/model/gigs_dataDDCD.pkl',  'name': 'DDCD',        'n_mirage': 22},
    'C-Dataset':{'train': 'data/C-Dataset/Evaluation/train.csv',  'gigs': 'code/model/gigs_dataC.pkl',     'name': 'C-Dataset',   'n_mirage': 18},
    'F-Dataset':{'train': 'data/F-Dataset/Evaluation/train.csv',  'gigs': 'code/model/gigs_dataF.pkl',     'name': 'F-Dataset',   'n_mirage': 18},
}

# ================================================================
FEAT_MIRAGE = ['count_drug','count_disease',
               'q_score_Description','q_score_Pathway','q_score_Slim',
               'p_score_Target','p_score_Category','p_score_Conditions',
               'p_score_Description','p_score_Mechanism','p_score_Pharmacodynamics','p_score_Smile',
               'adj_q_score_Description','adj_q_score_Pathway','adj_q_score_Slim',
               'adj_p_score_Target','adj_p_score_Category','adj_p_score_Conditions',
               'adj_p_score_Description','adj_p_score_Mechanism','adj_p_score_Pharmacodynamics','adj_p_score_Smile']

FEAT_GIGS_128 = [f'gigs_drug_emb_{i}' for i in range(64)] + \
                [f'gigs_disease_emb_{i}' for i in range(64)]


def inject_gigs(df, gigs_data):
    if gigs_data is None:
        for i in range(64):
            df[f'gigs_drug_emb_{i}'] = 0.0
            df[f'gigs_disease_emb_{i}'] = 0.0
        return df
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
    return pd.concat([df, pd.DataFrame(emb, columns=FEAT_GIGS_128, index=df.index)], axis=1)


def get_methods(ds_cfg, gigs_data, df_full):
    """返回 (X_array, y_array) 对列表, 每个方法对应一组特征"""
    available = [c for c in FEAT_MIRAGE + FEAT_GIGS_128 if c in df_full.columns]
    # 确保 GiGs 列的缺失处理
    _ = df_full  # 已注入
    X_1d = df_full[[c for c in FEAT_MIRAGE if c in df_full.columns] +
                   ([c for c in ['score_gigs'] if c in df_full.columns] if 'score_gigs' in df_full.columns else [])]
    # 计算 1D dot
    if gigs_data is not None and 'score_gigs' not in X_1d.columns:
        Xm, Ym = gigs_data['X'], gigs_data['Y']
        d2i, s2i = gigs_data['drug_to_idx'], gigs_data['disease_to_idx']
        dc = 'drugID' if 'drugID' in df_full.columns else 'DrugID'
        sc = 'diseaseID' if 'diseaseID' in df_full.columns else 'DiseaseID'
        di = np.array([d2i.get(str(d).strip(), -1) for d in df_full[dc]], dtype=np.int32)
        si = np.array([s2i.get(str(s).strip(), -1) for s in df_full[sc]], dtype=np.int32)
        v = (di >= 0) & (si >= 0)
        dot = np.zeros(len(df_full), dtype=np.float32)
        if v.any():
            dot[v] = np.sum(Xm[di[v]] * Ym[si[v]], axis=1)
        df_full = df_full.copy()
        df_full['score_gigs'] = dot
        X_1d = df_full[[c for c in FEAT_MIRAGE if c in df_full.columns] + ['score_gigs']]
    X_128 = df_full[[c for c in available if c in df_full.columns]]

    return {
        '1D Dot + RF':       (X_1d, RandomForestClassifier(n_estimators=100, max_depth=12, n_jobs=-1, random_state=RANDOM_SEED)),
        '128D Embed + RF':   (X_128, RandomForestClassifier(n_estimators=100, max_depth=12, n_jobs=-1, random_state=RANDOM_SEED)),
        '1D Dot + XGBoost':  (X_1d, XGBClassifier(n_estimators=500, max_depth=10, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, tree_method='hist', device='cuda', random_state=RANDOM_SEED, verbosity=0)),
        '128D Embed + XGBoost (Ours)': (X_128, XGBClassifier(n_estimators=500, max_depth=10, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, tree_method='hist', device='cuda', random_state=RANDOM_SEED, verbosity=0)),
        'LR (1D)':           (X_1d, LogisticRegression(max_iter=1000, class_weight='balanced', random_state=RANDOM_SEED)),
    }


def cv_predictions(X, y, clf_factory, n_folds=N_FOLDS):
    """5-Fold CV 聚合所有预测"""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    y_true_all, y_prob_all = [], []
    for tr, va in skf.split(X, y):
        clf = clf_factory() if callable(clf_factory) else clf_factory
        clf.fit(X.iloc[tr], y.iloc[tr])
        y_prob_all.append(clf.predict_proba(X.iloc[va])[:, 1])
        y_true_all.append(y.iloc[va].values)
    return np.concatenate(y_true_all), np.concatenate(y_prob_all)


def main():
    n_datasets = len(DATASETS)
    fig, axes = plt.subplots(n_datasets, 2, figsize=(14, 5.5 * n_datasets))
    if n_datasets == 1:
        axes = axes.reshape(1, 2)

    for row, (ds_key, ds_cfg) in enumerate(DATASETS.items()):
        print(f"\n{'='*60}")
        print(f" {ds_key}")
        print(f"{'='*60}")

        # 加载
        df = pd.read_csv(ds_cfg['train']).fillna(0.0)
        gigs = None
        if os.path.exists(ds_cfg['gigs']):
            with open(ds_cfg['gigs'], 'rb') as f:
                gigs = pickle.load(f)
        df = inject_gigs(df, gigs)
        methods = get_methods(ds_cfg, gigs, df)
        y_full = df['label']

        # Subsample for speed (DDCD too big)
        if len(df) > 300000:
            pos_idx = np.where(y_full == 1)[0]
            neg_idx = np.random.RandomState(RANDOM_SEED).choice(
                np.where(y_full == 0)[0], 300000 - len(pos_idx), replace=False)
            keep = np.concatenate([pos_idx, neg_idx])
            y_full = y_full.iloc[keep].reset_index(drop=True)

        ax_roc, ax_pr = axes[row, 0], axes[row, 1]
        colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))

        for (mname, (X_m, clf_factory)), color in zip(methods.items(), colors):
            t0 = time.time()
            X_use = X_m.iloc[keep].reset_index(drop=True) if 'keep' in dir() else X_m
            print(f"  {mname}...", end=' ', flush=True)
            yt, yp = cv_predictions(X_use, y_full, clf_factory)
            fpr, tpr, _ = roc_curve(yt, yp)
            roc_auc_val = auc(fpr, tpr)
            prec, rec, _ = precision_recall_curve(yt, yp)
            pr_auc_val = auc(rec, prec)

            ax_roc.plot(fpr, tpr, color=color, lw=1.5,
                        label=f'{mname} (AUC={roc_auc_val:.3f})')
            ax_pr.plot(rec, prec, color=color, lw=1.5,
                       label=f'{mname} (AUC={pr_auc_val:.3f})')
            print(f'ROC={roc_auc_val:.3f} PR={pr_auc_val:.3f} ({time.time()-t0:.0f}s)')

        # ROC 子图
        ax_roc.plot([0,1],[0,1], 'k--', lw=0.8, alpha=0.5)
        ax_roc.set_xlabel('False Positive Rate', fontsize=10)
        ax_roc.set_ylabel('True Positive Rate', fontsize=10)
        ax_roc.set_title(f'{ds_cfg["name"]} — ROC Curves', fontsize=12, fontweight='bold')
        ax_roc.legend(fontsize=7, loc='lower right')
        ax_roc.set_xlim(0.0, 1.0)
        ax_roc.set_ylim(0.0, 1.02)

        # PR 子图
        ax_pr.set_xlabel('Recall', fontsize=10)
        ax_pr.set_ylabel('Precision', fontsize=10)
        ax_pr.set_title(f'{ds_cfg["name"]} — PR Curves', fontsize=12, fontweight='bold')
        ax_pr.legend(fontsize=7, loc='lower left')
        ax_pr.set_xlim(0.0, 1.0)
        ax_pr.set_ylim(0.0, 1.02)

    plt.suptitle('ROC and PR Curves of MiRAGE Variants on Benchmark Datasets (5-Fold CV)',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    save_path = os.path.join(RESULTS_DIR, 'roc_pr_curves.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✅ Figure saved to {save_path}")


if __name__ == "__main__":
    main()
