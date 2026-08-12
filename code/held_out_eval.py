"""
P0-C 方案B: held-out test 评估 (无偏的阈值相关指标)
对 DDCD / C / F 三个数据集:
  1. 在 train 上跑 5-Fold CV, 得到每折 validation 最优阈值 -> 取平均
  2. 用全量 train 训练 XGBoost (128D GiGs 特征)
  3. 在 held-out test 上用平均阈值评估:
       AUROC / AUPR   (阈值无关, 直接算)
       F1 / Accuracy / Recall / Precision (用平均阈值, 无偏)
输出: results/held_out_summary.csv
"""
import os
import pickle
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    f1_score, recall_score, precision_score, precision_recall_curve
)
import xgboost as xgb

from eval_helpers_4variants import get_gigs_features

RANDOM_SEED = 42
N_FOLDS = 5
XGB_N = 500
XGB_D = 10
LR = 0.05

# ================================================================
# 数据集配置
# ================================================================
FEAT_22 = [
    'count_drug', 'count_disease',
    'q_score_Description', 'q_score_Pathway', 'q_score_Slim',
    'p_score_Target', 'p_score_Category', 'p_score_Conditions',
    'p_score_Description', 'p_score_Mechanism', 'p_score_Pharmacodynamics', 'p_score_Smile',
    'adj_q_score_Description', 'adj_q_score_Pathway', 'adj_q_score_Slim',
    'adj_p_score_Target', 'adj_p_score_Category', 'adj_p_score_Conditions',
    'adj_p_score_Description', 'adj_p_score_Mechanism', 'adj_p_score_Pharmacodynamics', 'adj_p_score_Smile',
]
FEAT_18 = [
    'count_drug', 'count_disease',
    'q_score_PS',
    'p_score_Target', 'p_score_Category', 'p_score_Conditions',
    'p_score_Description', 'p_score_Mechanism', 'p_score_Pharmacodynamics', 'p_score_Smile',
    'adj_q_score_PS',
    'adj_p_score_Target', 'adj_p_score_Category', 'adj_p_score_Conditions',
    'adj_p_score_Description', 'adj_p_score_Mechanism', 'adj_p_score_Pharmacodynamics', 'adj_p_score_Smile',
]

DATASETS = {
    'DDCD': {
        'train': 'data/DDCD/Evaluation/train.csv',
        'test':  'data/DDCD/Evaluation/test.csv',
        'gigs':  'code/model/gigs_dataDDCD.pkl',
        'features': FEAT_22,
    },
    'C-Dataset': {
        'train': 'data/C-Dataset/Evaluation/train.csv',
        'test':  'data/C-Dataset/Evaluation/test.csv',
        'gigs':  'code/model/gigs_dataC.pkl',
        'features': FEAT_18,
    },
    'F-Dataset': {
        'train': 'data/F-Dataset/Evaluation/train.csv',
        'test':  'data/F-Dataset/Evaluation/test.csv',
        'gigs':  'code/model/gigs_dataF.pkl',
        'features': FEAT_18,
    },
}


def make_clf(spw):
    return xgb.XGBClassifier(
        n_estimators=XGB_N, max_depth=XGB_D, learning_rate=LR,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw,
        tree_method='hist', device='cuda',
        random_state=RANDOM_SEED, eval_metric='logloss', verbosity=0
    )


def main():
    print("=" * 70)
    print(" P0-C 方案B: held-out test 评估 (无偏阈值相关指标)")
    print("=" * 70)

    summary = []
    for name, cfg in DATASETS.items():
        t0 = time.time()
        print(f"\n{'='*60}\n {name}\n{'='*60}")

        # 1. 加载
        train = pd.read_csv(cfg['train']).fillna(0.0)
        test = pd.read_csv(cfg['test']).fillna(0.0)
        with open(cfg['gigs'], 'rb') as f:
            gigs = pickle.load(f)
        feats = [c for c in cfg['features'] if c in train.columns]
        print(f"  train: {len(train):,} | test: {len(test):,} | 特征: {len(feats)}")

        # 2. 注入 128D GiGs
        emb_tr, _ = get_gigs_features(train, gigs, 'embed')
        emb_te, _ = get_gigs_features(test, gigs, 'embed')
        Xtr = np.hstack([train[feats].values, emb_tr]).astype(np.float32)
        ytr = train['label'].values
        Xte = np.hstack([test[feats].values, emb_te]).astype(np.float32)
        yte = test['label'].values
        spw = (ytr == 0).sum() / max(1, (ytr == 1).sum())

        # 3. 5-Fold CV 取平均阈值
        print("  5-Fold CV (取平均阈值)...")
        skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        threshs = []
        for tr, va in skf.split(Xtr, ytr):
            clf = make_clf(spw)
            clf.fit(Xtr[tr], ytr[tr])
            yv = clf.predict_proba(Xtr[va])[:, 1]
            p, r, t = precision_recall_curve(ytr[va], yv)
            f1s = 2 * p * r / (p + r + 1e-9)
            best = f1s.argmax()
            threshs.append(t[best] if best < len(t) else 0.5)
        avg_thresh = float(np.mean(threshs))
        print(f"  平均阈值: {avg_thresh:.4f}")

        # 4. 全量训练
        print("  全量训练 XGBoost...")
        clf = make_clf(spw)
        clf.fit(Xtr, ytr)

        # 5. held-out test 评估
        yprob = clf.predict_proba(Xte)[:, 1]
        auroc = roc_auc_score(yte, yprob)
        aupr = average_precision_score(yte, yprob)
        ypred = (yprob >= avg_thresh).astype(int)
        f1 = f1_score(yte, ypred)
        acc = accuracy_score(yte, ypred)
        rec = recall_score(yte, ypred)
        prec = precision_score(yte, ypred, zero_division=0)

        row = {
            'Dataset': name,
            'AUROC': round(auroc, 5),
            'AUPR': round(aupr, 5),
            'F1': round(f1, 5),
            'Accuracy': round(acc, 5),
            'Recall': round(rec, 5),
            'Precision': round(prec, 5),
            'Threshold': round(avg_thresh, 4),
            'Time(s)': round(time.time() - t0),
        }
        summary.append(row)
        print(f"  >>> AUROC={auroc:.5f} AUPR={aupr:.5f} F1={f1:.5f} Acc={acc:.5f} "
              f"Recall={rec:.5f} Precision={prec:.5f}")

    # 汇总
    df = pd.DataFrame(summary)
    print(f"\n{'='*70}\n 汇总 (held-out test, 无偏阈值相关指标)\n{'='*70}")
    print(df.to_string(index=False))
    os.makedirs('results', exist_ok=True)
    df.to_csv('results/held_out_summary.csv', index=False)
    print(f"\n已保存: results/held_out_summary.csv")


if __name__ == "__main__":
    main()
