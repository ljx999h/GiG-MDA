"""
C-Dataset 三段式 Hard Negative Mining (论文严格实现)
 输入: results/MiRAGE_score_C.csv (18 维特征)
 输出: data/C-Dataset/Evaluation/train.csv + test.csv
"""
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# ================================================================
# 配置
# ================================================================
INPUT_FILE = r"code\results\MiRAGE_score_C.csv"
OUTPUT_TRAIN = r"data/C-Dataset/Evaluation/train.csv"
OUTPUT_TEST  = r"data/C-Dataset/Evaluation/test.csv"

COL_LABEL = 'label'
TEST_SIZE = 0.2
RANDOM_SEED = 42
MINING_CLASSIFIER = 'dt'  # 'xgb' GPU | 'dt' CPU

# ================================================================
# 18 维特征
# ================================================================
FEATURE_COLS = [
    'count_drug', 'count_disease',
    'q_score_PS',
    'p_score_Target', 'p_score_Category', 'p_score_Conditions',
    'p_score_Description', 'p_score_Mechanism', 'p_score_Pharmacodynamics', 'p_score_Smile',
    'adj_q_score_PS',
    'adj_p_score_Target', 'adj_p_score_Category', 'adj_p_score_Conditions',
    'adj_p_score_Description', 'adj_p_score_Mechanism', 'adj_p_score_Pharmacodynamics', 'adj_p_score_Smile',
]


def paper_negative_sampling(df_pos, df_neg_pool, valid_feats, classifier_type='xgb', random_state=42):
    n_pos = len(df_pos)
    n_pool = len(df_neg_pool)

    pos_train, pos_test = train_test_split(df_pos, test_size=TEST_SIZE, random_state=random_state)
    n_pos_train = len(pos_train)
    print(f"\n  [Step 1] 正样本 {n_pos:,} → 训练 {n_pos_train:,} / 测试 {len(pos_test):,}")

    # 段数 k = 未标注总数 / 正样本数
    segment_size = n_pos_train
    k = max(1, n_pool // segment_size)
    df_neg_shuffled = shuffle(df_neg_pool, random_state=random_state)

    segments = []
    for i in range(k):
        s, e = i * segment_size, min((i+1) * segment_size, n_pool)
        if s < n_pool:
            segments.append(df_neg_shuffled.iloc[s:e])
    if len(segments) > 1 and len(segments[-1]) < segment_size // 2:
        segments[-2] = pd.concat([segments[-2], segments[-1]], axis=0)
        segments = segments[:-1]
        k = len(segments)

    print(f"  [Step 2] 未标注 {n_pool:,} → {k} 段 (每段 ~{segment_size:,})")

    # 弱分类器投票
    print(f"  [Step 3] 弱分类器投票 (classifier={classifier_type})...")
    num_preds = pd.Series(0, index=df_neg_pool.index)

    for i, seg in enumerate(segments):
        if len(seg) < 5: continue
        neg_train_seg, neg_test_seg = train_test_split(seg, test_size=TEST_SIZE, random_state=random_state + i)
        if len(neg_train_seg) < 2 or len(neg_test_seg) < 2: continue

        X_train = pd.concat([pos_train[valid_feats], neg_train_seg[valid_feats]], axis=0)
        y_train = pd.concat([pd.Series(1, index=pos_train.index),
                             pd.Series(0, index=neg_train_seg.index)], axis=0)

        if classifier_type == 'knn':
            from sklearn.neighbors import KNeighborsClassifier
            clf = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
        else:
            # 论文: max_depth=4 浅层 DecisionTree
            from sklearn.tree import DecisionTreeClassifier
            clf = DecisionTreeClassifier(max_depth=4, min_samples_leaf=15, random_state=random_state)

        clf.fit(X_train, y_train)
        y_pred = clf.predict(neg_test_seg[valid_feats])
        pred_pos_idx = neg_test_seg.index[y_pred == 1]
        num_preds.loc[pred_pos_idx] += 1

        if (i+1) % max(1, k//5) == 0:
            print(f"    Segment {i+1}/{k} 完成 | num_preds>0: {(num_preds>0).sum():,}")

    # 可靠负样本
    reliable_idx = num_preds[num_preds == 0].index
    df_neg_reliable = df_neg_pool.loc[reliable_idx].copy()
    ratio = len(df_neg_reliable) / max(1, n_pos_train)
    print(f"\n  [Step 4] 可靠负样本 (num_preds=0): {len(df_neg_reliable):,} | 比例 1:{ratio:.1f}")

    df_train = pd.concat([pos_train, df_neg_reliable], axis=0)
    df_train = shuffle(df_train, random_state=random_state)
    return df_train, pos_test, num_preds, k


def main():
    print("=" * 60)
    print(" C-Dataset 三段式 Hard Negative Mining")
    print("=" * 60)

    if not os.path.exists(INPUT_FILE):
        print(f"[FATAL] 找不到 {INPUT_FILE}, 请先运行 MiRAGE_C.ipynb")
        return

    df = pd.read_csv(INPUT_FILE).fillna(0.0)
    available = [c for c in FEATURE_COLS if c in df.columns]
    print(f"\n[1/5] 加载: {len(df):,} 行, {len(available)} 特征")

    df_pos = df[df[COL_LABEL] == 1].copy()
    df_neg = df[df[COL_LABEL] == 0].copy()
    print(f"[2/5] 正 {len(df_pos):,} | 负 {len(df_neg):,} (自然 1:{len(df_neg)/len(df_pos):.1f})")

    print(f"[3/5] 三段式采样...")
    train_df, pos_test, num_preds, k = paper_negative_sampling(
        df_pos, df_neg, available, classifier_type=MINING_CLASSIFIER
    )

    test_df = pd.concat([pos_test, df_neg], axis=0)
    test_df = shuffle(test_df, random_state=RANDOM_SEED)
    test_df['numPreds'] = num_preds.reindex(test_df.index, fill_value=-1).clip(lower=0).values

    os.makedirs(os.path.dirname(OUTPUT_TRAIN), exist_ok=True)
    train_df.to_csv(OUTPUT_TRAIN, index=False)
    test_df.to_csv(OUTPUT_TEST, index=False)

    print(f"\n[5/5] 完成!")
    print(f"  Train: {len(train_df):,} (正 {(train_df[COL_LABEL]==1).sum():,}, 负 {(train_df[COL_LABEL]==0).sum():,})")
    print(f"  Test : {len(test_df):,} (正 {(test_df[COL_LABEL]==1).sum():,}, 负 {(test_df[COL_LABEL]==0).sum():,})")
    print(f"  K={k} | 弱分类器={MINING_CLASSIFIER}")


if __name__ == "__main__":
    main()
