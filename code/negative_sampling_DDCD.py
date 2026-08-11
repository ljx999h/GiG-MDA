"""
=============================================================================
 MiRAGE Negative Sampling — 严格遵循论文三段式 Hard Negative Mining
 论文: Hassanali Aragh et al., BIB 2024, bbae337
       Pages 4-5 "Negative sampling" section
=============================================================================

 论文原文逻辑 (三段式):

   Step 1 — 数据划分:
     - 正样本: 80% 训练 / 20% 测试 (固定不变)
     - 未标注对: 分成 k 段 (segment), 每段 ≈ |正样本| 对
     - 每段内部: 80% 训练 / 20% 测试

   Step 2 — 弱分类器投票:
     - 对每个 segment i, 用 (正训练 + 段内负训练) 训练 DT/KNN
     - 在段内负测试集上预测
     - 记录每个未标注对被预测为正的次数 → num_preds ∈ [0, k]

   Step 3 — 筛选可靠负样本:
     - 保留 num_preds = 0 的对 (被所有弱分类器一致判为负)
     - 这些就是 "reliable negatives"

   特点:
     - 比例不由超参数决定, 由数据自然产生
     - num_preds = 0 意味着该对在所有 k 段中都被判定为可靠的负样本
     - k = ⌊未标注对总数 / 正样本数⌋
=============================================================================
"""

import pandas as pd
import numpy as np
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# ================================================================
# 配置
# ================================================================

INPUT_FILE = r"code\results\MiRAGE_score_DDCD.csv"

OUTPUT_TRAIN = r"data\DDCD\Evaluation\train.csv"
OUTPUT_TEST  = r"data\DDCD\Evaluation\test.csv"

COL_LABEL = 'label'
COL_DRUG_ID = 'drugID'
COL_DISEASE_ID = 'diseaseID'

TEST_SIZE = 0.2
RANDOM_SEED = 42

# 弱分类器类型: 'dt' (Decision Tree) / 'knn' / 'xgb' (XGBoost GPU)
# 论文 Section "Ensuring classifier independence": 用 DT/KNN 做负采样, RF 做最终分类
MINING_CLASSIFIER = 'xgb'  # 论文对齐: DecisionTree

# ================================================================
# 22 维 MiRAGE 特征
# ================================================================
FEATURE_COLS = [
    'count_drug', 'count_disease',
    'q_score_Description', 'q_score_Pathway', 'q_score_Slim',
    'p_score_Target', 'p_score_Category', 'p_score_Conditions',
    'p_score_Description', 'p_score_Mechanism', 'p_score_Pharmacodynamics',
    'p_score_Smile',
    'adj_q_score_Description', 'adj_q_score_Pathway', 'adj_q_score_Slim',
    'adj_p_score_Target', 'adj_p_score_Category', 'adj_p_score_Conditions',
    'adj_p_score_Description', 'adj_p_score_Mechanism', 'adj_p_score_Pharmacodynamics',
    'adj_p_score_Smile',
]

# ================================================================


def load_data(filename, feature_cols):
    """加载 MiRAGE 特征 CSV"""
    if not os.path.exists(filename):
        alt = filename.replace("score_", "scores_") if "scores" not in filename \
              else filename.replace("scores_", "score_")
        if os.path.exists(alt):
            filename = alt
        else:
            print(f"[FATAL] 找不到: {filename}")
            return None, None

    print(f"[LOAD] {filename}")
    df = pd.read_csv(filename)
    df.columns = df.columns.str.strip()
    df = df.fillna(0.0)

    available = [c for c in feature_cols if c in df.columns]
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(f"  [WARN] 缺失 {len(missing)} 个特征: {missing[:5]}...")
    print(f"  [INFO] {len(df):,} 行 | 特征 {len(available)} | "
          f"正样本 {(df[COL_LABEL]==1).sum():,}")
    return df, available


def paper_negative_sampling(df_pos, df_neg_pool, valid_feats,
                             classifier_type='dt', random_state=RANDOM_SEED):
    """
    论文三段式 Hard Negative Mining (精确实现)

    Parameters
    ----------
    df_pos : DataFrame — 正样本
    df_neg_pool : DataFrame — 全量未标注对 (负样本池)
    valid_feats : list — 特征列名
    classifier_type : str — 'dt' (Decision Tree) 或 'knn'
    random_state : int

    Returns
    -------
    df_train : DataFrame — 训练集 (正样本 + 可靠负样本)
    df_neg_reliable : DataFrame — 筛选出的可靠负样本
    num_preds_series : Series — 所有未标注对的 num_preds 值
    k : int — 段数
    """
    n_pos = len(df_pos)
    n_pool = len(df_neg_pool)

    # --------------------------------------------------------------
    # Step 1: 划分正样本 (80/20)
    # --------------------------------------------------------------
    pos_train, pos_test = train_test_split(
        df_pos, test_size=TEST_SIZE, random_state=random_state
    )
    n_pos_train = len(pos_train)
    print(f"\n  [Step 1] 正样本: {n_pos:,} (全量)")
    print(f"    训练正样本: {n_pos_train:,} (80%)")
    print(f"    测试正样本: {len(pos_test):,} (20%)")

    # --------------------------------------------------------------
    # Step 2: 将未标注对分成 k 段
    #   "we divide the unknown pairs into k segments,
    #    each containing approximately the same number of pairs
    #    as the positive samples" (论文原文)
    # --------------------------------------------------------------
    # 每段大小 ≈ 训练正样本数
    segment_size = n_pos_train
    k = max(1, n_pool // segment_size)

    # 随机打乱未标注对, 均分为 k 段 (保留原始索引, 确保 num_preds 对齐)
    df_neg_shuffled = shuffle(df_neg_pool, random_state=random_state)

    segments = []
    for i in range(k):
        start = i * segment_size
        end = min((i + 1) * segment_size, n_pool)
        if start < n_pool:
            segments.append(df_neg_shuffled.iloc[start:end])

    # 如果最后一段太小, 合并到前一段
    if len(segments) > 1 and len(segments[-1]) < segment_size // 2:
        segments[-2] = pd.concat([segments[-2], segments[-1]], axis=0)
        segments = segments[:-1]
        k = len(segments)

    print(f"\n  [Step 2] 未标注对: {n_pool:,} → {k} 段")
    print(f"    每段 ≈ {segment_size:,} 对 (对齐正样本数)")
    for i, seg in enumerate(segments[:3]):
        print(f"    Segment {i+1}: {len(seg):,} 对")
    if k > 3:
        print(f"    ... (共 {k} 段)")

    # --------------------------------------------------------------
    # Step 3: 对每段训练弱分类器 + 投票
    #   "By running the classification model, we determine the
    #    number of times each unknown pair is predicted as a
    #    positive sample. This count is recorded as num_pred."
    # --------------------------------------------------------------
    gpu_tag = " GPU" if classifier_type == 'xgb' else ""
    print(f"\n  [Step 3] 弱分类器投票 (classifier={classifier_type}{gpu_tag})...")

    # 为所有未标注对初始化 num_preds = 0
    num_preds = pd.Series(0, index=df_neg_pool.index)
    segment_results = []

    for i, seg_neg in enumerate(segments):
        # 段内 80/20 划分
        if len(seg_neg) < 5:
            continue  # 太小, 跳过
        neg_train_seg, neg_test_seg = train_test_split(
            seg_neg, test_size=TEST_SIZE, random_state=random_state + i
        )

        if len(neg_train_seg) < 2 or len(neg_test_seg) < 2:
            continue

        # 训练集: 正样本训练 + 段内负训练
        X_train = pd.concat([pos_train[valid_feats],
                             neg_train_seg[valid_feats]], axis=0)
        y_train = pd.concat([pd.Series(1, index=pos_train.index),
                             pd.Series(0, index=neg_train_seg.index)], axis=0)

        # 弱分类器 (论文原文: Decision Tree, 与原版 MiRAGE 一致)
        if classifier_type == 'knn':
            from sklearn.neighbors import KNeighborsClassifier
            clf = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
        else:
            # 论文: max_depth=4 浅层 DecisionTree (与原版 MiRAGE 一致)
            clf = DecisionTreeClassifier(
                max_depth=4, min_samples_leaf=15, random_state=random_state
            )

        clf.fit(X_train, y_train)

        # 预测段内负测试集
        y_pred = clf.predict(neg_test_seg[valid_feats])
        # 被预测为正 (1) 的, num_preds += 1
        pred_positive_idx = neg_test_seg.index[y_pred == 1]
        num_preds.loc[pred_positive_idx] += 1

        if (i + 1) % max(1, k // 5) == 0:
            n_pos_pred = (num_preds > 0).sum()
            print(f"    Segment {i+1}/{k} 完成 | "
                  f"累计 num_preds>0: {n_pos_pred:,} "
                  f"({n_pos_pred/n_pool*100:.1f}%)")

    # --------------------------------------------------------------
    # Step 4: 筛选可靠负样本 (num_preds = 0)
    #   "We utilize all unknown drug–disease pairs with a
    #    num_preds value of zero as negative sampling pairs"
    # --------------------------------------------------------------
    reliable_idx = num_preds[num_preds == 0].index
    df_neg_reliable = df_neg_pool.loc[reliable_idx].copy()
    df_neg_reliable['numPreds'] = 0  # 论文中 num_preds = 0

    n_reliable = len(df_neg_reliable)
    ratio = n_reliable / n_pos_train if n_pos_train > 0 else 0

    print(f"\n  [Step 4] 可靠负样本筛选 (num_preds = 0)")
    print(f"    可靠负样本: {n_reliable:,} ({n_reliable/n_pool*100:.1f}% of 未标注)")
    print(f"    正:负 = 1:{ratio:.1f}")
    print(f"    num_preds 分布: "
          f"0→{n_reliable:,}, "
          f"1→{(num_preds==1).sum():,}, "
          f"2→{(num_preds==2).sum():,}, "
          f"3+→{(num_preds>=3).sum():,}")

    # 训练集: 训练正样本 + 可靠负样本
    df_train = pd.concat([pos_train, df_neg_reliable], axis=0)
    df_train = shuffle(df_train, random_state=random_state)

    return df_train, pos_test, df_neg_reliable, num_preds, k


def main():
    print("=" * 70)
    print(" MiRAGE Negative Sampling (论文三段式 Hard Negative Mining)")
    print(" 论文: BIB 2024, bbae337, Pages 4-5")
    print("=" * 70)

    # ----------------------------------------------------------
    # Step 1: 加载
    # ----------------------------------------------------------
    print(f"\n[1/5] 加载 MiRAGE 特征...")
    df_raw, valid_feats = load_data(INPUT_FILE, FEATURE_COLS)
    if df_raw is None or len(valid_feats) == 0:
        return

    # ----------------------------------------------------------
    # Step 2: 分离正负
    # ----------------------------------------------------------
    print(f"\n[2/5] 分离正/未标注样本...")
    df_pos_all = df_raw[df_raw[COL_LABEL] == 1].copy()
    df_neg_all = df_raw[df_raw[COL_LABEL] == 0].copy()
    print(f"  正样本 (全量):   {len(df_pos_all):,}")
    print(f"  未标注 (全量):   {len(df_neg_all):,}")
    print(f"  自然不平衡比:    1:{len(df_neg_all)/len(df_pos_all):.1f}")

    # ----------------------------------------------------------
    # Step 3: 论文三段式负采样
    # ----------------------------------------------------------
    print(f"\n[3/5] 论文三段式 Hard Negative Mining...")
    train_df, pos_test, df_neg_reliable, num_preds, k = \
        paper_negative_sampling(df_pos_all, df_neg_all, valid_feats,
                                classifier_type=MINING_CLASSIFIER)

    # ----------------------------------------------------------
    # Step 4: 测试集 = 测试正样本 + 全量未标注对
    #   (测试集保留完整分布, 不采样 — 真实药物重定位场景)
    # ----------------------------------------------------------
    print(f"\n[4/5] 构建测试集 (完整分布)...")
    test_df = pd.concat([pos_test, df_neg_all], axis=0)
    test_df = shuffle(test_df, random_state=RANDOM_SEED)
    # 将 num_preds 信息写入测试集 (方便后续分析)
    test_df['numPreds'] = num_preds.reindex(test_df.index, fill_value=-1).values
    test_df['numPreds'] = test_df['numPreds'].clip(lower=0)

    # ----------------------------------------------------------
    # Step 5: 保存
    # ----------------------------------------------------------
    print(f"\n[5/5] 保存...")
    os.makedirs(os.path.dirname(OUTPUT_TRAIN), exist_ok=True)

    train_df.to_csv(OUTPUT_TRAIN, index=False)
    test_df.to_csv(OUTPUT_TEST, index=False)

    # ----------------------------------------------------------
    # 汇总
    # ----------------------------------------------------------
    print(f"\n{'='*70}")
    print(f" 完成! 数据汇总")
    print(f"{'='*70}")

    n_train_pos = (train_df[COL_LABEL] == 1).sum()
    n_train_neg = (train_df[COL_LABEL] == 0).sum()
    n_test_pos = (test_df[COL_LABEL] == 1).sum()
    n_test_neg = (test_df[COL_LABEL] == 0).sum()

    print(f"  Train: {len(train_df):,} 行 | "
          f"正 {n_train_pos:,} | 负 {n_train_neg:,} | "
          f"比例 1:{n_train_neg/max(1,n_train_pos):.1f}")
    print(f"  Test : {len(test_df):,} 行 | "
          f"正 {n_test_pos:,} | 负 {n_test_neg:,} | "
          f"比例 1:{n_test_neg/max(1,n_test_pos):.1f}")
    print(f"  K = {k} 段 | 弱分类器 = {MINING_CLASSIFIER}")
    print(f"  可靠负样本 (num_preds=0): {n_train_neg:,}")
    print(f"  输出: {OUTPUT_TRAIN}")
    print(f"  输出: {OUTPUT_TEST}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
