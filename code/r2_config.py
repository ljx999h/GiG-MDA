"""
R2 统一协议/配置源 (唯一真相)

所有 R2 账本相关脚本 (build_mirage_features / negative_mining_oof /
build_results_ledger / generate_R2_tables) 从这里读取数据集路径、统一
XGBoost 配置与协议常量, 消除此前各脚本配置不一致 (诊断报告 P0-6).

统一 XGBoost 配置 = 标准配置 (与 eval_sota_baselines.make_xgb 一致):
  n_estimators=500, max_depth=10, learning_rate=0.1,
  tree_method='hist', device='cuda', eval_metric='logloss',
  random_state=42; 无 scale_pos_weight / subsample / colsample.
GiG-MDA 三种特征变体与 XGBoost baseline 共用此配置.
"""
import os

# ---------------------------------------------------------------- 协议常量
TEST_SIZE = 0.2          # 固定独立测试比例 (方案B)
OOF_K = 5                # out-of-fold 负采样折数
SEED = 42                # 全局随机种子 (本阶段单种子)
N_BOOT = 1000            # bootstrap CI 次数
THRESHOLD_RULE = "train-in-CV PR F1-max, mean over folds"
PROTOCOL = "B-fixed-holdout"
FEATURE_SOURCE = "r2train"   # 特征在 R2 train 划分上重建
GRMF_SOURCE = "P_train"      # GiGs 嵌入在 P_train 上训练 (pretrain_gigs_split)

# 标准 XGBoost 配置
XGB_CONFIG = dict(
    n_estimators=500, max_depth=10, learning_rate=0.1,
    tree_method='hist', device='cuda', eval_metric='logloss',
    random_state=SEED, verbosity=0,
)

# ---------------------------------------------------------------- 特征列
# 基础特征 (r2 特征文件 / ledger 用, 无 MolFormer): C/F 18 维, DDCD 22 维
FEAT_18 = ['count_drug', 'count_disease',
           'q_score_PS',
           'p_score_Target', 'p_score_Category', 'p_score_Conditions',
           'p_score_Description', 'p_score_Mechanism', 'p_score_Pharmacodynamics',
           'p_score_Smile',
           'adj_q_score_PS',
           'adj_p_score_Target', 'adj_p_score_Category', 'adj_p_score_Conditions',
           'adj_p_score_Description', 'adj_p_score_Mechanism',
           'adj_p_score_Pharmacodynamics', 'adj_p_score_Smile']

FEAT_22 = ['count_drug', 'count_disease',
           'q_score_Description', 'q_score_Pathway', 'q_score_Slim',
           'p_score_Target', 'p_score_Category', 'p_score_Conditions',
           'p_score_Description', 'p_score_Mechanism', 'p_score_Pharmacodynamics',
           'p_score_Smile',
           'adj_q_score_Description', 'adj_q_score_Pathway', 'adj_q_score_Slim',
           'adj_p_score_Target', 'adj_p_score_Category', 'adj_p_score_Conditions',
           'adj_p_score_Description', 'adj_p_score_Mechanism',
           'adj_p_score_Pharmacodynamics', 'adj_p_score_Smile']

# MolFormer 增强特征 (build_mirage_features 的 mol/_cold 输出文件用): C/F 20 维, DDCD 24 维
_MOL_INSERT = [('p_score_MolFormer', 'p_score_Smile'), ('adj_p_score_MolFormer', 'adj_p_score_Smile')]


def _with_molformer(feats):
    out = list(feats)
    for col, after in _MOL_INSERT:
        out.insert(out.index(after) + 1, col)
    return out


FEAT_18_MOL = _with_molformer(FEAT_18)
FEAT_22_MOL = _with_molformer(FEAT_22)

# ---------------------------------------------------------------- 数据集路径
# score: R2 重建的泄漏安全特征文件 (由 build_mirage_features.py --neighbor-source r2train 生成)
# gigs:  P_train 上训练的 GRMF 嵌入 (pretrain_gigs_split.py 产物, 已存在)
DATASETS = {
    'C': {
        'manifest': 'data/C-Dataset/Splits/split_manifest.csv',
        'score': 'code/results/MiRAGE_score_C_r2.csv',
        'gigs': 'data/C-Dataset/Splits/gigs_split_C.pkl',
        'mapping': 'data/C-Dataset/Mapping/mapping_C.csv',
        'train_pos': 'data/C-Dataset/Splits/train_positives.csv',
        'train_neg': 'data/C-Dataset/Splits/train_negatives.csv',
        'feats': FEAT_18,
        'id_space': 'C-index',   # score 文件 drugID 为整数索引, 下游需映射回 DB 串
    },
    'F': {
        'manifest': 'data/F-Dataset/Splits/split_manifest.csv',
        'score': 'code/results/MiRAGE_score_F_r2.csv',
        'gigs': 'data/F-Dataset/Splits/gigs_split_F.pkl',
        'mapping': 'data/F-Dataset/Mapping/mapping_F.csv',
        'train_pos': 'data/F-Dataset/Splits/train_positives.csv',
        'train_neg': 'data/F-Dataset/Splits/train_negatives.csv',
        'feats': FEAT_18,
        'id_space': 'string',
    },
    'DDCD': {
        'manifest': 'data/DDCD/Splits/split_manifest.csv',
        'score': 'code/results/MiRAGE_score_DDCD_r2.csv',
        'gigs': 'data/DDCD/Splits/gigs_split_DDCD.pkl',
        'mapping': 'data/DDCD/Mapping/mapping.csv',
        'train_pos': 'data/DDCD/Splits/train_positives.csv',
        'train_neg': 'data/DDCD/Splits/train_negatives.csv',
        'feats': FEAT_22,
        'id_space': 'string',
    },
}

# 旧 score 文件 (mapping80 版本, 仅作移植校验对照, 不进入账本)
LEGACY_SCORE = {
    'C': 'code/results/MiRAGE_score_C.csv',
    'F': 'code/results/MiRAGE_score_F.csv',
    'DDCD': None,
}

RESULTS_R2 = 'results/R2'
MANIFEST_OUT = os.path.join(RESULTS_R2, 'results_manifest.csv')
TABLES_OUT = os.path.join(RESULTS_R2, 'tables')
