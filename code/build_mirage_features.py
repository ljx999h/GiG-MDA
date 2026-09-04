"""
R2: 可复现 MiRAGE 特征构建器 (build_mirage_features)

逐 cell 移植 MiRAGE_C / MiRAGE_F / MiRAGE _DDCD 三个 notebook 的核心打分逻辑
(论文 Eq.3-4: q/p_score = max sim over train-neighbors, adj_* = score × 交叉计数),
改为命令行脚本并做 numpy 向量化 (与 notebook 逐 cell 语义等价).

邻居来源 (--neighbor-source):
  mapping80 : 用旧 mapping80_{DS}.csv 构建邻居 → 用于校验移植正确性
              (输出应与旧 MiRAGE_score_{DS}.csv 数值完全一致)
  r2train   : 用 split_manifest 的 train 正样本构建邻居 → 泄漏安全,
              R2 方案B 唯一特征来源 (输出 MiRAGE_score_{DS}_r2.csv)
  full      : 用全量 mapping 构建邻居 (部署模型: 全部已知关联 + 目标对留一排除);
              案例研究三表 (case_study_deploy) 用

特征公式 (与 notebook 完全一致):
  count_disease = |Ad|   (drug d 的 train 疾病邻居, 排除自身)
  count_drug    = |Bs|   (disease s 的 train 药物邻居, 排除自身)
  q_score       = max_{s' in Ad} Sim(s, s')     (疾病侧)
  p_score       = max_{d' in Bs} Sim(d, d')     (药物侧)
  adj_q_score   = q_score × count_drug
  adj_p_score   = p_score × count_disease
  label         = 1 iff (d,s) 在全量 mapping 中

用法:
  python code/build_mirage_features.py --dataset C  --neighbor-source mapping80   # 校验
  python code/build_mirage_features.py --dataset C  --neighbor-source r2train     # 泄漏安全
  python code/build_mirage_features.py --dataset DDCD --neighbor-source r2train
"""
import argparse
import os
import numpy as np
import pandas as pd

# ---------------------------------------------------------------- 数据集配置
DATA_ROOT = 'data'

DISEASE_SIM_FILENAMES = {
    'C': {'PS': 'DiseasePS.csv'},
    'F': {'PS': 'DiseasePS.csv'},
    'DDCD': {'Description': 'diseaseDecription_bert.csv',
             'Pathway': 'diseasePathwayName_jaccard.csv',
             'Slim': 'diseaseSlimmapping_jaccard.csv'},
}
DRUG_SIM_FILENAMES = {
    'C': {'Target': 'target_similarity_C.csv', 'Category': 'category_simialrity_C.csv',
          'Conditions': 'condition_similarity_C.csv', 'Description': 'description_similarity_C.csv',
          'Mechanism': 'mechanism_similarity_C.csv', 'Pharmacodynamics': 'pharmacodynamics_similarity_C.csv',
          'Smile': 'SMILE_similarity_C.csv',
          'MolFormer': 'code/results/molformer/C_similarity.csv'},
    'F': {'Target': 'target_similarity_F.csv', 'Category': 'category_simialrity_F.csv',
          'Conditions': 'condition_similarity_F.csv', 'Description': 'description_similarity_F.csv',
          'Mechanism': 'mechanism_similarity_F.csv', 'Pharmacodynamics': 'pharmacodynamics_similarity_F.csv',
          'Smile': 'SMILE_similarity_F.csv',
          'MolFormer': 'code/results/molformer/F_similarity.csv'},
    'DDCD': {'Target': 'drugTarget_jaccard', 'Category': 'drugCategory_jaccard',
             'Conditions': 'drugConditions_jaccard', 'Description': 'drugDescription_bert',
             'Mechanism': 'drugMechanism_bert', 'Pharmacodynamics': 'drugPharmacodynamics_bert',
             'Smile': 'drugSmile_tanimoto',
             'MolFormer': 'code/results/molformer/DDCD_similarity.csv'},
}

DATASETS = {
    'C': {'sim_dir': f'{DATA_ROOT}/C-Dataset/SimilarityMatrices',
          'mapping_full': f'{DATA_ROOT}/C-Dataset/Mapping/mapping_C.csv',
          'mapping80': f'{DATA_ROOT}/C-Dataset/Mapping/mapping80_C.csv',
          'manifest': f'{DATA_ROOT}/C-Dataset/Splits/split_manifest.csv',
          'out': 'code/results/MiRAGE_score_C_r2.csv',
          'id_mode': 'C-index'},
    'F': {'sim_dir': f'{DATA_ROOT}/F-Dataset/SimilarityMatrices',
          'mapping_full': f'{DATA_ROOT}/F-Dataset/Mapping/mapping_F.csv',
          'mapping80': f'{DATA_ROOT}/F-Dataset/Mapping/mapping80_F.csv',
          'manifest': f'{DATA_ROOT}/F-Dataset/Splits/split_manifest.csv',
          'out': 'code/results/MiRAGE_score_F_r2.csv',
          'id_mode': 'string'},
    'DDCD': {'sim_dir': f'{DATA_ROOT}/DDCD/SimilarityMatrices',
             'mapping_full': f'{DATA_ROOT}/DDCD/Mapping/mapping.csv',
             'mapping80': f'{DATA_ROOT}/DDCD/Mapping/mapping80.csv',
             'manifest': f'{DATA_ROOT}/DDCD/Splits/split_manifest.csv',
             'out': 'code/results/MiRAGE_score_DDCD_r2.csv',
             'id_mode': 'string'},
}


def _coerce_int(x):
    try:
        return int(x)
    except Exception:
        return str(x)


def load_sim_matrices(ds, cfg):
    """返回 (疾病sim dict, 药物sim dict). C 的疾病矩阵索引按 notebook 规整为 int,
    药物矩阵索引规整为 DB 字符串 (C 的输出 drugID 为整数, 靠 int↔DB 映射查矩阵)."""
    disease_sims, drug_sims = {}, {}
    for fname, f in DISEASE_SIM_FILENAMES[ds].items():
        df = pd.read_csv(os.path.join(cfg['sim_dir'], f), index_col=0)
        if ds == 'C':
            df.index = pd.Index([_coerce_int(x) for x in df.index])
            df.columns = pd.Index([_coerce_int(x) for x in df.columns])
        disease_sims[fname] = df
    for fname, f in DRUG_SIM_FILENAMES[ds].items():
        fpath = f if ('/' in f or '\\' in f) else os.path.join(cfg['sim_dir'], f)
        df = pd.read_csv(fpath, index_col=0)
        if ds == 'C':
            df.index = pd.Index([str(x).strip() for x in df.index])
            df.columns = pd.Index([str(x).strip() for x in df.columns])
        drug_sims[fname] = df
    return disease_sims, drug_sims


def _c_int_to_db(cfg):
    """C 专用: 整数 drug 索引 ↔ 排序后的 DrugBank 字符串 (与 notebook 一致)."""
    full = pd.read_csv(cfg['mapping_full'])
    full = full.iloc[:, :2]
    full.columns = ['drugID', 'diseaseID']
    sorted_db = sorted(full['drugID'].astype(str).str.strip().unique())
    return {i: db for i, db in enumerate(sorted_db)}, {db: i for i, db in enumerate(sorted_db)}


def load_neighbor_pairs(ds, cfg, source):
    """返回邻居对 DataFrame (drugID, diseaseID), 输出 ID 空间: C=整数, F/DDCD=字符串."""
    if source == 'full':
        df = pd.read_csv(cfg['mapping_full'])
        df = _drop_unnamed_cols(df)
        df = df.iloc[:, :2]
        df.columns = ['drugID', 'diseaseID']
        if ds == 'C':
            df['drugID'] = df['drugID'].astype(int)
            df['diseaseID'] = df['diseaseID'].astype(int)
        else:
            df['drugID'] = df['drugID'].astype(str).str.strip()
            df['diseaseID'] = df['diseaseID'].astype(str).str.strip()
        return df
    if source == 'mapping80':
        df = pd.read_csv(cfg['mapping80'])
        df = df.iloc[:, :2]
        df.columns = ['drugID', 'diseaseID']
        if ds == 'C':
            df['drugID'] = df['drugID'].astype(int)
            df['diseaseID'] = df['diseaseID'].astype(int)
        else:
            df['drugID'] = df['drugID'].astype(str).str.strip()
            df['diseaseID'] = df['diseaseID'].astype(str).str.strip()
        return df
    else:  # r2train
        mf = pd.read_csv(cfg['manifest'])
        mf['drugID'] = mf['drugID'].astype(str).str.strip()
        mf['diseaseID'] = mf['diseaseID'].astype(str).str.strip()
        train = mf[(mf['split'] == 'train') & (mf['label'] == 1)][['drugID', 'diseaseID']].copy()
        if ds == 'C':
            _, db_to_int = _c_int_to_db(cfg)
            train['drugID'] = train['drugID'].map(lambda x: db_to_int.get(x, -1))
            train['diseaseID'] = train['diseaseID'].astype(int)
            train = train[train['drugID'] >= 0]
        return train


def _drop_unnamed_cols(df):
    """去掉 'Unnamed: 0' 等索引列 (mapping_F.csv 首列为行号)."""
    return df[[c for c in df.columns if not str(c).startswith('Unnamed')]]


def load_full_mapping(ds, cfg):
    """全量映射 (标签真值), 返回 (drugID, diseaseID), C=整数空间.
    加载时显式去重 (P0-5: F-Dataset mapping 曾含 1 行重复 pair) 并输出审计."""
    df = pd.read_csv(cfg['mapping_full'])
    df = _drop_unnamed_cols(df)
    df = df.iloc[:, :2]
    df.columns = ['drugID', 'diseaseID']
    n_dup = df.duplicated().sum()
    if n_dup:
        print(f'  [audit] mapping 重复行 {n_dup} (已去重)')
    df = df.drop_duplicates()
    if ds == 'C':
        _, db_to_int = _c_int_to_db(cfg)
        df['drugID'] = df['drugID'].astype(str).str.strip().map(lambda x: db_to_int.get(x, -1))
        df['diseaseID'] = df['diseaseID'].astype(int)
        df = df[df['drugID'] >= 0]
    else:
        df['drugID'] = df['drugID'].astype(str).str.strip()
        df['diseaseID'] = df['diseaseID'].astype(str).str.strip()
    return df


def entity_space(ds, cfg, source):
    """实体空间: mapping80/full=邻居对实体 (复刻 notebook); r2train=split_manifest 全候选实体."""
    if source in ('mapping80', 'full'):
        mf = pd.read_csv(cfg['mapping80'] if source == 'mapping80' else cfg['entity_full'])
        mf = _drop_unnamed_cols(mf) if source == 'full' else mf
        mf = mf.iloc[:, :2]
        mf.columns = ['drugID', 'diseaseID']
        if ds == 'C':
            mf['drugID'] = mf['drugID'].astype(int)
            mf['diseaseID'] = mf['diseaseID'].astype(int)
        else:
            mf['drugID'] = mf['drugID'].astype(str).str.strip()
            mf['diseaseID'] = mf['diseaseID'].astype(str).str.strip()
    else:
        mf = pd.read_csv(cfg['manifest'])
        mf['drugID'] = mf['drugID'].astype(str).str.strip()
        mf['diseaseID'] = mf['diseaseID'].astype(str).str.strip()
        if ds == 'C':
            _, db_to_int = _c_int_to_db(cfg)
            mf['drugID'] = mf['drugID'].map(lambda x: db_to_int.get(x, -1))
            mf['diseaseID'] = mf['diseaseID'].astype(int)
            mf = mf[mf['drugID'] >= 0]
    all_drugs = sorted(mf['drugID'].unique().tolist())
    all_diseases = sorted(mf['diseaseID'].unique().tolist())
    return all_drugs, all_diseases


def leave_one_out_max(sub, col_ids, row_ids):
    """sub: (n_targets × k); col_ids: 邻居 ID (k); row_ids: 每个 target 的自身 ID (n_targets).
    每行取排除自身列后的 max; 无合法列 → 0."""
    mask = col_ids[None, :] == row_ids[:, None]
    v = np.where(mask, -np.inf, sub)
    m = np.nanmax(v, axis=1)
    return np.nan_to_num(m, nan=0.0, neginf=0.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, choices=['C', 'F', 'DDCD'])
    parser.add_argument('--neighbor-source', required=True, choices=['mapping80', 'r2train', 'full'],
                        help='mapping80=校验移植(应与旧文件一致); r2train=R2泄漏安全特征; full=部署特征(全部已知为邻居)')
    parser.add_argument('--mapping-full', default=None,
                        help='覆盖全量映射文件 (邻居/标签来源; 如 therapeutic-only 子集). '
                             '实体空间仍取原始全量映射 (--entity-full 可另行覆盖)')
    parser.add_argument('--entity-full', default=None,
                        help='覆盖实体空间文件 (默认与原 mapping_full 相同)')
    parser.add_argument('--out', default=None, help='覆盖输出路径')
    parser.add_argument('--manifest', default=None,
                        help='覆盖 split_manifest 路径 (冷启动划分用)')
    parser.add_argument('--exclude', default=None,
                        help='排除的相似度模态名 (逗号分隔, 如 Conditions,Category; 来源泄漏消融用)')
    args = parser.parse_args()
    ds = args.dataset
    cfg = dict(DATASETS[ds])   # 复制, 不污染模块级配置
    cfg['entity_full'] = cfg['mapping_full']   # 实体空间默认 = 原始全量映射
    if args.mapping_full:
        cfg['mapping_full'] = args.mapping_full   # 邻居/标签来源覆盖
    if args.entity_full:
        cfg['entity_full'] = args.entity_full
    if args.manifest:
        cfg['manifest'] = args.manifest
    if args.exclude:
        exclude = {m.strip() for m in args.exclude.split(',')}
        for fname in list(DRUG_SIM_FILENAMES[ds]):
            if fname in exclude:
                del DRUG_SIM_FILENAMES[ds][fname]
        for fname in list(DISEASE_SIM_FILENAMES[ds]):
            if fname in exclude:
                del DISEASE_SIM_FILENAMES[ds][fname]
        print(f"  排除模态: {sorted(exclude)}")

    print(f"\n{'='*60}\n 构建 MiRAGE 特征: {ds} | 邻居来源={args.neighbor_source}\n{'='*60}")

    # 1. 相似度矩阵
    disease_sims, drug_sims = load_sim_matrices(ds, cfg)
    print(f"  疾病相似度: {list(disease_sims)} | 药物相似度: {list(drug_sims)}")

    # 2. 邻居对与全量映射
    neighbor_df = load_neighbor_pairs(ds, cfg, args.neighbor_source)
    full_mapping = load_full_mapping(ds, cfg)
    print(f"  邻居对(train): {len(neighbor_df):,} | 全量映射(label): {len(full_mapping):,}")

    # 3. 实体空间
    all_drugs, all_diseases = entity_space(ds, cfg, args.neighbor_source)
    n_dr, n_di = len(all_drugs), len(all_diseases)
    print(f"  实体: {n_dr:,} drugs × {n_di:,} diseases = {n_dr*n_di:,} 对")

    # 4. 邻居索引 + 标签对
    drug_to_diseases = neighbor_df.groupby('drugID')['diseaseID'].apply(set).to_dict()
    disease_to_drugs = neighbor_df.groupby('diseaseID')['drugID'].apply(set).to_dict()
    known_pairs = set(zip(full_mapping['drugID'], full_mapping['diseaseID']))

    drug_pos = {d: i for i, d in enumerate(all_drugs)}
    dis_pos = {d: i for i, d in enumerate(all_diseases)}
    dis_ids = np.array(all_diseases)

    # C 专用: 整数 drug → DB 字符串 (drug sim 矩阵索引)
    int_to_db = None
    if ds == 'C':
        int_to_db, _ = _c_int_to_db(cfg)
    sim_ids = [int_to_db[d] if ds == 'C' else d for d in all_drugs]
    sim_id_arr = np.array([str(x) for x in sim_ids])

    # 5. 疾病侧 (目标=疾病)
    dis_feats = list(disease_sims.keys())
    Q = {}
    for fname, mat in disease_sims.items():
        S = mat.reindex(index=all_diseases, columns=all_diseases).to_numpy(dtype=np.float64)
        M = np.zeros((n_dr, n_di), dtype=np.float64)
        for d in all_drugs:
            ad = drug_to_diseases.get(d, set())
            valid = [x for x in ad if x in dis_pos and x in mat.index]
            if not valid:
                continue
            cols = np.array([dis_pos[x] for x in valid])
            sub = S[:, cols]                     # n_di × k (行=目标疾病, 列=邻居疾病)
            M[drug_pos[d], :] = leave_one_out_max(sub, np.array(valid), dis_ids)
        Q[fname] = M
        print(f"  q_score_{fname} 完成 [{M.min():.4f}, {M.max():.4f}]")

    # 6. 药物侧 (目标=药物)
    dr_feats = list(drug_sims.keys())
    P = {}
    for fname, mat in drug_sims.items():
        S = mat.reindex(index=sim_ids, columns=sim_ids).to_numpy(dtype=np.float64)
        M = np.zeros((n_dr, n_di), dtype=np.float64)
        for s in all_diseases:
            bs = disease_to_drugs.get(s, set())
            valid = [x for x in bs if x in drug_pos and sim_ids[drug_pos[x]] in mat.index]
            if not valid:
                continue
            cols = np.array([drug_pos[x] for x in valid])
            sub = S[:, cols]                     # n_dr × k (行=目标药物, 列=邻居药物)
            M[:, dis_pos[s]] = leave_one_out_max(sub, sim_id_arr[cols], sim_id_arr)
        P[fname] = M
        print(f"  p_score_{fname} 完成 [{M.min():.4f}, {M.max():.4f}]")

    # 7. 计数矩阵 (排除自身)
    known_disease = np.zeros((n_dr, n_di), dtype=np.int32)
    for d in all_drugs:
        for s in drug_to_diseases.get(d, set()):
            if s in dis_pos:
                known_disease[drug_pos[d], dis_pos[s]] = 1
    known_drug = np.zeros((n_dr, n_di), dtype=np.int32)
    for s in all_diseases:
        for d in disease_to_drugs.get(s, set()):
            if d in drug_pos:
                known_drug[drug_pos[d], dis_pos[s]] = 1
    count_disease = known_disease.sum(axis=1, keepdims=True) - known_disease   # |Ad| - self
    count_drug = known_drug.sum(axis=0, keepdims=True) - known_drug            # |Bs| - self
    count_disease = count_disease.astype(np.float64)
    count_drug = count_drug.astype(np.float64)

    # 8. label 矩阵
    label = np.zeros((n_dr, n_di), dtype=np.int8)
    for d, s in known_pairs:
        if d in drug_pos and s in dis_pos:
            label[drug_pos[d], dis_pos[s]] = 1
    print(f"  label 正样本: {label.sum():,}")

    # 9. 组装 DataFrame (行序: drug 外层, disease 内层; 列序与 notebook 一致:
    #    ID + count + q + p + adj_q + adj_p + label)
    drug_ids = np.repeat(np.array(all_drugs), n_di)
    disease_ids = np.tile(np.array(all_diseases), n_dr)
    data = {
        'drugID': drug_ids, 'diseaseID': disease_ids,
        'count_drug': count_drug.ravel(), 'count_disease': count_disease.ravel(),
    }
    for fname in dis_feats:
        data[f'q_score_{fname}'] = Q[fname].ravel()
    for fname in dr_feats:
        data[f'p_score_{fname}'] = P[fname].ravel()
    for fname in dis_feats:
        data[f'adj_q_score_{fname}'] = (Q[fname] * count_drug).ravel()
    for fname in dr_feats:
        data[f'adj_p_score_{fname}'] = (P[fname] * count_disease).ravel()
    data['label'] = label.ravel()

    df = pd.DataFrame(data)
    feature_cols = [c for c in df.columns if c not in ['drugID', 'diseaseID', 'label']]
    n_feat_exp = 2 + 2 * len(dis_feats) + 2 * len(dr_feats)   # count + 2×疾病模态 + 2×药物模态
    assert len(feature_cols) == n_feat_exp, f'特征维度错误: {len(feature_cols)} vs {n_feat_exp}'
    n_label_expected = sum(1 for d, s in known_pairs if d in drug_pos and s in dis_pos)
    assert (df['label'] == 1).sum() == n_label_expected, \
        f'label 数量不一致: {(df["label"]==1).sum()} vs 实体空间内已知对 {n_label_expected}'
    print(f"  特征维度: {len(feature_cols)} | 行数: {len(df):,} | 正: {(df['label']==1).sum():,} | 负: {(df['label']==0).sum():,}")

    # 10. 保存
    out = args.out or cfg['out']
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    print(f"  ✅ 已保存: {out}")


if __name__ == '__main__':
    main()
