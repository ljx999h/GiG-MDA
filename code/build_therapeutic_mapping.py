# -*- coding: utf-8 -*-
"""
方案A (窄版): 从官方 CTD 构建 DDCD 实体空间内的 therapeutic-only 关联集
=======================================================================
评审 P0-2: DDCD 现有 mapping 不含关系类型, positive 混有治疗/毒性/不良反应.
方案A = 案例研究与药物重定向语义使用 CTD DirectEvidence 含 'therapeutic' 的
curated 行 (疾病侧 MeSH 直接匹配 DDCD; 药物侧 CTD 化学名 ↔ DrugBank 名规范化匹配).

输出:
  data/DDCD/Mapping/mapping_therapeutic.csv     (DrugID,DiseaseID) 治疗关联集
  data/ctd_raw/therapeutic_audit.csv            匹配明细 (审计用)

注意: 该集合是"官方 CTD therapeutic curated ∩ DDCD 实体空间", 与原 DDCD
42,200 对并非同一口径; 论文中将如实注明.
"""
import io
import os

import pandas as pd

CTD_GZ = 'data/ctd_raw/CTD_chemicals_diseases.csv.gz'
DRUG_INFO = 'data/DDCD/Features/drugsInfo.csv'
DISEASE_INFO = 'data/DDCD/Features/diseasesInfo.csv'
ORIG_MAPPING = 'data/DDCD/Mapping/mapping.csv'
OUT = 'data/DDCD/Mapping/mapping_therapeutic.csv'
AUDIT = 'data/ctd_raw/therapeutic_audit.csv'


def norm_name(s):
    """规范化: 小写 + 去非字母数字."""
    return ''.join(ch for ch in str(s).lower() if ch.isalnum())


def load_ctd_therapeutic():
    rows = []
    with io.TextIOWrapper(__import__('gzip').open(CTD_GZ, 'rb'),
                          encoding='utf-8', errors='replace') as f:
        header = None
        for line in f:
            if line.startswith('#'):
                if line.startswith('# ChemicalName,'):
                    header = line[2:].rstrip('\n').split(',')
                continue
            parts = line.rstrip('\n').split(',')
            if header is None:
                continue
            rec = dict(zip(header, parts))
            if 'therapeutic' in rec.get('DirectEvidence', ''):
                rows.append(rec)
    df = pd.DataFrame(rows)
    print(f'CTD therapeutic curated 行数: {len(df):,}')
    print('DirectEvidence 取值示例:', sorted(df['DirectEvidence'].unique())[:10])
    return df


def main():
    df = load_ctd_therapeutic()

    drugs = pd.read_csv(DRUG_INFO, usecols=['DrugID', 'DrugName'], dtype=str)
    dis = pd.read_csv(DISEASE_INFO, usecols=['DiseaseID', 'DiseaseName'], dtype=str)
    drugs['DrugName'] = drugs['DrugName'].astype(str).str.strip()
    dis['DiseaseName'] = dis['DiseaseName'].astype(str).str.strip()
    db_by_norm = {}
    for _, r in drugs.iterrows():
        db_by_norm.setdefault(norm_name(r['DrugName']), []).append(r['DrugID'])
    db_name_norm = {r['DrugID']: norm_name(r['DrugName']) for _, r in drugs.iterrows()}

    df['diseaseID'] = df['DiseaseID'].astype(str).str.strip()
    df['chem_norm'] = df['ChemicalName'].astype(str).str.strip().map(norm_name)
    in_disease_space = df['diseaseID'].isin(set(dis['DiseaseID']))
    df = df[in_disease_space].copy()

    # 药物匹配: 规范化名精确匹配 (含冲突消解: 一个规范化名映射多 DB ID → 保留全部, 审计)
    matched = df['chem_norm'].isin(db_by_norm)
    print(f'疾病空间内行: {len(df):,} | 药物名命中: {matched.sum():,} '
          f'({matched.mean() * 100:.1f}%)')

    audit_rows = []
    pair_set = set()
    multi = []
    for _, r in df[matched].iterrows():
        ids = db_by_norm[r['chem_norm']]
        if len(ids) > 1 and r['chem_norm'] not in multi:
            multi.append(r['chem_norm'])
        for did in ids:
            pair_set.add((did, r['diseaseID']))
            audit_rows.append({'DB_DrugID': did, 'CTD_ChemicalName': r['ChemicalName'],
                               'CTD_ChemicalID': r['ChemicalID'],
                               'DiseaseID': r['diseaseID'],
                               'DiseaseName': r['DiseaseName']})
    print(f'多义规范化名 (1→多 DB): {len(multi)} 个, 例如: {multi[:8]}')

    aud = pd.DataFrame(audit_rows).drop_duplicates()
    aud.to_csv(AUDIT, index=False)
    print(f'审计明细: {AUDIT} ({len(aud):,} 行)')

    mp = pd.DataFrame(sorted(pair_set), columns=['DrugID', 'DiseaseID'])
    mp.to_csv(OUT, index=False)
    print(f'therapeutic 关联集: {OUT} | 行数 {len(mp):,} | '
          f'药物 {mp.DrugID.nunique():,} / 1410 | 疾病 {mp.DiseaseID.nunique():,} / 1573')

    orig = pd.read_csv(ORIG_MAPPING, dtype=str)
    orig.columns = [c.strip() for c in orig.columns]
    inter = len(set(map(tuple, orig.values)) & pair_set)
    print(f'与原 DDCD mapping 交集: {inter:,} / 42,200 (原 mapping 中占比 {inter / 42200:.1%})')

    # 未匹配的 DDCD 药物
    matched_db = set(pd.DataFrame(audit_rows)['DB_DrugID'])
    unmatched = drugs[~drugs['DrugID'].isin(matched_db)]
    print(f'\n未匹配到任何 CTD therapeutic 行的 DDCD 药物: {len(unmatched)}')
    print(unmatched['DrugName'].head(60).to_string(index=False))

    # 关键实体的 therapeutic 度数 (用于案例研究措辞)
    for label, col, ent in [('Methotrexate', 'DrugID', 'DB00563'),
                            ('Cocaine', 'DrugID', 'DB00907'),
                            ('Stomach Neoplasms', 'DiseaseID', 'MESH:D013274'),
                            ('Colonic Neoplasms', 'DiseaseID', 'MESH:D003110')]:
        n = (mp[col] == ent).sum()
        print(f'therapeutic 度数 {label}: {n}')
    deg = mp.groupby('DrugID').size().sort_values(ascending=False)
    dnames = dict(zip(drugs.DrugID, drugs.DrugName))
    print('\ntherapeutic 度数 Top-10 药物:')
    for d, n in deg.head(10).items():
        print(f'  {d} {dnames.get(d, "?")}: {n}')


if __name__ == '__main__':
    main()
