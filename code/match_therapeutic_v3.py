# -*- coding: utf-8 -*-
"""
therapeutic 匹配 v3 (final):
  1) 精确规范化名匹配 (v1)
  2) 盐型后缀剥离匹配 (CTD 侧/DB 侧, 含 disoproxil 前药标记)
  3) 人工同义词映射 (INN/USAN/成分基名/复数等)
输出: data/DDCD/Mapping/mapping_therapeutic.csv (覆盖 v1) + audit
"""
import gzip
import io

import pandas as pd

CTD_GZ = 'data/ctd_raw/CTD_chemicals_diseases.csv.gz'
DRUG_INFO = 'data/DDCD/Features/drugsInfo.csv'
DISEASE_INFO = 'data/DDCD/Features/diseasesInfo.csv'
OUT = 'data/DDCD/Mapping/mapping_therapeutic.csv'
AUDIT = 'data/ctd_raw/therapeutic_audit.csv'

SALT_TOKENS = {'hydrochloride', 'mesylate', 'besylate', 'tosylate', 'embonate',
               'maleate', 'tartrate', 'citrate', 'succinate', 'fumarate',
               'sulfate', 'sulphate', 'phosphate', 'disodium', 'monosodium',
               'sesquihydrate', 'monohydrate', 'dihydrate', 'disoproxil'}
DROP_TOKENS = {'disoproxil'}  # 前药标记 (Tenofovir disoproxil -> Tenofovir)

# 同义词: DB DrugName (规范化) -> 可接受的 CTD 规范化名
SYNONYMS = {
    'levocarnitine': ['carnitine', 'lcarnitine'],
    'norethisterone': ['norethindrone', 'norethindroneacetate'],
    'ergocalciferol': ['ergocalciferols', 'vitamind2'],
    'metamfetamine': ['methamphetamine', 'dextromethamphetamine', 'levomethamphetamine'],
    'vitaminc': ['ascorbicacid'],
    'mycophenolatemofetil': ['mycophenolicacid'],
    'beclomethasonedipropionate': ['beclomethasone'],
    'liothyronine': ['triiodothyronine'],
    'salbutamol': ['albuterol', 'salbutamolsulfate'],
    'mesalazine': ['mesalamine'],
    'clomifene': ['clomiphene'],
    'meticillin': ['methicillin'],
    'cefalotin': ['cephalothin'],
    'isoprenaline': ['isoproterenol'],
    'orciprenaline': ['metaproterenol'],
    'benzylpenicillin': ['penicilling', 'penicillin'],
    'ubidecarenone': ['coenzymeq10', 'ubiquinone', 'coenzymeq10' 'coenzymeq'],
    'lipoicacid': ['thiocticacid', 'alphalipoicacid', 'dihydrolipoicacid'],
    'pamidronicacid': ['pamidronate', 'pamidronatedisodium'],
    'thimerosal': ['thiomersal'],
    'nah': ['nad'],
    'metamizole': ['dipyrone', 'metamizolesodium'],
    'oxybuprocaine': ['benoxinate'],
    'phenacetin': ['phenacetine'],
    'atp': ['adenosinetriphosphate'],
    'alphatocopherol': ['tocopherol'],
    'ubidecarenone': ['ubiquinone'],
}


def norm(s, strip=False, drop=set()):
    toks = ''.join(ch for ch in str(s).lower() if ch.isalnum() or ch == ' ').split()
    if strip:
        toks = [t for t in toks if t not in SALT_TOKENS and t not in drop]
    return ''.join(toks)


def iter_ctd_therapeutic():
    dis = set(pd.read_csv(DISEASE_INFO, dtype=str)['DiseaseID'])
    with io.TextIOWrapper(gzip.open(CTD_GZ, 'rb'), encoding='utf-8',
                          errors='replace') as f:
        header = None
        for line in f:
            if line.startswith('#'):
                if line.startswith('# ChemicalName,'):
                    header = line[2:].rstrip('\n').split(',')
                continue
            if header is None:
                continue
            parts = line.rstrip('\n').split(',')
            if len(parts) < len(header):
                parts += [''] * (len(header) - len(parts))
            rec = dict(zip(header, parts))
            if 'therapeutic' in rec.get('DirectEvidence', ''):
                yield rec


def main():
    drugs = pd.read_csv(DRUG_INFO, usecols=['DrugID', 'DrugName'], dtype=str)
    drugs['DrugName'] = drugs['DrugName'].astype(str).str.strip()
    dis = set(pd.read_csv(DISEASE_INFO, dtype=str)['DiseaseID'])

    # 每药可接受的规范化名集合
    accept = {}
    for _, r in drugs.iterrows():
        e = norm(r['DrugName'])
        s = norm(r['DrugName'], strip=True)
        acc = {e, s} | {x for x in SYNONYMS.get(e, [])}
        accept[r['DrugID']] = acc

    # 收集 therapeutic 行 (疾病空间内), 按 (norm_ctd_salt) 索引
    rows = []
    for rec in iter_ctd_therapeutic():
        did = rec['DiseaseID'].strip()
        if did not in dis:
            continue
        rows.append((rec['ChemicalName'].strip(), rec['ChemicalID'].strip(), did,
                     rec['DiseaseName'].strip(), rec.get('DirectEvidence', '')))
    print(f'疾病空间内 therapeutic 行: {len(rows):,}')

    # 反向索引: CTD 规范化名(含盐剥离) -> 行
    from collections import defaultdict
    by_norm = defaultdict(list)
    by_salt = defaultdict(list)
    for r in rows:
        by_norm[norm(r[0])].append(r)
        by_salt[norm(r[0], strip=True)].append(r)

    matched_drugs = set()
    pair_set = set()
    audit = []
    for _, r in drugs.iterrows():
        db = r['DrugID']
        acc = accept[db]
        e = norm(r['DrugName'])
        cand_rows = []
        seen = set()
        for a in acc:
            for lst in (by_norm.get(a, []), by_salt.get(a, [])):
                for x in lst:
                    key = (x[1], x[2])  # (ChemicalID, DiseaseID)
                    if key not in seen:
                        seen.add(key)
                        cand_rows.append(x)
        # 多映射审计 (1 DB -> 多 CTD 化学物)
        if cand_rows:
            matched_drugs.add(db)
        for x in cand_rows:
            pair_set.add((db, x[2]))
            audit.append({'DB_DrugID': db, 'DB_DrugName': r['DrugName'],
                          'CTD_ChemicalName': x[0], 'CTD_ChemicalID': x[1],
                          'DiseaseID': x[2], 'DiseaseName': x[3]})

    mp = pd.DataFrame(sorted(pair_set), columns=['DrugID', 'DiseaseID'])
    mp.to_csv(OUT, index=False)
    aud = pd.DataFrame(audit).drop_duplicates()
    aud.to_csv(AUDIT, index=False)
    print(f'映射输出: {OUT} | 行 {len(mp):,} | 药物 {mp.DrugID.nunique():,}/1410 | '
          f'疾病 {mp.DiseaseID.nunique():,}/1573 | 匹配药物数 {len(matched_drugs)}')

    # 供审阅: 药物有 CTD 命中但对到 0 个 DB (说明该药无 in-space therapeutic 行)
    drugs_matched_ctd = set(by_norm.keys()) | set(by_salt.keys())
    unmatched = drugs[~drugs['DrugID'].isin(matched_drugs)]
    print(f'\n仍未匹配药物: {len(unmatched)}')
    print(unmatched['DrugName'].to_string(index=False))

    # 关键度数复核
    for label, col, ent in [('Methotrexate', 'DrugID', 'DB00563'),
                            ('Dexamethasone', 'DrugID', 'DB01234'),
                            ('Aspirin', 'DrugID', 'DB00945'),
                            ('Resveratrol', 'DrugID', 'DB02709'),
                            ('Stomach Neoplasms', 'DiseaseID', 'MESH:D013274'),
                            ('Colonic Neoplasms', 'DiseaseID', 'MESH:D003110')]:
        n = (mp[col] == ent).sum()
        print(f'therapeutic 度数 {label}: {n}')
    deg = mp.groupby('DrugID').size().sort_values(ascending=False)
    dnames = dict(zip(drugs.DrugID, drugs.DrugName))
    print('\ntherapeutic 度数 Top-12 药物:')
    for d, n in deg.head(12).items():
        print(f'  {d} {dnames.get(d, "?"):28s} {n}')


if __name__ == '__main__':
    main()
