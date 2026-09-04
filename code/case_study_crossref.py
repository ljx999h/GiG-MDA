# -*- coding: utf-8 -*-
"""
Crossref 证据检索 (案例研究路线1):
对 results/case_study_cold_top30.csv / case_study_cold_random30.csv 中的每个
(drug, disease) 对查询 Crossref works API (bibliographic query), 保存 top-5
命中的标题/期刊/年份/DOI, 供人工按明确标准判定 evidence (direct/indirect/none).
输出: results/case_study_evidence_top30.json / random30.json
"""
import json
import time
import urllib.parse
import urllib.request

import pandas as pd

API = ('https://api.crossref.org/works?rows=5&select=title,container-title,DOI,issued'
       '&query.bibliographic={q}&mailto=wenli@hnu.edu.cn')


def query(q):
    url = API.format(q=urllib.parse.quote(q))
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                d = json.load(r)
            items = d['message']['items']
            return [{'title': it.get('title', [''])[0],
                     'journal': (it.get('container-title') or [''])[0],
                     'year': (it.get('issued', {}).get('date-parts', [[None]])[0][0]),
                     'doi': it.get('DOI', '')} for it in items]
        except Exception as e:  # noqa: BLE001
            print(f'    retry {attempt + 1}: {e}')
            time.sleep(2)
    return []


for tag in ['top30', 'random30']:
    df = pd.read_csv(f'results/case_study_cold_{tag}.csv')
    out = []
    for i, r in df.iterrows():
        q = f"{r['DrugName']} {r['DiseaseName']}"
        print(f"[{tag}] {i + 1}/{len(df)}: {q}")
        hits = query(q)
        out.append({'drug': r['DrugName'], 'disease': r['DiseaseName'],
                    'label': int(r['label']), 'score': float(r['score']),
                    'hits': hits})
        time.sleep(0.6)
    with open(f'results/case_study_evidence_{tag}.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f'[{tag}] done -> results/case_study_evidence_{tag}.json')
