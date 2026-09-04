# -*- coding: utf-8 -*-
"""
案例研究 (导师三表方案) 证据检索: Crossref works API

对 results/case_study_deploy_top15_{global,mtx,gastric}.csv 与
case_study_deploy_random15_{global,mtx,gastric}.csv 中的每个 (drug, disease) 对
查询 Crossref (bibliographic query), 保存 top-5 命中的标题/期刊/年份/DOI,
供人工按明确标准判定 evidence (direct/indirect/none).
输出: results/case_study_deploy_evidence_{tag}.json (top 与 random 同 tag 分开)
"""
import json
import time
import urllib.parse
import urllib.request

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


for tag in ['global', 'drug', 'gastric']:
    for kind in ['top15', 'random15']:
        df = __import__('pandas').read_csv(f'results/case_study_deploy_{kind}_{tag}.csv')
        out = []
        for i, r in df.iterrows():
            q = f"{r['DrugName']} {r['DiseaseName']}"
            print(f"[{tag}/{kind}] {i + 1}/{len(df)}: {q}")
            hits = query(q)
            out.append({'drug': r['DrugName'], 'disease': r['DiseaseName'],
                        'score': float(r['score']), 'hits': hits})
            time.sleep(0.6)
        fname = f'results/case_study_deploy_evidence_{kind}_{tag}.json'
        with open(fname, 'w', encoding='utf-8') as f:
            json.dump(out, f, ensure_ascii=False, indent=1)
        print(f'[{tag}/{kind}] done -> {fname}')
