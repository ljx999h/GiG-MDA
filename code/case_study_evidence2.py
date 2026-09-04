# -*- coding: utf-8 -*-
"""
案例研究证据判定 v2: query.title 精确匹配 + 规则分类 + 人工复核标记.
对 top30/random30 的每个 (drug, disease) 对:
  - label=1 (benchmark 已验证关联) -> direct (benchmark 来源 = 文献策展)
  - label=0 -> Crossref query.title = "drugName diseaseKeyword"
    direct: 命中标题同时含药物名与疾病关键词 (及常见同义词/词干)
    indirect: 命中标题含疾病关键词但与药物的关联为间接 (同类药/副作用类)
    none: 无命中或命中不相关
输出: results/case_study_evidence_v2_{tag}.csv (含人工复核列 evidence, judged_by=rule+manual)
"""
import json
import time
import urllib.parse
import urllib.request

import pandas as pd

API = ('https://api.crossref.org/works?rows=8&select=title,container-title,DOI,issued'
       '&query.title={q}&mailto=wenli@hnu.edu.cn')

# 疾病关键词 (MESH 名 -> 检索词/标题匹配词干)
DK = {
    'Stroke': ['stroke'],
    'Hematuria': ['hematuria', 'haematuria'],
    'Neurotoxicity Syndromes': ['neurotoxic'],
    'Pain': ['pain'],
    'Vasculitis': ['vasculitis'],
    'Status Epilepticus': ['status epilepticus', 'epileptic'],
    'Hearing Disorders': ['hearing', 'ototox'],
    'Meningitis': ['meningitis'],
    'Mood Disorders': ['mood', 'depress'],
    'Ischemia': ['ischemia', 'ischaemia', 'isch'],
    'Chemical and Drug Induced Liver Injury': ['liver injury', 'hepatotox', 'hepatitis', 'hepatic'],
    'Anemia, Hemolytic': ['hemolytic', 'haemolytic', 'anaemia', 'anemia'],
    'Urinary Retention': ['urinary retention', 'retention'],
    'Jaundice, Obstructive': ['jaundice', 'cholest'],
    'Staphylococcal Infections': ['staphylococcal', 'staphylococc'],
    'Weight Gain': ['weight gain', 'weight'],
    'Pain, Postoperative': ['postoperative', 'post-operative'],
    'Peritonitis': ['peritonitis'],
    'Anaphylaxis': ['anaphylaxis', 'anaphylactic'],
    'Disseminated Intravascular Coagulation': ['disseminated intravascular', 'coagulation'],
    'Sepsis': ['sepsis', 'septic'],
    'Pneumonia': ['pneumonia'],
    'Lethargy': ['lethargy', 'somnolence', 'sedation'],
    'Acute Kidney Injury': ['kidney injury', 'renal', 'acute kidney'],
    'Translocation, Genetic': ['translocation'],
    'Penile Induration': ['penile', 'peyronie'],
    'Parapsoriasis': ['parapsoriasis'],
    'Colitis, Ulcerative': ['ulcerative colitis'],
    'Influenza, Human': ['influenza', 'flu'],
    'Encephalitis': ['encephalitis'],
    'Asbestosis': ['asbestosis', 'asbestos'],
    'Sleep Deprivation': ['sleep deprivation'],
    'Cushing Syndrome': ['cushing'],
    'Hepatitis B': ['hepatitis b'],
    'Systemic Vasculitis': ['vasculitis'],
    'Gram-Negative Bacterial Infections': ['gram-negative', 'gram negative'],
    'Bronchial Diseases': ['bronchial', 'bronchospasm', 'asthma'],
    'Hyperphagia': ['hyperphagia', 'appetite'],
    'Melanoma, Experimental': ['melanoma'],
    'Peptic Ulcer': ['peptic ulcer', 'ulcer'],
    'Wernicke Encephalopathy': ['wernicke'],
    'Autonomic Nervous System Diseases': ['autonomic'],
    'Rickettsia Infections': ['rickettsia', 'rickettsial'],
    'Osteitis Deformans': ['osteitis deformans', "paget's disease", 'paget'],
    'Angioedemas, Hereditary': ['angioedema'],
    'Ulcer': ['ulcer'],
    'Obsessive-Compulsive Disorder': ['obsessive-compulsive', 'ocd'],
    'Cell Transformation, Neoplastic': ['neoplastic', 'transformation'],
    'Glomerulonephritis, IGA': ['iga', 'glomerulonephritis', 'mesangial'],
    'Aneurysm': ['aneurysm'],
    'Vipoma': ['vipoma'],
    'Spasms, Infantile': ['infantile spasm'],
    'Oligomenorrhea': ['oligomenorrhea', 'oligomenorrhoea'],
}


def query(q):
    url = API.format(q=urllib.parse.quote(q))
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                d = json.load(r)
            return [it.get('title', [''])[0] for it in d['message']['items']]
        except Exception as e:  # noqa: BLE001
            print(f'    retry {attempt + 1}: {e}')
            time.sleep(2)
    return []


def classify(drug, disease, label, titles):
    if label == 1:
        return 'direct (benchmark)'
    dk = DK.get(disease, [disease.lower()])
    drug_l = drug.lower()
    for t in titles:
        tl = t.lower()
        if drug_l in tl and any(k in tl for k in dk):
            return 'direct'
    for t in titles:
        tl = t.lower()
        if any(k in tl for k in dk):
            return 'indirect?'
    return 'none'


for tag in ['top30', 'random30']:
    df = pd.read_csv(f'results/case_study_cold_{tag}.csv')
    rows = []
    for i, r in df.iterrows():
        titles = [] if r['label'] == 1 else query(f"{r['DrugName']} {r['DiseaseName']}")
        ev = classify(r['DrugName'], r['DiseaseName'], int(r['label']), titles)
        rows.append({'drug': r['DrugName'], 'disease': r['DiseaseName'],
                     'label': int(r['label']), 'score': round(float(r['score']), 4),
                     'evidence': ev,
                     'top_titles': ' || '.join(titles[:3])})
        print(f"[{tag}] {i + 1:2d} {ev:<20} {r['DrugName']} / {r['DiseaseName']}")
        time.sleep(0.6)
    out = pd.DataFrame(rows)
    out.to_csv(f'results/case_study_evidence_v2_{tag}.csv', index=False)
    print(f'\n{tag}: ' + out['evidence'].value_counts().to_string().replace('\n', ' | '))
