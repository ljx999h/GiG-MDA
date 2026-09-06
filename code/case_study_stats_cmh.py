# -*- coding: utf-8 -*-
"""
案例研究统计 (机器可读, P1-7): 三视图 2x2 + CMH + Breslow-Day + 双审 kappa.
输入: results/case_study_deploy_evidence_top15_{global,drug,gastric}_verified.json (R1)
      results/second_reviewer_blind.json (R2)
      results/case_study_deploy_random15_{tag}_verified.json (controls, R1)
输出: results/case_study_stats_cmh.json + .csv
"""
import csv
import json

from statsmodels.stats.contingency_tables import StratifiedTable


def load(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


TAGS = ['global', 'drug', 'gastric']
main = {t: load(f'results/case_study_deploy_evidence_top15_{t}_verified.json') for t in TAGS}
ctrl = {t: load(f'results/case_study_deploy_evidence_random15_{t}_verified.json')
        for t in TAGS}
r2 = load('results/second_reviewer_blind.json')

# 2x2 每视图: [[main_pos, main_neg], [ctrl_pos, ctrl_neg]]
tables, rows = [], []
for t in TAGS:
    mp = sum(1 for x in main[t] if x['evidence'] == 'direct')
    cp = sum(1 for x in ctrl[t] if x['evidence'] == 'direct')
    tables.append([[mp, 15 - mp], [cp, 15 - cp]])
    rows.append({'view': t, 'main_direct': mp, 'main_n': 15,
                 'ctrl_direct': cp, 'ctrl_n': 15})
st = StratifiedTable(tables)
or_mh = float(st.oddsratio_pooled)
chi2 = float(st.test_null_odds().statistic)
p_mh = float(st.test_null_odds().pvalue)
p_bd = float(st.test_equal_odds().pvalue)

# kappa (39/45 初判一致)
agree = sum(1 for a, b in zip([x for t in TAGS for x in main[t]], r2)
            if a['evidence'] == b['evidence'])
p0 = agree / 45
n_d1 = sum(1 for x in [y for t in TAGS for y in main[t]] if x['evidence'] == 'direct')
p1d, p1n = n_d1 / 45, 1 - n_d1 / 45
n_d2 = sum(1 for x in r2 if x['evidence'] == 'direct')
p2d, p2n = n_d2 / 45, 1 - n_d2 / 45
pe = p1d * p2d + p1n * p2n
kappa = (p0 - pe) / (1 - pe)

out = {'per_view': rows,
       'cmh': {'odds_ratio': round(or_mh, 3), 'chi2': round(chi2, 3),
               'p': p_mh, 'breslow_day_p': p_bd},
       'reviewer_agreement': {'agree': agree, 'n': 45, 'kappa': round(kappa, 3)}}
with open('results/case_study_stats_cmh.json', 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
with open('results/case_study_stats_cmh.csv', 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=['view', 'main_direct', 'main_n',
                                      'ctrl_direct', 'ctrl_n'])
    w.writeheader()
    w.writerows(rows)
print(json.dumps(out, indent=1))
