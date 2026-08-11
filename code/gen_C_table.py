"""
C-Dataset 完整对比表生成器
 读取:
   - results/baselines_orig/{AMDGT,DDAGDL,DRHGCN,DRWBNCF,HINGRL,PREDICT}_C.csv (6 SOTA)
   - results/contribution_demo_C/对比.csv (你的 4 变体, 5-Fold CV)
 输出:
   - results/C_Dataset/final_comparison_table.csv (一行一方法的汇总)
   - print 终端表格
"""
import os
import pandas as pd

# 相对路径: 项目根目录 = 脚本上级目录的上级
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
results = []

# 6 SOTA baselines
sota_files = ['AMDGT', 'DDAGDL', 'DRHGCN', 'DRWBNCF', 'HINGRL', 'PREDICT']
sota_display = {
    'AMDGT':   'AMDGT (Transformer)',
    'DDAGDL':  'DDAGDL (HIN)',
    'DRHGCN':  'DRHGCN (GNN)',
    'DRWBNCF': 'DRWBNCF (NN)',
    'HINGRL':  'HINGRL (GNN, RF n=999)',
    'PREDICT': 'PREDICT (LR)',
}
for f in sota_files:
    path = f"{BASE}/results/baselines_orig/{f}_C.csv"
    df = pd.read_csv(path)
    row = df.iloc[0].to_dict()
    results.append({
        'Method':     sota_display[f],
        'Dim':        '-',
        'AUROC':      row['AUROC'],
        'AUPR':       row['AUPR'],
        'Accuracy':   row['Accuracy'],
        'Recall':     row['Recall'],
        'F1':         row['F1'],
    })

# 我的 4 变体 (按 F1 降序排, 你的最优放第一)
mypath = f"{BASE}/results/contribution_demo_C/对比.csv"
mine = pd.read_csv(mypath)
mine_display = {
    '1D Dot + RF (论文方法)':         'MiRAGE 1D+RF (论文方法)',
    '128D Embed + RF (扩展)':         'MiRAGE 128D+RF (扩展)',
    '1D Dot + XGBoost (扩展)':        'MiRAGE 1D+XGB (扩展)',
    '128D Embed + XGBoost (你的最优)': 'MiRAGE 128D+XGB (你的) ⭐',
}
for _, r in mine.iterrows():
    name = r['Method']
    if name in mine_display:
        results.append({
            'Method':     mine_display[name],
            'Dim':        int(r['Dimension']),
            'AUROC':      r['AUROC_mean'],
            'AUPR':       r['AUPRC_mean'],
            'Accuracy':   r['Accuracy_mean'],
            'Recall':     r['Recall_mean'],
            'F1':         r['F1_mean'],
        })

# 按 F1 降序排 (你的最优第一)
df = pd.DataFrame(results)
df = df.sort_values('F1', ascending=False).reset_index(drop=True)

# 保存
out_csv = f"{BASE}/results/C_Dataset/final_comparison_table.csv"
os.makedirs(os.path.dirname(out_csv), exist_ok=True)
df.to_csv(out_csv, index=False)

# 终端打印对齐表格
print()
print("=" * 110)
print(" C-Dataset 完整对比表 (我重跑 5-Fold CV)")
print("=" * 110)
header = f"{'方法':<33s} {'Dim':>5s} {'AUROC':>8s} {'AUPR':>8s} {'Accuracy':>10s} {'Recall':>8s} {'F1':>8s}"
print(header)
print("-" * 110)
for _, r in df.iterrows():
    print(f"{r['Method']:<33s} {str(r['Dim']):>5s} {r['AUROC']:>8.4f} {r['AUPR']:>8.4f} "
          f"{r['Accuracy']:>10.4f} {r['Recall']:>8.4f} {r['F1']:>8.4f}")
print("=" * 110)
print(f"\n  完整 CSV 已保存: {out_csv}")
