"""
重现 MiRAGE 128D+XGB (你的最优) 在 C-Dataset 上的 5-Fold CV 数字
 读:
   - results/contribution_demo_C/128D_XGB每折.csv (5 折的 AUROC/AUPR/Acc/Recall/F1)
 算: mean ± std
 打印: 完整 5 指标表格
"""
import os
import pandas as pd

# 相对路径: 项目根目录 = 脚本上级目录的上级
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
csv_path = os.path.join(BASE, "results", "contribution_demo_C", "128D_XGB每折.csv")

if not os.path.exists(csv_path):
    print(f"[ERROR] 找不到: {csv_path}")
    print(f"        请先跑: python code/contribution_demo_C.py")
    raise SystemExit(1)

# 读 5 折结果
df = pd.read_csv(csv_path)
print("=" * 70)
print(" MiRAGE 128D Embed + XGBoost (你的最优) — C-Dataset 5-Fold CV")
print("=" * 70)
print()
print(df.to_string(index=False))
print()

# 计算 mean ± std
metrics = {
    'AUROC':     df['AUROC'].mean(),
    'AUPR':      df['AUPRC'].mean(),
    'Accuracy':  df['Accuracy'].mean(),
    'Recall':    df['Recall'].mean(),
    'F1':        df['F1-Score'].mean(),
}
stds = {
    'AUROC':     df['AUROC'].std(),
    'AUPR':      df['AUPRC'].std(),
    'Accuracy':  df['Accuracy'].std(),
    'Recall':    df['Recall'].std(),
    'F1':        df['F1-Score'].std(),
}

print("─" * 70)
print("汇总 (mean over 5 folds):")
print("─" * 70)
print(f"{'Metric':<10s}  {'Mean ± Std':>20s}")
print("─" * 70)
for m in metrics:
    print(f"{m:<10s}  {metrics[m]:.4f} ± {stds[m]:.4f}")
print("─" * 70)
print()
print("─" * 70)
print("最终对比表 (1 行):")
print("─" * 70)
print(f"  MiRAGE 128D+XGB (你的) ⭐ │ 146 │ {metrics['AUROC']:.4f} │ "
      f"{metrics['AUPR']:.4f} │ {metrics['Accuracy']:.4f} │ "
      f"{metrics['Recall']:.4f} │ {metrics['F1']:.4f}")
print("─" * 70)
