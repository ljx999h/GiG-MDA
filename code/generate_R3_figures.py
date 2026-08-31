"""
R3 论文图生成 (generate_R3_figures)

fig1_cold_lift.png : 冷启动 lift 条形图 (8 划分 × 2 通道, 分面 C/DDCD)
fig2_framework.png : 框架流程示意图

调色板 (dataviz 校验 PASS): MolEmb32 #2a78d6 (blue), GRMF embed #eb6834 (orange)
数据: results/R2/cold_start_results.csv
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

R2 = 'results/R2'
BLUE = '#2a78d6'
ORANGE = '#eb6834'
INK = '#0b0b0b'
GRID = '#d9d9d4'
SURFACE = '#fcfcfb'

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 9,
    'axes.facecolor': SURFACE, 'figure.facecolor': SURFACE,
    'axes.edgecolor': INK, 'axes.linewidth': 0.8,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
    'axes.axisbelow': True,
})


def fig_cold_lift():
    df = pd.read_csv(os.path.join(R2, 'cold_start_results.csv'))
    df = df[df['dataset'].isin(['C', 'DDCD'])]
    labels = [f"{r['dataset']}-{r['cold_seed']}" for _, r in df.iterrows()]
    mol = df['MolEmb32_lift'].values * 100
    emb = df['embed_lift'].values * 100

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), sharey=True)
    for ax, ds in zip(axes, ['C', 'DDCD']):
        m = df['dataset'] == ds
        x = np.arange(m.sum())
        w = 0.36
        ax.bar(x - w/2, mol[m], w, color=BLUE, edgecolor='white', linewidth=0.8, label='MolEmb32 (molecular channel)')
        ax.bar(x + w/2, emb[m], w, color=ORANGE, edgecolor='white', linewidth=0.8, label='GRMF embeddings (graph channel)')
        ax.set_xticks(x)
        ax.set_xticklabels([f"s{int(s)}" for s in df[m]['cold_seed']], fontsize=8)
        ax.set_title(ds + '-Dataset', fontsize=10)
        ax.set_ylim(0, 240)
        ax.set_yticks([0, 50, 100, 150, 200])
        for xi, v in zip(x - w/2, mol[m]):
            ax.text(xi, v + 4, f"{v:+.0f}%", ha='center', va='bottom', fontsize=7, color=INK)
        for xi, v in zip(x + w/2, emb[m]):
            ax.text(xi, v + 4, f"{v:+.0f}%", ha='center', va='bottom', fontsize=7, color=INK)
    axes[0].set_ylabel('AUPR lift vs base (%)')
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc='lower center', ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.08), fontsize=8)
    fig.suptitle('Cold-start (cold-drug 20%) AUPR lift over base MiRAGE features', fontsize=11)
    fig.tight_layout(rect=[0, 0.08, 1, 0.96])
    out = os.path.join(R2, 'fig1_cold_lift.png')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('✅', out)


def fig_framework():
    fig, ax = plt.subplots(figsize=(9.5, 3.4))
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)

    def box(x, y, w, h, text, fc='#f2f4f7', ec=INK, fs=8):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.06',
                                    fc=fc, ec=ec, lw=0.8))
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fs, color=INK)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>',
                                     mutation_scale=11, lw=1.0, color=INK))

    # 输入
    box(0.15, 3.6, 1.5, 1.0, 'Drug\n(DrugBank\nfeatures + SMILES)')
    box(0.15, 0.7, 1.5, 1.0, 'Disease\n(CTD features)')

    # 通道
    box(2.4, 3.7, 2.3, 0.9, 'GBA features\n(22-d, neighbor-max scoring)', fs=7.5)
    box(2.4, 2.4, 2.3, 0.9, 'Molecular channel\n(MoLFormer 768-d → PCA-32)', fs=7.5)
    box(2.4, 1.1, 2.3, 0.9, 'Graph channel\n(GRMF embeddings, 128-d)', fs=7.5)

    # 拼接
    box(5.5, 2.4, 1.5, 1.0, 'Concatenate\n(feature vector)', fs=8)

    # 分类器
    box(7.4, 2.4, 1.3, 1.0, 'XGBoost\n(unified config)', fs=8)

    # 输出
    box(9.05, 2.4, 0.85, 1.0, 'Rank', fs=8)

    arrow(1.65, 4.1, 2.4, 4.15)
    arrow(1.65, 4.1, 2.4, 2.85)
    arrow(1.65, 1.2, 2.4, 1.55)
    arrow(1.65, 1.2, 2.4, 2.85)
    arrow(4.7, 4.15, 5.5, 3.05)
    arrow(4.7, 2.85, 5.5, 2.9)
    arrow(4.7, 1.55, 5.5, 2.75)
    arrow(7.0, 2.9, 7.4, 2.9)
    arrow(8.7, 2.9, 9.05, 2.9)

    ax.text(0.15, 4.85, 'Leakage-aware protocol: pair-disjoint splits; features, embeddings, and negatives built fold-locally',
            fontsize=7.5, style='italic', color='#52514e')
    ax.text(9.9, 1.0, '', fontsize=1)
    out = os.path.join(R2, 'fig2_framework.png')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('✅', out)


if __name__ == '__main__':
    fig_cold_lift()
    fig_framework()
