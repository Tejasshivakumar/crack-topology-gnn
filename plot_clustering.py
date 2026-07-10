"""
T6 — Clustering stratified evaluation figure (publication quality)

Two-panel figure:
  Left:  Stratified node AP — all 5 models × {simple, complex} crack stratum
  Right: Quality-filter node AP — all 5 models × {all graphs, clean graphs}

Data: outputs/clustering/stratified_results.json

Outputs:
  outputs/figures/clustering_stratified.png  (300 dpi)
  outputs/figures/clustering_stratified.pdf  (vector)
"""

import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Load data ──────────────────────────────────────────────────────────────────
data = json.load(open('outputs/clustering/stratified_results.json'))

MODELS      = ['mlp', 'gcn', 'gat', 'sage', 'gine']
MODEL_LABEL = ['MLP\n(baseline)', 'GCN', 'GAT', 'SAGE', 'GINE']
MODEL_COLOUR = {
    'mlp':  '#9e9e9e',   # grey  — no-graph baseline
    'gcn':  '#5c85d6',   # blue
    'gat':  '#9c5cb4',   # purple
    'sage': '#4caf50',   # green
    'gine': '#f44336',   # red — best topology model
}

# ── Panel A: Stratified ────────────────────────────────────────────────────────
strat = data['stratified']

ap_s0 = [strat[m]['stratum_0']['node_ap'] for m in MODELS]   # simple  cracks
ap_s1 = [strat[m]['stratum_1']['node_ap'] for m in MODELS]   # complex cracks
drops = [s0 - s1 for s0, s1 in zip(ap_s0, ap_s1)]

n_s0 = strat['mlp']['stratum_0']['n_graphs']   # 227
n_s1 = strat['mlp']['stratum_1']['n_graphs']   # 409

# ── Panel B: Quality filter ────────────────────────────────────────────────────
qf = data['quality_filter']

ap_all   = [qf[m]['all_graphs']['node_ap']   for m in MODELS]
ap_clean = [qf[m]['clean_graphs']['node_ap'] for m in MODELS]

n_all   = qf['mlp']['all_graphs']['n_graphs']    # 636
n_clean = qf['mlp']['clean_graphs']['n_graphs']  # 588

# ── Figure ─────────────────────────────────────────────────────────────────────
fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 5.2))
fig.subplots_adjust(wspace=0.38)

x = np.arange(len(MODELS))
w = 0.34   # bar width

# ── Panel A ────────────────────────────────────────────────────────────────────
bars_s0 = ax_a.bar(x - w/2, ap_s0, width=w,
                   color=[MODEL_COLOUR[m] for m in MODELS],
                   alpha=0.95, edgecolor='white', linewidth=0.6,
                   label=f'Simple / linear  (n={n_s0})')

bars_s1 = ax_a.bar(x + w/2, ap_s1, width=w,
                   color=[MODEL_COLOUR[m] for m in MODELS],
                   alpha=0.45, edgecolor='white', linewidth=0.6,
                   label=f'Complex / branching  (n={n_s1})')

# Drop arrows for MLP and GINE only
for idx, m in enumerate(MODELS):
    if m not in ('mlp', 'gine'):
        continue
    d = drops[idx]
    ax_a.annotate(
        '',
        xy=(x[idx] + w/2, ap_s1[idx] + 0.005),
        xytext=(x[idx] - w/2, ap_s0[idx] - 0.005),
        arrowprops=dict(arrowstyle='->', color='#333333', lw=1.4),
    )
    ax_a.text(x[idx] + w/2 + 0.05, (ap_s0[idx] + ap_s1[idx]) / 2,
              f'−{d:.3f}', va='center', ha='left', fontsize=8.5, color='#333333')

# Value labels on bars
for bars in (bars_s0, bars_s1):
    for bar in bars:
        h = bar.get_height()
        ax_a.text(bar.get_x() + bar.get_width() / 2, h + 0.007,
                  f'{h:.3f}', ha='center', va='bottom', fontsize=7.5, color='#333333')

ax_a.set_xticks(x)
ax_a.set_xticklabels(MODEL_LABEL, fontsize=10)
ax_a.set_ylabel('Node AP (↑ better)', fontsize=11, labelpad=6)
ax_a.set_ylim(0, 0.82)
ax_a.set_title('(A) Stratified by Crack Topology Complexity\n'
               'K-Means strata — low vs high branching ratio',
               fontsize=11, pad=10)
ax_a.spines['top'].set_visible(False)
ax_a.spines['right'].set_visible(False)

# Legend for panel A — solid vs faded
patch_s0 = mpatches.Patch(facecolor='#888888', alpha=0.95, label=f'Simple/linear (n={n_s0})')
patch_s1 = mpatches.Patch(facecolor='#888888', alpha=0.42, label=f'Complex/branching (n={n_s1})')
ax_a.legend(handles=[patch_s0, patch_s1], loc='upper right', fontsize=8.5,
            framealpha=0.85, edgecolor='#cccccc')

# Annotation box
ax_a.annotate(
    'GNN advantage concentrates\nin complex branching cracks\n(MLP: −0.320  |  GINE: −0.139)',
    xy=(x[MODELS.index('gine')] + w/2, ap_s1[MODELS.index('gine')]),
    xytext=(3.6, 0.70),
    fontsize=8, color='#222222',
    bbox=dict(boxstyle='round,pad=0.35', facecolor='#fff9c4', edgecolor='#cccc00', alpha=0.9),
    arrowprops=dict(arrowstyle='->', color='#555555', lw=1.0),
)

# ── Panel B ────────────────────────────────────────────────────────────────────
bars_all   = ax_b.bar(x - w/2, ap_all,   width=w,
                      color=[MODEL_COLOUR[m] for m in MODELS],
                      alpha=0.95, edgecolor='white', linewidth=0.6,
                      label=f'All test graphs  (n={n_all})')

bars_clean = ax_b.bar(x + w/2, ap_clean, width=w,
                      color=[MODEL_COLOUR[m] for m in MODELS],
                      alpha=0.45, edgecolor='white', linewidth=0.6,
                      label=f'After removing artifacts  (n={n_clean})')

# Value labels
for bars in (bars_all, bars_clean):
    for bar in bars:
        h = bar.get_height()
        ax_b.text(bar.get_x() + bar.get_width() / 2, h + 0.007,
                  f'{h:.3f}', ha='center', va='bottom', fontsize=7.5, color='#333333')

ax_b.set_xticks(x)
ax_b.set_xticklabels(MODEL_LABEL, fontsize=10)
ax_b.set_ylabel('Node AP (↑ better)', fontsize=11, labelpad=6)
ax_b.set_ylim(0, 0.82)
ax_b.set_title('(B) Quality Filter — Removing Skeleton Artifacts\n'
               '48 degenerate graphs removed (spectral cluster, zero junctions)',
               fontsize=11, pad=10)
ax_b.spines['top'].set_visible(False)
ax_b.spines['right'].set_visible(False)

patch_all   = mpatches.Patch(facecolor='#888888', alpha=0.95, label=f'All graphs (n={n_all})')
patch_clean = mpatches.Patch(facecolor='#888888', alpha=0.42, label=f'Artifacts removed (n={n_clean})')
ax_b.legend(handles=[patch_all, patch_clean], loc='upper left', fontsize=8.5,
            framealpha=0.85, edgecolor='#cccccc')

ax_b.annotate(
    'Rankings unchanged after\nremoving skeleton artifacts\n— topology proof is robust',
    xy=(x[MODELS.index('gine')] + w/2, ap_clean[MODELS.index('gine')]),
    xytext=(2.8, 0.70),
    fontsize=8, color='#222222',
    bbox=dict(boxstyle='round,pad=0.35', facecolor='#e8f5e9', edgecolor='#81c784', alpha=0.9),
    arrowprops=dict(arrowstyle='->', color='#555555', lw=1.0),
)

# ── Model colour legend (shared) ───────────────────────────────────────────────
model_patches = [
    mpatches.Patch(color=MODEL_COLOUR[m], label=lbl)
    for m, lbl in zip(MODELS, ['MLP (baseline)', 'GCN', 'GAT', 'SAGE', 'GINE (best)'])
]
fig.legend(handles=model_patches, loc='lower center', ncol=5,
           fontsize=9, framealpha=0.85, edgecolor='#cccccc',
           bbox_to_anchor=(0.5, -0.02))

fig.suptitle('Clustering Analysis — Node Task AP by Crack Topology Stratum',
             fontsize=13, y=1.01)

# ── Save ───────────────────────────────────────────────────────────────────────
os.makedirs('outputs/figures', exist_ok=True)
png_path = 'outputs/figures/clustering_stratified.png'
pdf_path = 'outputs/figures/clustering_stratified.pdf'
fig.savefig(png_path, dpi=300, bbox_inches='tight')
fig.savefig(pdf_path, bbox_inches='tight')
print(f'Saved → {png_path}')
print(f'Saved → {pdf_path}')

# ── Print summary ──────────────────────────────────────────────────────────────
print()
print('Stratified Node AP (simple vs complex cracks):')
print(f'{"Model":<8}  {"Simple":>8}  {"Complex":>8}  {"Drop":>8}')
print('-' * 40)
for m, lbl, s0, s1, d in zip(MODELS, MODEL_LABEL, ap_s0, ap_s1, drops):
    print(f'{m:<8}  {s0:>8.4f}  {s1:>8.4f}  {-d:>+8.4f}')

print()
print('Key finding:')
print(f'  MLP node AP gap (simple→complex): −{drops[MODELS.index("mlp")]:.3f}  (−{drops[MODELS.index("mlp")]/ap_s0[MODELS.index("mlp")]:.0%})')
print(f'  GINE node AP gap (simple→complex): −{drops[MODELS.index("gine")]:.3f}  (−{drops[MODELS.index("gine")]/ap_s0[MODELS.index("gine")]:.0%})')
print(f'  GINE advantage in simple cracks:  {ap_s0[MODELS.index("gine")] - ap_s0[MODELS.index("mlp")]:+.3f}')
print(f'  GINE advantage in complex cracks: {ap_s1[MODELS.index("gine")] - ap_s1[MODELS.index("mlp")]:+.3f}')
