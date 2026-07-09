"""
T12 — Edge Feature Ablation Figure (publication quality)

Reads outputs/edge_ablation_results.json and produces a horizontal bar chart
showing how much Node AP drops when each edge feature group is removed from GINE.

Output: outputs/figures/edge_feature_ablation.png  (300 dpi)
        outputs/figures/edge_feature_ablation.pdf  (vector)
"""

import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Load data ──────────────────────────────────────────────────────────────────
data = json.load(open('outputs/edge_ablation_results.json'))

BASELINE_AP = data['full']['node_ap']   # 0.7265

# Define display order (ascending drop = bars get longer going down)
rows = [
    ('no_angle',     'Angle encoding\n[sin(2θ), cos(2θ)]'),
    ('no_tortuosity','Tortuosity\n[path / euclid dist]'),
    ('no_geometry',  'Geometry\n[path length + euclidean dist]'),
    ('no_thickness', 'Crack thickness\n[avg / min / max]'),
    ('no_edge_feats','All edge features\n(GCN-like baseline)'),
]

labels  = [r[1] for r in rows]
aps     = [data[r[0]]['node_ap'] for r in rows]
drops   = [BASELINE_AP - ap for ap in aps]

# ── Colour by severity ─────────────────────────────────────────────────────────
def bar_colour(drop):
    if drop < 0.05:
        return '#4caf50'   # green  — negligible
    elif drop < 0.15:
        return '#ff9800'   # orange — moderate
    else:
        return '#f44336'   # red    — critical

colours = [bar_colour(d) for d in drops]

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))

y = np.arange(len(rows))
bars = ax.barh(y, drops, height=0.55, color=colours, edgecolor='white', linewidth=0.6)

# Annotate each bar: drop value + resulting AP
for i, (drop, ap) in enumerate(zip(drops, aps)):
    ax.text(drop + 0.005, i, f'−{drop:.3f}  (AP {ap:.3f})',
            va='center', ha='left', fontsize=9.5, color='#222222')

# Reference line at x=0 (= baseline AP drop)
ax.axvline(0, color='#333333', linewidth=1.2, linestyle='-')

# Baseline AP annotation
ax.axvline(BASELINE_AP - BASELINE_AP, color='none')   # invisible anchor
ax.text(0.003, len(rows) - 0.15,
        f'Full GINE AP = {BASELINE_AP:.3f}',
        fontsize=9, color='#333333', style='italic')

# Axis formatting
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=10)
ax.set_xlabel('Node AP Drop (↑ = more important)', fontsize=11, labelpad=8)
ax.set_xlim(-0.01, 0.42)
ax.set_title('GINE Edge Feature Ablation — Node Task (AP)\n'
             'Feature group zeroed at test time; model weights unchanged',
             fontsize=12, pad=12)

# Legend
legend_handles = [
    mpatches.Patch(color='#4caf50', label='Negligible  (< 0.05 drop)'),
    mpatches.Patch(color='#ff9800', label='Moderate  (0.05 – 0.15 drop)'),
    mpatches.Patch(color='#f44336', label='Critical  (> 0.15 drop)'),
]
ax.legend(handles=legend_handles, loc='lower right', fontsize=9,
          framealpha=0.85, edgecolor='#cccccc')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(axis='x', labelsize=9.5)

plt.tight_layout()

# ── Save ───────────────────────────────────────────────────────────────────────
os.makedirs('outputs/figures', exist_ok=True)
png_path = 'outputs/figures/edge_feature_ablation.png'
pdf_path = 'outputs/figures/edge_feature_ablation.pdf'
fig.savefig(png_path, dpi=300, bbox_inches='tight')
fig.savefig(pdf_path, bbox_inches='tight')
print(f'Saved → {png_path}')
print(f'Saved → {pdf_path}')

# ── Print summary table ────────────────────────────────────────────────────────
print()
print(f'{"Feature group dropped":<38}  {"Node AP":>8}  {"Drop":>8}')
print('-' * 58)
print(f'{"Full GINE (baseline)":<38}  {BASELINE_AP:>8.4f}  {"—":>8}')
for (key, label_raw), ap, drop in zip(rows, aps, drops):
    label_short = label_raw.split('\n')[0]
    print(f'{label_short:<38}  {ap:>8.4f}  {-drop:>+8.4f}')
