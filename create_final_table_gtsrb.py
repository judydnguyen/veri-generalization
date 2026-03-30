import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

plt.style.use('default')
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR

parser = argparse.ArgumentParser()
parser.add_argument('combined_csv', type=Path,
                    default=SCRIPT_DIR / "alpha_beta_CROWN/results/gtsrb_combined.csv",
                    nargs='?', help='Path to the combined GTSRB/BTSD/OOD-PixMix results CSV')
args = parser.parse_args()

# Load combined CSV
combined_csv = args.combined_csv
all_results = pd.read_csv(combined_csv)

# Normalize dataset column to display names
DATASET_NAME_MAP = {
    'gtsrb':           'GTSRB',
    'btsd':            'BTSD',
    'gtsrb_ood_pixmix':'OOD-PixMix',
    'gtsrb_pixmix':    'OOD-PixMix',
}
all_results['dataset'] = all_results['dataset'].map(DATASET_NAME_MAP).fillna(all_results['dataset'])

# Derive model_id from filename stem
all_results['model_id'] = all_results['model_path'].apply(
    lambda p: os.path.splitext(os.path.basename(str(p)))[0] if pd.notna(p) else None
)

# Calculate CRA (Certified Robust Accuracy)
all_results['CRA'] = (
    all_results['safe'] /
    (all_results['safe'] + all_results['unsafe'] + all_results['unknown']) * 100
)

print("Dataset distribution:")
print(all_results['dataset'].value_counts())
print("\nModel distribution:")
print(all_results['model_id'].value_counts())
print()

DATASETS_DISPLAY = ['GTSRB', 'BTSD', 'OOD-PixMix']

# ── Summary table ────────────────────────────────────────────────────────────
summary_data = []
for model_id, group in all_results.groupby('model_id'):
    for dataset in DATASETS_DISPLAY:
        ds_rows = group[group['dataset'] == dataset]
        if ds_rows.empty:
            continue

        avg_safe    = ds_rows['safe'].mean()
        avg_unsafe  = ds_rows['unsafe'].mean()
        avg_CRA     = ds_rows['CRA'].mean()
        avg_time    = ds_rows['time'].mean()

        eps4 = ds_rows[ds_rows['epsilon'] == '4/255']
        eps4_CRA    = eps4['CRA'].iloc[0]    if not eps4.empty else None
        eps4_safe   = eps4['safe'].iloc[0]   if not eps4.empty else None
        eps4_unsafe = eps4['unsafe'].iloc[0] if not eps4.empty else None

        summary_data.append({
            'Model':            model_id,
            'Dataset':          dataset,
            'Avg_Safe':         round(avg_safe, 1),
            'Avg_Unsafe':       round(avg_unsafe, 1),
            'Avg_CRA (%)':      round(avg_CRA, 2),
            'CRA_eps4/255 (%)': round(eps4_CRA, 2) if eps4_CRA is not None else None,
            'Eps4_Safe':        int(eps4_safe)      if eps4_safe is not None else None,
            'Eps4_Unsafe':      int(eps4_unsafe)    if eps4_unsafe is not None else None,
            'Avg_Time':         round(avg_time, 4),
        })

final_table = pd.DataFrame(summary_data)
final_table = final_table[[
    'Model', 'Dataset',
    'Avg_Safe', 'Avg_Unsafe', 'Avg_CRA (%)', 'CRA_eps4/255 (%)',
    'Eps4_Safe', 'Eps4_Unsafe', 'Avg_Time'
]]

output_file = BASE_DIR / "final_gtsrb_results.csv"
final_table.to_csv(output_file, index=False)
print(f"Final table saved to {output_file}")
print("\n" + "=" * 100)
print("Final Combined Results Table:")
print("=" * 100)
print(final_table.to_string(index=False))

# ── Plots ────────────────────────────────────────────────────────────────────
model_ids  = sorted(all_results['model_id'].dropna().unique())
eps_values = [1/255, 2/255, 3/255, 4/255]
eps_strs   = [f'{round(e*255)}/255' for e in eps_values]

fig, axes = plt.subplots(2, 2, figsize=(18, 12))
ax1, ax2, ax3, ax4 = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

# colour/marker cycling
cmap      = plt.cm.tab10
markers   = ['o', 's', '^', 'D', 'v', 'P', 'X']
color_of  = {m: cmap(i / max(len(model_ids) - 1, 1)) for i, m in enumerate(model_ids)}
marker_of = {m: markers[i % len(markers)]             for i, m in enumerate(model_ids)}

# ── Plot 1: CRA vs epsilon – GTSRB (in-distribution) ───────────────────────
for model_id in model_ids:
    rows = all_results[(all_results['model_id'] == model_id) & (all_results['dataset'] == 'GTSRB')]
    if rows.empty:
        continue
    cra = [rows[rows['epsilon'] == e]['CRA'].iloc[0] if not rows[rows['epsilon'] == e].empty else np.nan
           for e in eps_strs]
    ax1.plot(eps_values, cra, marker=marker_of[model_id], color=color_of[model_id],
             label=model_id, linewidth=2.5, markersize=10)

ax1.set_xlabel('Epsilon (perturbation)', fontsize=13, fontweight='bold')
ax1.set_ylabel('CRA (%)', fontsize=13, fontweight='bold')
ax1.set_title('CRA vs Epsilon – GTSRB (in-distribution)', fontsize=15, fontweight='bold')
ax1.legend(fontsize=9, loc='best')
ax1.set_ylim(bottom=0)

# ── Plot 2: Avg Safe vs Unsafe stacked bar (GTSRB / BTSD / OOD-PixMix) ──────
width   = 0.25
x       = np.arange(len(model_ids))
offsets = {'GTSRB': -width, 'BTSD': 0, 'OOD-PixMix': width}
hatch   = {'GTSRB': '',     'BTSD': '---', 'OOD-PixMix': '///'}
safe_color   = '#2ecc71'
unsafe_color = '#e74c3c'

for dataset in DATASETS_DISPLAY:
    safe_vals   = []
    unsafe_vals = []
    for model_id in model_ids:
        ds_rows = all_results[(all_results['model_id'] == model_id) & (all_results['dataset'] == dataset)]
        safe_vals.append(ds_rows['safe'].mean()   if not ds_rows.empty else 0)
        unsafe_vals.append(ds_rows['unsafe'].mean() if not ds_rows.empty else 0)
    off = offsets[dataset]
    ht  = hatch[dataset]
    ax2.bar(x + off, safe_vals,   width, label=f'{dataset} Safe',
            alpha=0.8, color=safe_color,   hatch=ht)
    ax2.bar(x + off, unsafe_vals, width, bottom=safe_vals,
            label=f'{dataset} Unsafe', alpha=0.8, color=unsafe_color, hatch=ht)

ax2.set_xlabel('Model', fontsize=13, fontweight='bold')
ax2.set_ylabel('Average Count', fontsize=13, fontweight='bold')
ax2.set_title('Average Safe vs Unsafe by Model (Stacked)', fontsize=15, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(model_ids, fontsize=8, rotation=15, ha='right')
ax2.legend(fontsize=8, loc='upper left', ncol=2)
ax2.grid(True, alpha=0.3, axis='y')

# ── Plot 3: CRA vs epsilon – GTSRB vs BTSD ──────────────────────────────────
colors_ds  = {'GTSRB': '#3498db', 'BTSD': '#e67e22'}
linestyles = {'GTSRB': '-',       'BTSD': '--'}

for model_id in model_ids:
    for dataset in ['GTSRB', 'BTSD']:
        rows = all_results[(all_results['model_id'] == model_id) & (all_results['dataset'] == dataset)]
        if rows.empty:
            continue
        cra = [rows[rows['epsilon'] == e]['CRA'].iloc[0] if not rows[rows['epsilon'] == e].empty else np.nan
               for e in eps_strs]
        ax3.plot(eps_values, cra,
                 marker=marker_of[model_id], color=colors_ds[dataset],
                 linestyle=linestyles[dataset],
                 label=f'{model_id}-{dataset}', linewidth=2, markersize=8, alpha=0.8)

ax3.set_xlabel('Epsilon (perturbation)', fontsize=13, fontweight='bold')
ax3.set_ylabel('CRA (%)', fontsize=13, fontweight='bold')
ax3.set_title('CRA: GTSRB vs BTSD by Epsilon', fontsize=15, fontweight='bold')
ax3.legend(fontsize=8, loc='best', ncol=2)
ax3.set_ylim(bottom=0)

# ── Plot 4: CRA vs epsilon – GTSRB vs OOD-PixMix ────────────────────────────
colors_ds2  = {'GTSRB': '#3498db', 'OOD-PixMix': '#e67e22'}
linestyles2 = {'GTSRB': '-',       'OOD-PixMix': '--'}

for model_id in model_ids:
    for dataset in ['GTSRB', 'OOD-PixMix']:
        rows = all_results[(all_results['model_id'] == model_id) & (all_results['dataset'] == dataset)]
        if rows.empty:
            continue
        cra = [rows[rows['epsilon'] == e]['CRA'].iloc[0] if not rows[rows['epsilon'] == e].empty else np.nan
               for e in eps_strs]
        ax4.plot(eps_values, cra,
                 marker=marker_of[model_id], color=colors_ds2[dataset],
                 linestyle=linestyles2[dataset],
                 label=f'{model_id}-{dataset}', linewidth=2, markersize=8, alpha=0.8)

ax4.set_xlabel('Epsilon (perturbation)', fontsize=13, fontweight='bold')
ax4.set_ylabel('CRA (%)', fontsize=13, fontweight='bold')
ax4.set_title('CRA: GTSRB vs OOD-PixMix by Epsilon', fontsize=15, fontweight='bold')
ax4.legend(fontsize=8, loc='best', ncol=2)
ax4.set_ylim(bottom=0)

plt.tight_layout()
plot_file = BASE_DIR / "final_gtsrb_results_plots.png"
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
print(f"\nPlots saved to {plot_file}")

# ── Detailed breakdown ────────────────────────────────────────────────────────
print("\n" + "=" * 100)
print("Detailed Breakdown by Epsilon:")
print("=" * 100)
for dataset in DATASETS_DISPLAY:
    print(f"\n{dataset}:")
    detail = (
        all_results[all_results['dataset'] == dataset][['model_id', 'epsilon', 'safe', 'unsafe', 'CRA', 'time']]
        .copy()
        .sort_values(['model_id', 'epsilon'])
    )
    detail.columns = ['Model', 'Epsilon', 'Safe', 'Unsafe', 'CRA (%)', 'Time']
    detail['CRA (%)'] = detail['CRA (%)'].round(2)
    detail['Time']    = detail['Time'].round(4)
    print(detail.to_string(index=False))

print("\n" + "=" * 100)
print("Analysis complete!")
print("=" * 100)
