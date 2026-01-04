import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

# Set style for better-looking plots
plt.style.use('default')
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

# Base directory - relative to script location
# Script is in veri-generalization/, unlearning-verification/ is sibling directory
SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent / "veri-generalization"
# Load data
results_dir = BASE_DIR / "results"
robustness_file = BASE_DIR / "robustness_results.csv"

# Read robustness results
robustness_df = pd.read_csv(robustness_file)

# Read verification results
mnist_results = pd.read_csv(results_dir / "final_results_mnist_fc.csv")
emnist_results = pd.read_csv(results_dir / "final_results_emnist_fc.csv")
pixmix_results = pd.read_csv(results_dir / "final_results_mnist_pixmix_fc.csv")

# Normalize paths for matching (extract just the filename)
def normalize_path(path):
    """Normalize path for matching - extract filename"""
    if pd.isna(path) or path is None:
        return None
    return os.path.basename(str(path))

# Create mapping from filename to model_id using robustness_results.csv
# robustness_results.csv already has Model column (model_A, model_B, model_C)
# Note: column name is Model_Path (capital P) in robustness_results.csv
robustness_df['model_filename'] = robustness_df['Model_Path'].apply(normalize_path)
filename_to_model_id = dict(zip(robustness_df['model_filename'], robustness_df['Model']))

# Debug: print the mapping
print("Model filename to ID mapping:")
for filename, model_id in filename_to_model_id.items():
    print(f"  {filename} -> {model_id}")
print()

# Add model identifier to results dataframes using the mapping
mnist_results['model_filename'] = mnist_results['model_path'].apply(normalize_path)
emnist_results['model_filename'] = emnist_results['model_path'].apply(normalize_path)
pixmix_results['model_filename'] = pixmix_results['model_path'].apply(normalize_path)

mnist_results['model_id'] = mnist_results['model_filename'].map(filename_to_model_id)
emnist_results['model_id'] = emnist_results['model_filename'].map(filename_to_model_id)
pixmix_results['model_id'] = pixmix_results['model_filename'].map(filename_to_model_id)

# Verify matching
print("MNIST results model_id distribution:")
print(mnist_results['model_id'].value_counts())
print("\nEMNIST results model_id distribution:")
print(emnist_results['model_id'].value_counts())
print("\nPixMix results model_id distribution:")
print(pixmix_results['model_id'].value_counts())
print()

# Add dataset identifier
mnist_results['dataset'] = 'MNIST'
emnist_results['dataset'] = 'EMNIST'
pixmix_results['dataset'] = 'PixMix'

# Combine results
all_results = pd.concat([mnist_results, emnist_results, pixmix_results], ignore_index=True)

# Calculate CRA (Certified Robust Accuracy) - percentage of safe cases
all_results['CRA'] = all_results['safe'] / (all_results['safe'] + all_results['unsafe'] + all_results['unknown']) * 100

# Map epsilon values to Robust_Acc column names in robustness_results.csv
# epsilon values: 1/255=0.003922, 2/255=0.007843, 3/255=0.011765, 4/255=0.015686
epsilon_to_column = {
    '1/255': 'Robust_Acc_eps0.003922',
    '2/255': 'Robust_Acc_eps0.007843',
    '3/255': 'Robust_Acc_eps0.011765',
    '4/255': 'Robust_Acc_eps0.015686'
}

# Add AA from robustness_results.csv by matching epsilon and model
all_results['AA'] = np.nan
for idx, row in all_results.iterrows():
    model_filename = row['model_filename']
    epsilon = row['epsilon']
    
    # Find matching model in robustness_df
    matching_model = robustness_df[robustness_df['model_filename'] == model_filename]
    if len(matching_model) > 0 and epsilon in epsilon_to_column:
        robust_acc_col = epsilon_to_column[epsilon]
        if robust_acc_col in matching_model.columns:
            robust_acc = matching_model[robust_acc_col].iloc[0]
            # AA = Robust_Acc (use original values from robustness_results.csv)
            if pd.notna(robust_acc):
                all_results.at[idx, 'AA'] = robust_acc

# Create summary table by merging based on model_path (normalized filename)
summary_data = []

# Iterate through each model in robustness_df
for idx, model_robust in robustness_df.iterrows():
    model_id = model_robust['Model']
    model_filename = model_robust['model_filename']
    
    # Find matching results based on model_filename
    matching_results = all_results[all_results['model_filename'] == model_filename]
    
    if len(matching_results) == 0:
        print(f"Warning: No verification results found for {model_id} ({model_filename})")
        continue
    
    # Process each dataset
    for dataset in ['MNIST', 'EMNIST']:
        dataset_results = matching_results[matching_results['dataset'] == dataset]
        
        if len(dataset_results) > 0:
            # Get average metrics across all epsilon values
            avg_safe = dataset_results['safe'].mean()
            avg_unsafe = dataset_results['unsafe'].mean()
            avg_CRA = dataset_results['CRA'].mean()
            avg_AA = dataset_results['AA'].mean()  # AA from robustness_results.csv
            avg_time = dataset_results['time'].mean()
            
            # Get metrics for epsilon = 4/255 (largest perturbation)
            eps4_results = dataset_results[dataset_results['epsilon'] == '4/255']
            if len(eps4_results) > 0:
                eps4_CRA = eps4_results['CRA'].iloc[0]
                eps4_AA = eps4_results['AA'].iloc[0]
                eps4_safe = eps4_results['safe'].iloc[0]
                eps4_unsafe = eps4_results['unsafe'].iloc[0]
            else:
                eps4_CRA = None
                eps4_AA = None
                eps4_safe = None
                eps4_unsafe = None
            
            summary_data.append({
                'Model': model_id,
                'Dataset': dataset,
                'Clean_Accuracy': model_robust['Clean_Accuracy'],
                'Robust_Acc_eps0.015686': model_robust['Robust_Acc_eps0.015686'],
                'Avg_Safe': round(avg_safe, 1),
                'Avg_Unsafe': round(avg_unsafe, 1),
                'Avg_CRA (%)': round(avg_CRA, 2),
                'CRA_eps4/255 (%)': round(eps4_CRA, 2) if eps4_CRA is not None else None,
                'Avg_AA (%)': round(avg_AA, 2),
                'AA_eps4/255 (%)': round(eps4_AA, 2) if eps4_AA is not None else None,
                'Eps4_Safe': int(eps4_safe) if eps4_safe is not None else None,
                'Eps4_Unsafe': int(eps4_unsafe) if eps4_unsafe is not None else None,
                'Avg_Time': round(avg_time, 4)
            })

# Create final summary dataframe
final_table = pd.DataFrame(summary_data)

# Reorder columns
final_table = final_table[['Model', 'Dataset', 'Clean_Accuracy', 'Robust_Acc_eps0.015686', 
                           'Avg_Safe', 'Avg_Unsafe', 'Avg_CRA (%)', 'CRA_eps4/255 (%)',
                           'Avg_AA (%)', 'AA_eps4/255 (%)',
                           'Eps4_Safe', 'Eps4_Unsafe', 'Avg_Time']]

# Save final table
output_file = BASE_DIR / "final_combined_results.csv"
final_table.to_csv(output_file, index=False)
print(f"Final table saved to {output_file}")
print("\n" + "="*100)
print("Final Combined Results Table:")
print("="*100)
print(final_table.to_string(index=False))

# Create plots - now with 4 plots (2x2 layout)
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
ax1 = axes[0, 0]
ax2 = axes[0, 1]
ax3 = axes[1, 0]
ax4 = axes[1, 1]

# Plot 1: AA by epsilon for each model (MNIST)
for model_id in ['model_A', 'model_B', 'model_C', 'model_D']:
    model_data = all_results[
        (all_results['model_id'] == model_id) & 
        (all_results['dataset'] == 'MNIST')
    ]
    if len(model_data) > 0:
        eps_values = [1/255, 2/255, 3/255, 4/255]
        aa_values = []
        for eps in eps_values:
            # FIX: Use round() instead of int() to avoid floating-point precision issues
            eps_str = f'{round(eps*255)}/255'
            eps_data = model_data[model_data['epsilon'] == eps_str]
            if len(eps_data) > 0:
                aa_values.append(eps_data['AA'].iloc[0])
            else:
                aa_values.append(np.nan)
        ax1.plot(eps_values, aa_values, marker='o', label=model_id, linewidth=2.5, markersize=10)
ax1.set_xlabel('Epsilon (perturbation)', fontsize=13, fontweight='bold')
ax1.set_ylabel('AA (%)', fontsize=13, fontweight='bold')
ax1.set_title('AA vs Epsilon - MNIST', fontsize=15, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(bottom=0)

# Plot 2: Comparison of Safe vs Unsafe cases grouped by model (including PixMix)
comparison_data = []
for model_id in ['model_A', 'model_B', 'model_C', 'model_D']:
    for dataset in ['MNIST', 'EMNIST', 'PixMix']:
        model_data = all_results[
            (all_results['model_id'] == model_id) & 
            (all_results['dataset'] == dataset)
        ]
        if len(model_data) > 0:
            comparison_data.append({
                'Model': model_id,
                'Dataset': dataset,
                'Avg_Safe': model_data['safe'].mean(),
                'Avg_Unsafe': model_data['unsafe'].mean()
            })
comp_df = pd.DataFrame(comparison_data)

# Group by model - 4 groups total, with stacked bars for each dataset
models = ['model_A', 'model_B', 'model_C', 'model_D']
x = np.arange(len(models))
width = 0.25  # Width of each bar group (now 3 datasets)

# Prepare data for stacked bars
mnist_safe = []
mnist_unsafe = []
emnist_safe = []
emnist_unsafe = []
pixmix_safe = []
pixmix_unsafe = []

for model_id in models:
    mnist_data = comp_df[(comp_df['Model'] == model_id) & (comp_df['Dataset'] == 'MNIST')]
    emnist_data = comp_df[(comp_df['Model'] == model_id) & (comp_df['Dataset'] == 'EMNIST')]
    pixmix_data = comp_df[(comp_df['Model'] == model_id) & (comp_df['Dataset'] == 'PixMix')]
    
    if len(mnist_data) > 0:
        mnist_safe.append(mnist_data['Avg_Safe'].iloc[0])
        mnist_unsafe.append(mnist_data['Avg_Unsafe'].iloc[0])
    else:
        mnist_safe.append(0)
        mnist_unsafe.append(0)
    
    if len(emnist_data) > 0:
        emnist_safe.append(emnist_data['Avg_Safe'].iloc[0])
        emnist_unsafe.append(emnist_data['Avg_Unsafe'].iloc[0])
    else:
        emnist_safe.append(0)
        emnist_unsafe.append(0)
    
    if len(pixmix_data) > 0:
        pixmix_safe.append(pixmix_data['Avg_Safe'].iloc[0])
        pixmix_unsafe.append(pixmix_data['Avg_Unsafe'].iloc[0])
    else:
        pixmix_safe.append(0)
        pixmix_unsafe.append(0)

# Create stacked bars: within each model group, show MNIST, EMNIST, and PixMix side by side
# Each bar is stacked with Safe (bottom) and Unsafe (top)
# Use same colors but different patterns (hatching) for each dataset
safe_color = '#2ecc71'  # Green for Safe
unsafe_color = '#e74c3c'  # Red for Unsafe

# MNIST bars (stacked) - no hatching
ax2.bar(x - width, mnist_safe, width, label='MNIST Safe', alpha=0.8, color=safe_color, hatch='')
ax2.bar(x - width, mnist_unsafe, width, bottom=mnist_safe, label='MNIST Unsafe', alpha=0.8, color=unsafe_color, hatch='')

# EMNIST bars (stacked) - horizontal hatching
ax2.bar(x, emnist_safe, width, label='EMNIST Safe', alpha=0.8, color=safe_color, hatch='---')
ax2.bar(x, emnist_unsafe, width, bottom=emnist_safe, label='EMNIST Unsafe', alpha=0.8, color=unsafe_color, hatch='---')

# PixMix bars (stacked) - diagonal hatching
ax2.bar(x + width, pixmix_safe, width, label='PixMix Safe', alpha=0.8, color=safe_color, hatch='///')
ax2.bar(x + width, pixmix_unsafe, width, bottom=pixmix_safe, label='PixMix Unsafe', alpha=0.8, color=unsafe_color, hatch='///')

ax2.set_xlabel('Model', fontsize=13, fontweight='bold')
ax2.set_ylabel('Average Count', fontsize=13, fontweight='bold')
ax2.set_title('Average Safe vs Unsafe Cases by Model (Stacked)', fontsize=15, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(models, fontsize=11)
ax2.legend(fontsize=9, loc='upper left', ncol=2)
ax2.grid(True, alpha=0.3, axis='y')

# Plot 3: CRA by epsilon for each model and dataset
ax3_eps = ax3
eps_values = [1/255, 2/255, 3/255, 4/255]
colors = {'MNIST': '#3498db', 'EMNIST': '#e67e22'}
markers = {'model_A': 'o', 'model_B': 's', 'model_C': '^', 'model_D': 'D', 'model_E': 'v'}
line_styles = {'MNIST': '-', 'EMNIST': '--'}

for model_id in ['model_A', 'model_B', 'model_C', 'model_D']:
    for dataset in ['MNIST', 'EMNIST']:
        model_data = all_results[
            (all_results['model_id'] == model_id) & 
            (all_results['dataset'] == dataset)
        ]
        if len(model_data) > 0:
            cra_values = []
            for eps in eps_values:
                eps_str = f'{round(eps*255)}/255'
                eps_data = model_data[model_data['epsilon'] == eps_str]
                if len(eps_data) > 0:
                    cra_values.append(eps_data['CRA'].iloc[0])
                else:
                    cra_values.append(np.nan)
            ax3_eps.plot(eps_values, cra_values, 
                        marker=markers[model_id], 
                        color=colors[dataset],
                        linestyle=line_styles[dataset],
                        label=f'{model_id}-{dataset}',
                        linewidth=2, markersize=8, alpha=0.8)

ax3_eps.set_xlabel('Epsilon (perturbation)', fontsize=13, fontweight='bold')
ax3_eps.set_ylabel('CRA (%)', fontsize=13, fontweight='bold')
ax3_eps.set_title('CRA: MNIST vs EMNIST by Epsilon', fontsize=15, fontweight='bold')
ax3_eps.legend(fontsize=9, loc='best', ncol=2)
ax3_eps.grid(True, alpha=0.3)
ax3_eps.set_ylim(bottom=0)

# Plot 4: CRA comparison between MNIST and PixMix by epsilon
eps_values = [1/255, 2/255, 3/255, 4/255]
colors = {'MNIST': '#3498db', 'PixMix': '#e67e22'}
markers = {'model_A': 'o', 'model_B': 's', 'model_C': '^', 'model_D': 'D'}
line_styles = {'MNIST': '-', 'PixMix': '--'}

for model_id in ['model_A', 'model_B', 'model_C', 'model_D']:
    for dataset in ['MNIST', 'PixMix']:
        model_data = all_results[
            (all_results['model_id'] == model_id) & 
            (all_results['dataset'] == dataset)
        ]
        if len(model_data) > 0:
            cra_values = []
            for eps in eps_values:
                eps_str = f'{round(eps*255)}/255'
                eps_data = model_data[model_data['epsilon'] == eps_str]
                if len(eps_data) > 0:
                    cra_values.append(eps_data['CRA'].iloc[0])
                else:
                    cra_values.append(np.nan)
            ax4.plot(eps_values, cra_values, 
                    marker=markers[model_id], 
                    color=colors[dataset],
                    linestyle=line_styles[dataset],
                    label=f'{model_id}-{dataset}',
                    linewidth=2, markersize=8, alpha=0.8)

ax4.set_xlabel('Epsilon (perturbation)', fontsize=13, fontweight='bold')
ax4.set_ylabel('CRA (%)', fontsize=13, fontweight='bold')
ax4.set_title('CRA: MNIST vs PixMix by Epsilon', fontsize=15, fontweight='bold')
ax4.legend(fontsize=9, loc='best', ncol=2)
ax4.grid(True, alpha=0.3)
ax4.set_ylim(bottom=0)

plt.tight_layout()
plot_file = BASE_DIR / "final_results_plots.png"
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
print(f"\nPlots saved to {plot_file}")

# Also create a detailed breakdown table
print("\n" + "="*100)
print("Detailed Breakdown by Epsilon:")
print("="*100)

for dataset in ['MNIST', 'EMNIST', 'PixMix']:
    print(f"\n{dataset}:")
    detail_table = all_results[all_results['dataset'] == dataset][
        ['model_id', 'epsilon', 'safe', 'unsafe', 'CRA', 'AA', 'time']
    ].copy()
    detail_table = detail_table.sort_values(['model_id', 'epsilon'])
    detail_table.columns = ['Model', 'Epsilon', 'Safe', 'Unsafe', 'CRA (%)', 'AA (%)', 'Time']
    detail_table['CRA (%)'] = detail_table['CRA (%)'].round(2)
    # AA is already numeric from robustness_results.csv, just round it
    detail_table['AA (%)'] = pd.to_numeric(detail_table['AA (%)'], errors='coerce')
    detail_table['AA (%)'] = detail_table['AA (%)'].round(2)
    detail_table['Time'] = detail_table['Time'].round(4)
    print(detail_table.to_string(index=False))

print("\n" + "="*100)
print("Analysis complete!")
print("="*100)