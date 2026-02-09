import pandas as pd
import numpy as np
import glob
import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix

# Configuration
# Adjust path since script is in src/notebooks/
OUTPUT_DIR = "../../outputs"
FILENAME_PATTERN = re.compile(r"(.+)-(.+)-(.+)\.csv") # [dataset]-[model]-[split].csv

def load_all_outputs(output_dir):
    all_files = glob.glob(os.path.join(output_dir, "*.csv"))
    dfs = []
    
    print(f"Found {len(all_files)} CSV files in {output_dir}")
    
    for fpath in all_files:
        fname = os.path.basename(fpath)
        match = FILENAME_PATTERN.match(fname)
        
        if match:
            dataset_name, model_name, split_name = match.groups()
            try:
                df = pd.read_csv(fpath)
                # Verify columns
                required_cols = {'sample_id', 'prediction', 'confidence', 'ground_truth'}
                if not required_cols.issubset(df.columns):
                    print(f"Warning: Skipping {fname} - missing columns. Found: {df.columns.tolist()}")
                    continue
                    
                df['dataset'] = dataset_name
                df['model'] = model_name
                df['split'] = split_name
                df['source_file'] = fname
                dfs.append(df)
                print(f"Loaded {fname}: {len(df)} samples")
            except Exception as e:
                print(f"Error loading {fname}: {e}")
        else:
            print(f"Warning: {fname} does not match naming convention [dataset]-[model]-[split].csv")
            
    if not dfs:
        print("No valid dataframes loaded!")
        return pd.DataFrame()
        
    return pd.concat(dfs, ignore_index=True)

full_df = load_all_outputs(OUTPUT_DIR)
print("Head of Full DF:")
print(full_df.head())

if not full_df.empty:
    summary_stats = []

    # Group by Experiment identifiers
    grouped = full_df.groupby(['dataset', 'model', 'split'])

    for (dset, model, split), group in grouped:
        total = len(group)
        
        # Coverage: Predictions that are NOT -1
        valid_preds = group[group['prediction'] != -1]
        n_valid = len(valid_preds)
        coverage = n_valid / total if total > 0 else 0
        
        # Pseudo-Label Accuracy: Accuracy within covered samples
        if n_valid > 0:
            pl_acc = accuracy_score(valid_preds['ground_truth'], valid_preds['prediction'])
        else:
            pl_acc = 0.0
            
        # Overall Accuracy: Accuracy on ALL samples (treating -1 as wrong)
        overall_acc = accuracy_score(group['ground_truth'], group['prediction'])
        
        summary_stats.append({
            'Dataset': dset,
            'Model': model,
            'Split': split,
            'Total Samples': total,
            'Coverage': coverage,
            'PL Accuracy': pl_acc,
            'Overall Accuracy': overall_acc
        })
        
    stats_df = pd.DataFrame(summary_stats)
    print("\nSummary Stats:")
    print(stats_df)
else:
    print("No data to analyze.")
