import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

# Set premium aesthetic
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['axes.titlepad'] = 20
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 12

def save_plot(name, out_dir):
    plt.tight_layout()
    path = out_dir / f"{name}.png"
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {path}")
    plt.close()

def plot_accuracy_vs_latency(df, out_dir):
    plt.figure(figsize=(12, 7))
    # Filter for 2-class and 4-class experiments
    for nc in df['num_classes'].unique():
        subset = df[df['num_classes'] == nc]
        sns.scatterplot(
            data=subset, 
            x='latency_ms', 
            y='accuracy', 
            hue='classifier', 
            style='stage', 
            s=150, 
            alpha=0.8
        )
    
    plt.title("Accuracy vs. Inference Latency (Efficiency Trade-off)")
    plt.xlabel("Latency (ms per sample)")
    plt.ylabel("Accuracy")
    plt.xscale('log') # Latency often spans orders of magnitude
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    save_plot("accuracy_vs_latency", out_dir)

def plot_performance_comparison(df, out_dir):
    # Prepare data for grouped bar chart
    metrics = ['accuracy', 'f1_macro']
    df_melted = df.melt(
        id_vars=['classifier', 'stage', 'num_classes'], 
        value_vars=metrics, 
        var_name='Metric', 
        value_name='Value'
    )
    
    for nc in df['num_classes'].unique():
        plt.figure(figsize=(14, 8))
        subset = df_melted[df_melted['num_classes'] == nc]
        sns.barplot(
            data=subset, 
            x='classifier', 
            y='Value', 
            hue='Metric', 
            palette='viridis'
        )
        plt.title(f"Classifier Performance Comparison ({nc}-class)")
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 1.0)
        plt.legend(title="Metric")
        save_plot(f"performance_{nc}class", out_dir)

def plot_complexity(df, out_dir):
    # Model size and Parameter count
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Filter for unique classifiers (complexity is same across experiments usually)
    unique_df = df.drop_duplicates(subset=['classifier'])
    
    sns.barplot(data=unique_df, x='classifier', y='model_size_mb', ax=ax1, palette='magma')
    ax1.set_title("Model Size (MB)")
    ax1.tick_params(axis='x', rotation=45)
    
    sns.barplot(data=unique_df, x='classifier', y='params', ax=ax2, palette='inferno')
    ax2.set_title("Parameter Count")
    ax2.tick_params(axis='x', rotation=45)
    
    plt.suptitle("Computational Complexity Analysis", fontsize=20)
    save_plot("complexity_analysis", out_dir)

def main():
    script_dir = Path(__file__).parent
    results_csv = script_dir / 'outputs' / 'experiment_results.csv'
    
    if not results_csv.exists():
        print(f"Error: {results_csv} not found.")
        return

    df = pd.read_csv(results_csv)
    
    out_dir = script_dir / 'outputs' / 'plots'
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating plots from {results_csv}...")
    
    plot_accuracy_vs_latency(df, out_dir)
    plot_performance_comparison(df, out_dir)
    plot_complexity(df, out_dir)
    
    print("\nAll plots generated successfully!")

if __name__ == "__main__":
    main()
