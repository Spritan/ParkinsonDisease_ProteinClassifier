import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from collections import Counter
from rich.console import Console

console = Console()

def create_plots_directory():
    import os
    if not os.path.exists('plots'):
        os.makedirs('plots')

def plot_amino_acid_distribution(sequences, labels):
    """Plot amino acid frequency distribution for each class"""
    plt.figure(figsize=(15, 6))
    
    # Count amino acids for each class
    aa_counts = {}
    for seq, label in zip(sequences, labels):
        if label not in aa_counts:
            aa_counts[label] = Counter()
        aa_counts[label].update(seq)
    
    # Convert to percentages
    for label in aa_counts:
        total = sum(aa_counts[label].values())
        aa_counts[label] = {k: (v/total)*100 for k, v in aa_counts[label].items()}
    
    # Plot
    x = list(set.union(*[set(counts.keys()) for counts in aa_counts.values()]))
    width = 0.35
    
    for i, (label, counts) in enumerate(aa_counts.items()):
        plt.bar([j + i*width for j in range(len(x))],
                [counts.get(aa, 0) for aa in x],
                width,
                label=f'Class {label}')
    
    plt.xlabel('Amino Acids')
    plt.ylabel('Frequency (%)')
    plt.title('Amino Acid Distribution by Class')
    plt.xticks(range(len(x)), x)
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/amino_acid_distribution.png')
    plt.close()

def plot_sequence_properties(sequences):
    """Plot various sequence properties"""
    properties = {
        'length': [len(seq) for seq in sequences],
        'gc_content': [(seq.count('G') + seq.count('C'))/len(seq) for seq in sequences],
        'hydrophobic': [sum(aa in 'AILMFWYV' for aa in seq)/len(seq) for seq in sequences],
        'charged': [sum(aa in 'DEKR' for aa in seq)/len(seq) for seq in sequences]
    }
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Sequence Properties Distribution')
    
    for (prop, values), ax in zip(properties.items(), axes.flat):
        sns.histplot(values, ax=ax)
        ax.set_title(f'{prop.replace("_", " ").title()} Distribution')
        ax.set_xlabel(prop)
        ax.set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig('plots/sequence_properties.png')
    plt.close()

def plot_kmer_frequency(sequences, k=2):
    """Plot top k-mer frequencies"""
    from itertools import product
    
    # Generate all possible k-mers
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    all_kmers = [''.join(p) for p in product(amino_acids, repeat=k)]
    
    # Count k-mers
    kmer_counts = Counter()
    for seq in sequences:
        for i in range(len(seq) - k + 1):
            kmer = seq[i:i+k]
            if set(kmer).issubset(set(amino_acids)):
                kmer_counts[kmer] += 1
    
    # Plot top 20 k-mers
    plt.figure(figsize=(15, 6))
    top_kmers = dict(sorted(kmer_counts.items(), key=lambda x: x[1], reverse=True)[:20])
    
    plt.bar(range(len(top_kmers)), list(top_kmers.values()))
    plt.xticks(range(len(top_kmers)), list(top_kmers.keys()), rotation=45)
    plt.title(f'Top {len(top_kmers)} {k}-mer Frequencies')
    plt.xlabel(f'{k}-mers')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(f'plots/kmer_frequency_{k}.png')
    plt.close()

def plot_feature_correlations(X, feature_names):
    """Plot feature correlation heatmap"""
    # Select a subset of features if there are too many
    if len(feature_names) > 50:
        # Select top 50 features by variance
        variances = np.var(X, axis=0)
        top_indices = np.argsort(variances)[-50:]
        X = X[:, top_indices]
        feature_names = [feature_names[i] for i in top_indices]
    
    corr_matrix = np.corrcoef(X.T)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, 
                xticklabels=feature_names,
                yticklabels=feature_names,
                cmap='coolwarm',
                center=0)
    plt.title('Feature Correlation Heatmap')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('plots/feature_correlations.png')
    plt.close()

def plot_feature_importance(feature_names, importance_scores, top_n=20):
    """Plot feature importance scores
    
    Args:
        feature_names: List of feature names
        importance_scores: Array of importance scores
        top_n: Number of top features to display
    """
    # Ensure arrays are the same length
    if len(feature_names) != len(importance_scores):
        feature_names = feature_names[:len(importance_scores)]
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance_scores
    }).sort_values('importance', ascending=False)
    
    # Select top N features
    importance_df = importance_df.head(top_n)
    
    # Create plot
    plt.figure(figsize=(12, 6))
    sns.barplot(data=importance_df, x='importance', y='feature')
    plt.title(f'Top {top_n} Most Important Features')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig('plots/feature_importance.png')
    plt.close()
    
    console.print(f"[green]✓[/green] Generated feature importance plot")

def plot_sequence_length_distribution(data):
    plt.figure(figsize=(10, 6))
    sns.histplot(data=data, x='length', hue='label', bins=50)
    plt.title('Sequence Length Distribution')
    plt.xlabel('Sequence Length')
    plt.ylabel('Count')
    plt.show()