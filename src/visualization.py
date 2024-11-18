import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
def plot_feature_importance(feature_names, importance_scores, top_n=20):
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance_scores
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=importance_df.head(top_n), x='importance', y='feature')
    plt.title(f'Top {top_n} Most Important Features')
    plt.tight_layout()
    plt.show()

def plot_sequence_length_distribution(data):
    plt.figure(figsize=(10, 6))
    sns.histplot(data=data, x='length', hue='label', bins=50)
    plt.title('Sequence Length Distribution')
    plt.xlabel('Sequence Length')
    plt.ylabel('Count')
    plt.show()