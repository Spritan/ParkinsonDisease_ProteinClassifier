import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from rich.console import Console
from rich.table import Table
from rich.progress import track

console = Console()

class SequenceDataAnalyzer:
    def __init__(self, X, y, feature_names):
        self.X = X
        self.y = y
        self.feature_names = feature_names
        
    def analyze_class_distribution(self):
        """Analyze and visualize class distribution"""
        class_dist = pd.Series(self.y).value_counts()
        
        plt.figure(figsize=(8, 6))
        sns.barplot(x=class_dist.index, y=class_dist.values)
        plt.title('Class Distribution')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.savefig('plots/class_distribution.png')
        plt.close()
        
        # Print class distribution
        table = Table(title="Class Distribution")
        table.add_column("Class", style="cyan")
        table.add_column("Count", style="green")
        table.add_column("Percentage", style="yellow")
        
        for cls, count in class_dist.items():
            percentage = (count / len(self.y)) * 100
            table.add_row(str(cls), str(count), f"{percentage:.2f}%")
        
        console.print(table)
        
    def analyze_feature_importance(self, n_top_features=20):
        """Analyze feature importance using multiple methods"""
        results = {}
        
        # Random Forest importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(self.X, self.y)
        rf_importance = pd.Series(rf.feature_importances_, index=self.feature_names)
        
        # ANOVA F-value
        f_scores = SelectKBest(score_func=f_classif, k='all').fit(self.X, self.y)
        f_importance = pd.Series(f_scores.scores_, index=self.feature_names) # type: ignore
        
        # Mutual Information
        mi_scores = SelectKBest(score_func=mutual_info_classif, k='all').fit(self.X, self.y)
        mi_importance = pd.Series(mi_scores.scores_, index=self.feature_names) # type: ignore
        
        # Combine and rank features
        importance_df = pd.DataFrame({
            'Random Forest': rf_importance,
            'ANOVA F-score': f_importance,
            'Mutual Information': mi_importance
        })
        
        # Plot top features for each method
        fig, axes = plt.subplots(3, 1, figsize=(12, 15))
        
        for i, (method, scores) in enumerate(importance_df.items()):
            top_features = scores.nlargest(n_top_features)
            sns.barplot(x=top_features.values, y=top_features.index, ax=axes[i])
            axes[i].set_title(f'Top {n_top_features} Features - {method}')
            
        plt.tight_layout()
        plt.savefig('plots/feature_importance_comparison.png')
        plt.close()
        
        return importance_df
        
    def select_features(self, n_features=50):
        """Select top features using ensemble of methods"""
        importance_df = self.analyze_feature_importance()
        
        # Normalize scores
        normalized_scores = importance_df.apply(lambda x: (x - x.min()) / (x.max() - x.min()))
        
        # Calculate mean importance across methods
        mean_importance = normalized_scores.mean(axis=1)
        top_features = mean_importance.nlargest(n_features)
        
        # Create mask for selected features
        selected_mask = np.isin(self.feature_names, top_features.index)
        
        console.print(f"\n[bold green]Selected {n_features} top features[/bold green]")
        
        return self.X[:, selected_mask], top_features.index.tolist() 