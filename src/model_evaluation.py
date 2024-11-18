import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from rich.console import Console
from rich.table import Table
from rich.progress import track, Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from src.models.lstm import PyTorchLSTMClassifier
console = Console()

def get_models(input_shape=None):
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'KNN': KNeighborsClassifier(),
        'Neural Network': MLPClassifier(random_state=42),
        'XGBoost': XGBClassifier(random_state=42),
        'LSTM': PyTorchLSTMClassifier(input_shape=input_shape)
    }
    return models

def evaluate_model(model, X, y, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': []
    }
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True
    ) as progress:
        task = progress.add_task("[cyan]Running cross-validation...", total=n_splits)
        
        for train_idx, val_idx in kf.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            
            scores['accuracy'].append(accuracy_score(y_val, y_pred))
            scores['precision'].append(precision_score(y_val, y_pred))
            scores['recall'].append(recall_score(y_val, y_pred))
            scores['f1'].append(f1_score(y_val, y_pred))
            
            progress.advance(task)
    
    return {metric: (np.mean(values), np.std(values)) 
            for metric, values in scores.items()}

def plot_model_comparison(results, metric='accuracy'):
    with console.status(f"[cyan]Generating {metric} comparison plot..."):
        # Create new figure for each plot
        plt.clf()
        fig = plt.figure(figsize=(12, 6))
        
        models = list(results.keys())
        means = [results[model][metric][0] for model in models]
        stds = [results[model][metric][1] for model in models]
        
        x = np.arange(len(models))
        plt.bar(x, means, yerr=stds, capsize=5)
        plt.xticks(x, models, rotation=45)
        plt.ylabel(metric.capitalize())
        plt.title(f'Model Comparison - {metric.capitalize()}')
        
        for i, v in enumerate(means):
            plt.text(i, v + stds[i], f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'plots/model_comparison_{metric}.png')
        plt.close(fig)
        
    console.print(f"[green]✓[/green] Saved comparison plot for {metric}")

def plot_metrics_heatmap(results):
    with console.status("[cyan]Generating metrics heatmap..."):
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        models = list(results.keys())
        
        data = np.zeros((len(models), len(metrics)))
        for i, model in enumerate(models):
            for j, metric in enumerate(metrics):
                data[i, j] = results[model][metric][0]
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(data, annot=True, fmt='.3f', 
                    xticklabels=metrics, 
                    yticklabels=models,
                    cmap='YlOrRd')
        plt.title('Model Performance Metrics')
        plt.tight_layout()
        plt.savefig('plots/metrics_heatmap.png')
        plt.close()
        
    console.print("[green]✓[/green] Saved metrics heatmap")

def display_model_results(results):
    table = Table(title="Model Performance Summary", show_header=True, header_style="bold magenta")
    table.add_column("Model", style="cyan")
    table.add_column("Accuracy", justify="right", style="green")
    table.add_column("Precision", justify="right", style="green")
    table.add_column("Recall", justify="right", style="green")
    table.add_column("F1 Score", justify="right", style="green")
    
    for model_name, metrics in results.items():
        table.add_row(
            model_name,
            f"{metrics['accuracy'][0]:.3f} ± {metrics['accuracy'][1]:.3f}",
            f"{metrics['precision'][0]:.3f} ± {metrics['precision'][1]:.3f}",
            f"{metrics['recall'][0]:.3f} ± {metrics['recall'][1]:.3f}",
            f"{metrics['f1'][0]:.3f} ± {metrics['f1'][1]:.3f}"
        )
    
    console.print(table) 