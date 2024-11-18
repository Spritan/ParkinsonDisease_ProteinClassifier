from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

def get_models():
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42),
        'XGBoost': XGBClassifier(random_state=42)
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
    
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        
        scores['accuracy'].append(accuracy_score(y_val, y_pred))
        scores['precision'].append(precision_score(y_val, y_pred))
        scores['recall'].append(recall_score(y_val, y_pred))
        scores['f1'].append(f1_score(y_val, y_pred))
    
    return {metric: (np.mean(values), np.std(values)) 
            for metric, values in scores.items()}

def plot_model_comparison(results, metric='accuracy'):
    plt.figure(figsize=(12, 6))
    models = list(results.keys())
    means = [results[model][metric][0] for model in models]
    stds = [results[model][metric][1] for model in models]
    
    # Create bar plot manually instead of using seaborn
    x = np.arange(len(models))
    plt.bar(x, means, yerr=stds, capsize=5)
    plt.xticks(x, models, rotation=45)
    plt.ylabel(metric.capitalize())
    plt.title(f'Model Comparison - {metric.capitalize()}')
    
    # Add value labels on top of bars
    for i, v in enumerate(means):
        plt.text(i, v + stds[i], f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f'plots/model_comparison_{metric}.png')
    plt.close()

def plot_metrics_heatmap(results):
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    models = list(results.keys())
    
    # Create data matrix for heatmap
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