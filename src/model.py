from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
from .model_evaluation import get_models, evaluate_model, plot_model_comparison, plot_metrics_heatmap

class SequenceClassifier:
    def __init__(self):
        self.models = get_models()
        self.results = {}
        
    def train_and_evaluate(self, X, y, test_size=0.2):
        # Split data for final evaluation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Evaluate all models
        for name, model in self.models.items():
            print(f"\nEvaluating {name}...")
            self.results[name] = evaluate_model(model, X, y)
            
            # Train on full training set and evaluate on test set
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            self.results[name]['test_report'] = classification_report(y_test, y_pred)
        
        # Create comparison plots
        plot_model_comparison(self.results, 'accuracy')
        plot_model_comparison(self.results, 'f1')
        plot_metrics_heatmap(self.results)
        
        # Find best model
        best_model = max(self.results.items(), 
                        key=lambda x: x[1]['f1'][0])
        self.model = self.models[best_model[0]]
        
        return {
            'model_results': self.results,
            'best_model': best_model[0]
        }
