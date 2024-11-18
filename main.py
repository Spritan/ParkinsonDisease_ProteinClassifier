from src.data_processing import load_fasta_data
from src.feature_extraction import SequenceFeatureExtractor
from src.model import SequenceClassifier
from src.visualization import plot_feature_importance, plot_sequence_length_distribution

def main():
    # Load data
    data = load_fasta_data(
        'data/cleaned_output (1).fasta',
        'data/uniprotkb_parkinson_disease_protein_AND_2024_11_15.fasta'
    )
    
    # Add sequence lengths to the DataFrame
    data['length'] = data['sequence'].str.len()
    
    # Extract features
    feature_extractor = SequenceFeatureExtractor(k=3)
    X, feature_names = feature_extractor.extract_features(data['sequence'])
    y = (data['label'] == 'parkinsons').astype(int)
    
    # Train and evaluate models
    classifier = SequenceClassifier()
    results = classifier.train_and_evaluate(X, y)
    
    # Print results
    print("\nModel Evaluation Results:")
    for model_name, metrics in results['model_results'].items():
        print(f"\n{model_name}:")
        # Print metrics (mean ± std)
        for metric_name in ['accuracy', 'precision', 'recall', 'f1']:
            if metric_name in metrics:
                mean, std = metrics[metric_name]
                print(f"{metric_name}: {mean:.3f} ± {std:.3f}")
        
        # Print classification report separately
        if 'test_report' in metrics:
            print("\nClassification Report:")
            print(metrics['test_report'])
    
    print(f"\nBest performing model: {results['best_model']}")
    
    # Plot feature importance for the best model (if applicable)
    if hasattr(classifier.model, 'feature_importances_'):
        plot_feature_importance(feature_names, classifier.model.feature_importances_)
    
    plot_sequence_length_distribution(data)

if __name__ == "__main__":
    main()