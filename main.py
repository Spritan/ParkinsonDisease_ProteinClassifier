from src.data_processing import load_fasta_data
from src.feature_extraction import SequenceFeatureExtractor
from src.model import SequenceClassifier
from src.visualization import plot_feature_importance, plot_sequence_length_distribution
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track

console = Console()

def main():
    # Create a loading spinner
    with console.status("[bold green]Loading data...") as status:
        data = load_fasta_data(
            'data/cleaned_output (1).fasta',
            'data/uniprotkb_parkinson_disease_protein_AND_2024_11_15.fasta'
        )
        status.update("[bold green]Processing sequences...")
        data['length'] = data['sequence'].str.len()
        
        # Extract features
        feature_extractor = SequenceFeatureExtractor(k=3)
        X, feature_names = feature_extractor.extract_features(data['sequence'])
        y = (data['label'] == 'parkinsons').astype(int)
        
        status.update("[bold green]Training and evaluating models...")
        classifier = SequenceClassifier()
        results = classifier.train_and_evaluate(X, y)

    # Print results using rich formatting
    console.print("\n[bold cyan]Model Evaluation Results[/bold cyan]")
    
    # Create a table for metrics
    for model_name, metrics in results['model_results'].items():
        table = Table(title=f"\n[bold]{model_name} Metrics[/bold]", 
                     show_header=True, 
                     header_style="bold magenta")
        
        table.add_column("Metric", style="cyan")
        table.add_column("Score (mean ± std)", justify="right", style="green")
        
        for metric_name in ['accuracy', 'precision', 'recall', 'f1']:
            if metric_name in metrics:
                mean, std = metrics[metric_name]
                table.add_row(
                    metric_name.capitalize(),
                    f"{mean:.3f} ± {std:.3f}"
                )
        
        console.print(table)
        
        # Print classification report in a panel
        if 'test_report' in metrics:
            console.print(Panel(
                metrics['test_report'],
                title="[bold]Classification Report[/bold]",
                border_style="blue"
            ))
    
    console.print(f"\n[bold green]Best performing model:[/bold green] [yellow]{results['best_model']}[/yellow]")
    
    # Plot feature importance if available
    if hasattr(classifier.model, 'feature_importances_'):
        with console.status("[bold green]Generating feature importance plot..."):
            plot_feature_importance(feature_names, classifier.model.feature_importances_)

if __name__ == "__main__":
    main()