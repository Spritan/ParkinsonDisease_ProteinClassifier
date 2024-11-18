from src.data_processing import load_fasta_data
from src.feature_extraction import SequenceFeatureExtractor
from src.model import SequenceClassifier
from src.visualization import plot_feature_importance, plot_sequence_length_distribution, create_plots_directory, plot_amino_acid_distribution, plot_sequence_properties, plot_kmer_frequency, plot_feature_correlations
from src.data_analysis import SequenceDataAnalyzer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track

console = Console()

def main():
    console.print("[bold blue]Starting Classification Pipeline[/bold blue]\n")
    
    # Create plots directory
    create_plots_directory()
    
    with console.status("[bold green]Loading and processing data...") as status:
        data = load_fasta_data(
            'data/cleaned_output (1).fasta',
            'data/uniprotkb_parkinson_disease_protein_AND_2024_11_15.fasta'
        )
        
        # Generate EDA plots
        console.print("[bold blue]Generating EDA plots...[/bold blue]")
        
        # Plot sequence properties
        plot_sequence_properties(data['sequence'])
        console.print("[green]✓[/green] Generated sequence properties plots")
        
        # Plot amino acid distribution
        plot_amino_acid_distribution(data['sequence'], data['label'])
        console.print("[green]✓[/green] Generated amino acid distribution plot")
        
        # Plot k-mer frequencies
        plot_kmer_frequency(data['sequence'], k=2)
        plot_kmer_frequency(data['sequence'], k=3)
        console.print("[green]✓[/green] Generated k-mer frequency plots")
        
        # Extract features and plot correlations
        feature_extractor = SequenceFeatureExtractor(k=3)
        X, feature_names = feature_extractor.extract_features(data['sequence'])
        y = (data['label'] == 'parkinsons').astype(int)
        
        plot_feature_correlations(X, feature_names)
        console.print("[green]✓[/green] Generated feature correlation plot")
        
        # Continue with existing analysis and model training
        analyzer = SequenceDataAnalyzer(X, y, feature_names)
        analyzer.analyze_class_distribution()
        X_selected, selected_features = analyzer.select_features(n_features=50)
        
        console.print(f"\n[bold green]Selected {len(selected_features)} features for model training[/bold green]")
        
        status.update("[bold green]Training and evaluating models...")
        classifier = SequenceClassifier()
        results = classifier.train_and_evaluate(X_selected, y)

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
    
    # Plot feature importance using selected features
    if hasattr(classifier.model, 'feature_importances_'):
        plot_feature_importance(selected_features, classifier.model.feature_importances_)

if __name__ == "__main__":
    main()