from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.live import Live
from rich.layout import Layout
from rich.style import Style
from .model_evaluation import get_models, evaluate_model, plot_model_comparison, plot_metrics_heatmap

console = Console()

class SequenceClassifier:
    def __init__(self):
        self.models = get_models()
        self.results = {}
        
    def train_and_evaluate(self, X, y, test_size=0.2):
        layout = Layout()
        layout.split_column(
            Layout(Panel("[bold blue]Model Training and Evaluation[/bold blue]", 
                        style="blue"), size=3),
            Layout(name="main")
        )
        
        with Live(layout, refresh_per_second=4):
            # Split data with progress indicator
            with console.status("[bold yellow]Splitting dataset...") as status:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )
                console.print("[green]✓[/green] Dataset split completed")
            
            # Model evaluation progress
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                for name, model in self.models.items():
                    task = progress.add_task(f"[cyan]Training {name}...", total=1)
                    
                    # Cross-validation evaluation
                    self.results[name] = evaluate_model(model, X, y)
                    progress.update(task, advance=0.5)
                    
                    # Final model training and evaluation
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    self.results[name]['test_report'] = classification_report(y_test, y_pred)
                    progress.update(task, advance=0.5)
                    
                    # Display immediate results
                    self._display_model_results(name)

            # Generate comparison plots
            with console.status("[bold green]Generating performance visualizations...") as status:
                plot_model_comparison(self.results, 'accuracy')
                status.update("[bold green]Creating F1 score comparison...")
                plot_model_comparison(self.results, 'f1')
                status.update("[bold green]Generating metrics heatmap...")
                plot_metrics_heatmap(self.results)
            
            # Find and display best model
            best_model = max(self.results.items(), key=lambda x: x[1]['f1'][0])
            self.model = self.models[best_model[0]]
            
            console.print("\n[bold green]✨ Training Complete![/bold green]")
            console.print(Panel(
                f"[bold cyan]Best Performing Model:[/bold cyan] {best_model[0]}\n"
                f"[cyan]F1 Score:[/cyan] {best_model[1]['f1'][0]:.3f} ± {best_model[1]['f1'][1]:.3f}",
                title="Final Results",
                border_style="green"
            ))
            
        return {
            'model_results': self.results,
            'best_model': best_model[0]
        }
    
    def _display_model_results(self, model_name):
        metrics = self.results[model_name]
        table = Table(
            title=f"[bold]{model_name} Results[/bold]",
            show_header=True,
            header_style="bold magenta",
            title_style="bold blue"
        )
        
        table.add_column("Metric", style="cyan")
        table.add_column("Score", justify="right", style="green")
        
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            mean, std = metrics[metric]
            table.add_row(
                metric.capitalize(),
                f"{mean:.3f} ± {std:.3f}"
            )
        
        console.print(table)
        console.print(Panel(
            metrics['test_report'],
            title="[bold]Classification Report[/bold]",
            border_style="blue"
        ))
