import random
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
import torch

console = Console()

class SequenceClassifier:
    def __init__(self, input_shape=None, random_seed=42):
        self.random_seed = random_seed
        # Set seeds for all relevant libraries
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_seed)
        
        self.models = get_models(input_shape)
        self.results = {}
        
    def train_and_evaluate(self, X, y, test_size=0.2):
        console.print("[bold blue]Model Training and Evaluation[/bold blue]")
        
        # Split data with progress indicator and consistent seed
        with console.status("[bold yellow]Splitting dataset...") as status:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.random_seed
            )
            console.print("[green]✓[/green] Dataset split completed")
        
        # Model evaluation progress
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            tasks = {
                name: progress.add_task(f"[cyan]Training {name}...", total=1)
                for name in self.models.keys()
            }
            
            for name, model in self.models.items():
                # Cross-validation evaluation
                self.results[name] = evaluate_model(model, X, y)
                progress.update(tasks[name], advance=0.5)
                
                # Final model training and evaluation
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                self.results[name]['test_report'] = classification_report(y_test, y_pred)
                progress.update(tasks[name], completed=True)
                
                # Display immediate results
                self._display_model_results(name)
        
        # Generate comparison plots
        console.print("[bold green]Generating performance visualizations...")
        plot_model_comparison(self.results, 'accuracy')
        console.print("[bold green]Creating F1 score comparison...")
        plot_model_comparison(self.results, 'f1')
        console.print("[bold green]Generating metrics heatmap...")
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
