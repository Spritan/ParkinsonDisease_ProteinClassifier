from Bio import SeqIO
import pandas as pd
from rich.progress import track
from rich.console import Console

console = Console()

def load_fasta_data(healthy_path, parkinsons_path):
    def read_fasta(file_path, label):
        sequences = []
        records = list(SeqIO.parse(file_path, "fasta"))
        
        for record in track(records, description=f"[cyan]Processing {label} sequences..."):
            sequences.append({
                'sequence': str(record.seq),
                'id': record.id,
                'label': label
            })
            
        console.print(f"[green]✓[/green] Loaded {len(sequences)} {label} sequences")
        return sequences
    
    with console.status("[bold blue]Loading FASTA files...") as status:
        healthy_sequences = read_fasta(healthy_path, 'healthy')
        parkinsons_sequences = read_fasta(parkinsons_path, 'parkinsons')
        
        df = pd.DataFrame(healthy_sequences + parkinsons_sequences)
        console.print(f"[bold green]✓[/bold green] Total sequences loaded: {len(df)}")
        
    return df
