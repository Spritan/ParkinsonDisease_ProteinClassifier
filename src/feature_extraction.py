from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
from rich.progress import track
from rich.console import Console

console = Console()

class SequenceFeatureExtractor:
    def __init__(self, k=3):
        self.k = k
        self.vectorizer = CountVectorizer(analyzer='char', ngram_range=(k, k))
        
    def compute_basic_features(self, sequence):
        return {
            'length': len(sequence),
            'gc_content': (sequence.count('G') + sequence.count('C')) / len(sequence),
            'hydrophobic_ratio': sum(aa in 'AILMFWYV' for aa in sequence) / len(sequence),
            'charged_ratio': sum(aa in 'DEKR' for aa in sequence) / len(sequence)
        }
    
    def extract_features(self, sequences):
        with console.status("[bold yellow]Extracting features...") as status:
            # K-mer features
            status.update("[cyan]Computing k-mer features...")
            kmer_features = self.vectorizer.fit_transform(sequences)
            
            # Basic features
            status.update("[cyan]Computing basic sequence features...")
            basic_features = pd.DataFrame([
                self.compute_basic_features(seq) for seq in track(sequences, description="Processing sequences")
            ])
            
            # Combine features
            status.update("[cyan]Combining feature matrices...")
            combined_features = np.hstack([
                kmer_features.toarray(), # type: ignore
                basic_features.values
            ])
            
            feature_names = (
                self.vectorizer.get_feature_names_out().tolist() +
                basic_features.columns.tolist()
            )
            
            console.print(f"[green]✓[/green] Extracted {len(feature_names)} features")
            
        return combined_features, feature_names
