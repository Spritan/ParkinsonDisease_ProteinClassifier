from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd

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
        # K-mer features
        kmer_features = self.vectorizer.fit_transform(sequences)
        
        # Basic features
        basic_features = pd.DataFrame([
            self.compute_basic_features(seq) for seq in sequences
        ])
        
        # Combine features
        combined_features = np.hstack([
            kmer_features.toarray(),
            basic_features.values
        ])
        
        feature_names = (
            self.vectorizer.get_feature_names_out().tolist() +
            basic_features.columns.tolist()
        )
        
        return combined_features, feature_names
