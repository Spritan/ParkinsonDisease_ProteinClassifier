from Bio import SeqIO
import pandas as pd

def load_fasta_data(healthy_path, parkinsons_path):
    def read_fasta(file_path, label):
        sequences = []
        for record in SeqIO.parse(file_path, "fasta"):
            sequences.append({
                'sequence': str(record.seq),
                'id': record.id,
                'label': label
            })
        return sequences
    
    healthy_sequences = read_fasta(healthy_path, 'healthy')
    parkinsons_sequences = read_fasta(parkinsons_path, 'parkinsons')
    
    return pd.DataFrame(healthy_sequences + parkinsons_sequences)
