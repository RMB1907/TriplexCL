from Bio import SeqIO
import numpy as np

def load_data(posf, negf):
    
    # Load positive sequences
    pos_records = list(SeqIO.parse(posf, "fasta"))
    pos_seqs = [str(r.seq) for r in pos_records]
    pos_names = [r.id for r in pos_records]
    pos_labels = np.ones(len(pos_seqs))
    
    # Load negative sequences
    neg_records = list(SeqIO.parse(negf, "fasta"))
    neg_seqs = [str(r.seq) for r in neg_records]
    neg_names = [r.id for r in neg_records]
    neg_labels = np.zeros(len(neg_seqs))
    
    # Combine
    sequences = pos_seqs + neg_seqs
    names = pos_names + neg_names
    labels = np.concatenate([pos_labels, neg_labels])
    
    return sequences, labels
