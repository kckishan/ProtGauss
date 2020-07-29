import numpy as np
import re
from Bio import SeqIO
import pandas as pd
import os 

def parse_sequences(sequence_file):
    """
    Parses FASTA sequence from file
    Parameters
    ----------
    input : string
        The name of the sequence file
    Returns
    -------
    sequences : dict
        The dictionary with protein names as keys and sequences as values
    """
    with open(sequence_file, 'r') as f:
        lines = f.readlines()
        sequence_list = []
        for line in lines:
            sequence_list.append(line.replace("  ", " ").split())

    sequences = {}
    i = 0
    for row in lines:
        if ">" in row:
            protein = row.rstrip("\n\r").lstrip(">")
        else:
            sequence = row.rstrip("\n\r")
        if i > 0 and i % 2 != 0: 
            sequences[protein] = sequence
        i += 1
    return sequences

def parse_sequences_txt(directory, outfile):
    print("Loading sequences from ", directory)
    files = os.listdir(directory)

    sequences = {}
    for file in files:
        seq = parse_sequences(directory + file)
        sequences.update(seq)

    # writing the dictionary to file
    df = pd.DataFrame.from_dict(sequences, orient = 'index')
    df = df.reset_index()
    print(df.head())
    df.columns = ['tf', 'seq']
    
    return df


def parse_fasta_sequences(sequence_file, outfile):
    """
    Parses FASTA sequence from file
    Parameters
    ----------
    input : string
        The name of the sequence file
    Returns
    -------
    sequences : dict
        The dictionary with protein names as keys and sequences as values
    """
    seq = []
    tf = []
    for record in SeqIO.parse(sequence_file, "fasta"):
        id = record.id.split("|")[-1]
        tf.append(id.split("_")[0])
        seq.append(str(record.seq))

    df = pd.DataFrame()
    df['tf'] = tf
    df['seq'] = seq
    df.insert(0, 'idx', range(len(df)))

    df.to_csv(outfile, header = False, index=False)
    return df

def pad_embeddings(embs, seq_length, d_emb):
    emb = np.zeros((seq_length, d_emb))
    if embs.shape[0] > seq_length:
        emb = embs[:seq_length,:] 
    else:
        emb[:embs.shape[0],:] = embs
    return emb


def generate_subsequences(sequence, embeddings, n, agg=False):
    # Convert to lowercases
    sequence = sequence.lower()

    # Replace all none alphanumeric characters with spaces
    sequence = re.sub(r'[^a-zA-Z0-9\s]', ' ', sequence)
    
    # Break sentence in the token, remove empty tokens
    tokens = [token for token in sequence.split(" ") if token != ""]

    # Use the zip function to help us generate n-grams
    # Concatentate the tokens into ngrams and return
    ngrams = zip(*[tokens[i:] for i in range(n)])
    ngrams = ["".join(ngram) for ngram in ngrams]
    embs = []
    for ngram in ngrams:
        # print(ngram, embeddings.get(ngram))
        if ngram in embeddings.keys():
            emb = embeddings[ngram]
            if len(emb) > 0:
                embs.append(emb)
        else:
            continue
    if agg:
        return np.mean(np.vstack(embs), axis=0)
    else:
        return np.vstack(embs)


def tokenizer(sequence, embeddings, subsequence_window_size):
    n_seq =" ".join(sequence)
    seq = re.sub(
          r"[\*\"“”\n\\…\+\-\/\=\(\)‘•:\[\]\|’\!;]", "", str(n_seq))
    return generate_subsequences(seq, embeddings, n=subsequence_window_size)


