import pandas as pd
import numpy as np
import pickle as pkl
import os
from utils.preprocessing_utils import *


def process_class_info(datafile, outfile):
    """

    Parameters
    ----------
    datafile : The excel file that contains the class labels for different TFs (provided inside data folder)
    outfile : Name to save the file

    Returns
    -------
    cls_final : The preprocessed dataframe
    """

    class_file = pd.read_excel(datafile, sheet_name=0)
    # remove the headers
    cls = class_file.iloc[23:, :]
    cls.columns = cls.iloc[0, :]
    cls = cls.iloc[1:, 1:]

    # List of columns that we are interested in
    cls_r_columns = ['TF', 'Domain', 'EMI penetration (lig147)', 'End preference', 'Periodic preference',
                     'Groove preference', 'Dyad preference',
                     'Gyre-spanning', 'Orientational preference', 'Nucleosome stability']
    cls_final = cls.loc[:, cls_r_columns]

    # processing the class labels to handle missing labels and convert string label to numeric values
    cls_final = cls_final.replace('No', 0)
    cls_final = cls_final.replace('Yes', 1)
    cls_final = cls_final.fillna(0)
    cls_final = cls_final.replace('Major', 1)
    cls_final = cls_final.replace('Stabilizer', 1)
    cls_final = cls_final.replace('Minor', 0)
    cls_final = cls_final.replace('Destabilizer', 0)
    cls_final = cls_final.replace('NA (SBP)', 0)

    # renaming the columns for easier access
    cls_final.columns = ['tf', 'domain', 'emi', 'end_preference', 'periodic_preference', 'groove_preference',
                         'dyad_preference',
                         'gyre_spanning', 'orientational_preference', 'nucleosome_stability']
    cls_final.orientational_preference = cls_final.orientational_preference.astype(int)
    cls_final.nucleosome_stability = cls_final.nucleosome_stability.astype(str).astype(int)

    #saving the file to avoid processing each time
    cls_final.to_csv(outfile, sep=",", header=True, index=False)
    return cls_final


def preprocess_raw_data(outfile, args):
    """
    Preprocess the raw class file if not already processed and parses the fasta sequences to dataframe
    Parameters
    ----------
    outfile : file name used to save
    args : argument parser object that contains the default values for parameters

    """
    class_file = 'data/rawData/class_file.csv'
    if not os.path.isfile(class_file):
        cls_final = process_class_info('data/rawData/Class_summary.xlsx', class_file)
    else:
        cls_final = pd.read_csv(class_file, sep=",", header=0)

    df = parse_sequences_txt("data/rawData/" + args.seqtype + "_sequences/",
                             "data/trainSequence_" + args.seqtype + ".csv")
    print("Training data size:", df.shape)

    test_df = parse_fasta_sequences("data/rawData/test_" + args.seqtype + ".fasta",
                                    "data/testSequence_" + args.seqtype + ".csv")
    print("Testing data size:", test_df.shape)
    data = pd.merge(df, cls_final, left_on='tf', right_on='tf')
    data = data.iloc[:, :11]
    data.columns = ['tf', 'seq', 'domain', 'emi', 'end_preference', 'periodic_preference', 'groove_preference',
                    'dyad_preference',
                    'gyre_spanning', 'orientational_preference', 'nucleosome_stability']

    data['y'] = data.apply(lambda x: list([x['end_preference'], x['periodic_preference'], x['groove_preference'],
                                           x['dyad_preference'], x['gyre_spanning'], x['orientational_preference'],
                                           x['nucleosome_stability']]), axis=1)

    df = data.iloc[:, [0, 1, 2, 3, -1]]
    df.to_csv(outfile, sep=",", index=None)


def preprocess_sequences(processedDataFile, args, test_data_file=None):
    """
    Preprocess and save the data
    Parameters
    ----------
    processedDataFile : Preprocessed data file
    args : argument parser object that contains the default values for parameters
    test_data_file : Data file that contains test sequences. If none, use a default test sequence

    """
    datafile = "data/trainSequence_" + args.seqtype + ".csv"
    if not os.path.isfile(datafile):
        preprocess_raw_data(datafile, args)

    # Define the window to consider as word
    subsequence_window_size = args.len

    # read data file that consists name, sequence, domain and the interaction type
    df = pd.read_csv(datafile, header=0)
    print(df.shape)

    if test_data_file is None:
        test_data_file = "data/testSequence_" + args.seqtype + ".csv"
    test_df = pd.read_csv(test_data_file, header=None)
    test_df.columns = ['idx', 'tf', 'seq']

    # load the embeddings of the biological words
    vectors = pd.read_csv("data/embeddings/" + str(args.len) + "grams_embeddings.txt", skiprows=1, sep=" ",
                          header=None).values
    print("There are", vectors.shape[0], "words in the sequences.")

    # define the dictionary to make access efficient later on
    print("Define a dictionary")
    embeddings = {}
    for i, vector in enumerate(vectors):
        embeddings[vector[0]] = np.array(vector[1:])

    print(len(embeddings))
    # remove the vectors variable to free the memory
    del vectors

    df.y = df.y.apply(lambda x: np.array(eval(x)))
    y = np.stack(df.y.values)

    dataset = {}
    dataset['train_tf'] = df.tf.values
    dataset['train_emb'] = df.seq.apply(lambda x: tokenizer(x, embeddings, subsequence_window_size)).values
    dataset['train_labels'] = y
    dataset['test_tf'] = test_df.tf.values
    dataset['test_emb'] = test_df.seq.apply(lambda x: tokenizer(x, embeddings, subsequence_window_size)).values

    print("Saving the processed dataset")
    with  open(processedDataFile, "wb") as filename:
        pkl.dump(dataset, filename)
