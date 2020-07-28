import argparse
import os
import numpy as np
from os.path import isfile
from preprocessing import preprocess_sequences
from run_biovec import run_biovec
from model import run_model
import warnings;

warnings.filterwarnings("ignore")
from texttable import Texttable
from utils.preprocessing_utils import *


def table_printer(args):
    """
    Print the parameters of the model in a Tabular format
    Parameters
    ---------
    args: argparser object
        The parameters used for the model
    """
    args = vars(args)
    keys = sorted(args.keys())
    table = Texttable()
    table.add_rows([["Parameter", "Value"]] +
                   [[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    print(table.draw())


def argument_parser():
    """
    Parses the argument to run the model.
    Returns
    -------
    model parameters: ArgumentParser Namespace
        The ArgumentParser Namespace that contains the model parameters.
    """
    parser = argparse.ArgumentParser(description="Run model")
    parser.add_argument("--seqtype", type=str, default="full",
                        help="Choose: eDBD or full")
    parser.add_argument("--len", type=int, default=3,
                        help="Length of the Biological words.")
    parser.add_argument("--binding", type=int, default=0,
                        help="binding types")
    return parser.parse_args()


def main(args):
    # Unitprot fasta sequence to train ProtVec model
    fastaFile = "data/fasta/uniprot_sprot.fasta"
    processedDataFile = "data/processedData/" + str(args.len) + "seq_embeddings_" + args.seqtype + ".pkl"

    # if biovec is not trained, train the model. Otherwise, load the preprocessed data
    if not isfile(processedDataFile):
        if not isfile("data/embeddings/" + str(args.len) + "grams_embeddings.txt"):
            print("Generating embeddings for subsequences of length " + str(args.len) + ".")
            run_biovec(fastaFile, args)
        else:
            print("Using pre-generated embeddings")

    organism = ["eDBD_195"]
    for org in organism:
        print(org)
        folder_name = "data/" + org + "/"
        if not os.path.exists(folder_name):
            os.makedirs((folder_name))

        test_fasta = folder_name + "seq.fasta"
        test_output = folder_name + "testSequence_" + args.seqtype + ".csv"
        test_df = parse_fasta_sequences(test_fasta, test_output)
        preprocess_sequences(processedDataFile, args, test_data_file=test_output)

        print("Running model")
        classes = ['end_preference', 'periodic_preference', 'groove_preference', 'dyad_preference', 'gyre_spanning',
                   'orientational_preference', 'nucleosome_stability']
        if args.binding == -1:
            selected_classes = np.arange(len(classes))
        else:
            selected_classes = np.asarray([args.binding])
        print("Training for ", list(map(classes.__getitem__, selected_classes)))

        run_model(org, folder_name, processedDataFile, args, classes, selected_classes)


if __name__ == "__main__":
    args = argument_parser()
    table_printer(args)
    main(args)
