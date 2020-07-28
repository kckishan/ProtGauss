import argparse
from biovec.models import ProtVec

def argument_parser():
    """
    Parses the argument to run the model.
    Returns
    -------
    model parameters: ArgumentParser Namespace
        The ArgumentParser Namespace that contains the model parameters.
    """
    parser = argparse.ArgumentParser(description="Run biovec model")
    parser.add_argument("--len", type=int, required=True,
                        help="Length of the Biological words.")
    return parser.parse_args()

args = argument_parser()

print("Generating embeddings for subsequences of length "+str(args.len)+".")

model = ProtVec("data/fasta/uniprot_sprot.fasta", corpus_fname="output_corpusfile_path.txt", n=args.len)
model.wv.save_word2vec_format("data/embeddings/"+str(args.len)+"grams_embeddings.txt")