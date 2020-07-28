from biovec.models import ProtVec

def run_biovec(datafile, args):
    model = ProtVec(datafile, corpus_fname="corpus/corpus_"+str(args.len)+".txt", n=args.len)
    model.wv.save_word2vec_format("data/embeddings/"+str(args.len)+"grams_embeddings.txt")