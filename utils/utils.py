def create_matrices(emb):
    seq_matrices = []
    for i, M in enumerate(emb):
        seq_matrices.append(M.astype(float))

    return seq_matrices