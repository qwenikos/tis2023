import numpy as np

filename = 'vectors.txt'

f = open(filename, 'r')

lines = f.readlines()
coocurence_matrix = {}

for line in lines:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')

    print(coefs.shape)
    
    coocurence_matrix[word] = coefs

    emb_vectors = coocurence_matrix.values()
    first_embvec = next(iter(emb_vectors))

    embvec_size = first_embvec.shape[0]
