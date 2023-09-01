import numpy as np


def np_generate_all_kmers_one_hot(k):
    nucleotides = ['A', 'T', 'C', 'G']
    kmers = ['']
    
    for a in range(k):
        # print (a)
        new_kmers = []
        for kmer in kmers:
            for nt in nucleotides:
                new_kmers.append(kmer + nt)
        kmers = new_kmers

    kmer_to_index = {kmer: i for i, kmer in enumerate(kmers)}
    num_kmers = len(kmers)
    kmersOnHotDict={}
    one_hot_enc_kmers = np.zeros((num_kmers, num_kmers), dtype=int)
    for i, kmer in enumerate(kmers):
        one_hot_enc_kmers[i, kmer_to_index[kmer]] = 1
        kmersOnHotDict[kmer]=one_hot_enc_kmers


    return kmersOnHotDict,kmers,one_hot_enc_kmers


k =4   # Change k to your desired k-mer length
kmersOnHotDict,kmers,one_hot_enc_kmers=np_generate_all_kmers(k)
print(kmersOnHotDict['AAAA'])
print(np.shape(one_hot_enc_kmers))


