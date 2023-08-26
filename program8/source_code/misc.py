import numpy as np
import sys 
import keras as kr
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from itertools import product
import random as rn

np.random.seed(2000)

# Necessary for starting core Python generated random numbers in a well-defined state.
rn.seed(2023)


def read_fasta_file (input_file, num_samples=0):
  f = open(input_file,'r')

  lines = f.readlines()

  genes = set()

  for line in lines:
    if line[0] == '>':
      continue 
    else:
      flag = 0
      for l in line:
        if l == 'N' or l == 'n':
          flag = 1
          break
      
      if flag == 1:
        continue

      genes.add(line)

  returned_genes = []

  for gene in genes:
    returned_genes.append(gene)
  
  if num_samples != 0:
    returned_genes = rn.sample(returned_genes, num_samples)
  
  f.close()

  return returned_genes



# removes the whitespaces characters at the beggining and 
# at the end of a sequence or skips a sequence which 
# contains the 'N' character
def clean_sequences(sequences, window_size):
  clean_seqs = np.zeros(shape = (sequences.shape[0], sequences.shape[1]), dtype=str)

  col_position=0

  for sequence in sequences.T:
    flag = 0

    for (row_position, row) in enumerate(sequence):
      if row == '\n' or row == '\t' or row == ' ':                                 
        clean_seqs = np.delete(clean_seqs, (col_position), axis=1)

        flag = 1
        break
      
      clean_seqs[row_position, col_position] = row[0].upper()

    if flag == 1:
      continue
        
    col_position += 1
  

  return clean_seqs


# converts a nucleotide into a 4-bit one-hot vector
def one_hot_conversion(nucleotide):
  one_hot_map = {
		"A": np.array([1, 0, 0, 0],dtype=np.float32), 
		"C": np.array([0, 1, 0, 0],dtype=np.float32), 
		"G": np.array([0, 0, 1, 0],dtype=np.float32), 
		"T": np.array([0, 0, 0, 1],dtype=np.float32),
    "": np.array([0, 0, 0, 0],dtype=np.float32)}

  return one_hot_map[nucleotide]


# Question: is it okay that in some samples there are zero columns in the end of the samples?

# converts the sequences into one-hot encoding sequences
def one_hot_encoding (sequences):

  one_hot_samples = np.zeros(shape = (sequences.shape[1], 4, sequences.shape[0]), dtype=np.float32)

  col_position=0

  for (i, sequence) in enumerate(sequences.T):
    for (col_position, nucleotide) in enumerate(sequence):
      # print(type(nucleotide), len(nucleotide), nucleotide, nucleotide[0])
      # exit()

      encoded_nuc = one_hot_conversion(nucleotide[0])

      for (row_position, one_hot) in enumerate(encoded_nuc):
        one_hot_samples[i, row_position, col_position] = one_hot


  return one_hot_samples
 

  # for (row_position, sequence) in enumerate(sequences):

  #   for (col_position, column) in enumerate(sequence):

  #     column = str(column)
  #     res = one_hot_conversion(column)
  #     encoded_column = res
  #     one_hot_samples[row_position, col_position, :] = encoded_column

  # return one_hot_samples


def coocurence_matrix(filename):
  f = open(filename, 'r')

  lines = f.readlines()
  coocurence_matrix = {}

  for line in lines:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')

    coocurence_matrix[word] = coefs

  emb_vectors = coocurence_matrix.values()
  first_embvec = next(iter(emb_vectors))

  embvec_size = first_embvec.shape[0]

  return coocurence_matrix, embvec_size


# returns the number of kmers of a sequence
def num_of_kmers(k, sequence, overlapping):
  if overlapping == "non-overlapping":
    return int(len(sequence) / k)
  else:
    return int(len(sequence) - k +1)


# returns a list of kmers of a sequence
def k_mers(sequence, k, overlapping):
  
  kmers = []
  shift = 0
  count_kmers = 0

  if overlapping == 'non-overlapping':
    shift = k
    # count_kmers = len(sequence)
    count_kmers = num_of_kmers(k, sequence, overlapping)


  else:
    shift = 1
    # count_kmers = len(sequence) - k +1
    count_kmers = num_of_kmers(k, sequence, overlapping)


  for i in range(0, count_kmers, shift):
    kmer = ''
    for j in range(k):
      kmer += sequence[i+j]

    kmers.append(kmer)
  return kmers



# returns the kmer-embedding vectors of the sequences, 
# which constracted by the GloVe algorithm
def kmer_embedding(sequences, k, filename, overlapping):
  c_matrix, len_embvec = coocurence_matrix(filename)        

  num_cols = num_of_kmers(k, sequences[:,0], overlapping)

  num_rows = len_embvec


  kmers_emb_samples = np.zeros(shape = (sequences.shape[1], num_rows, num_cols, 1), dtype=np.float16)

  for (i, sequence) in enumerate(sequences.T):
    # print(len(sequence))
    kmers = k_mers(sequence, k, overlapping)
    for (col_position, kmer) in enumerate(kmers):
      emb_vector = c_matrix[kmer]
      for (row_position, num) in enumerate(emb_vector):
        # print(row_position, col_position)
        kmers_emb_samples[i, row_position, col_position, :] = num

  return kmers_emb_samples


# SOSOSOSOS --> i prwti thesi tou sequence einai i mideniki
# returns a subsequence that starts at the the position[0] of the sequence 
# and ends to the position[1] of the sequence
def subsequences(sequences, positions, overlapping, k=3):
  # applying the above equeations, we will obtain in-frame TISs
  start_point=positions[0]
  end_point=positions[1]
  
  window_size = end_point - start_point + 1
  subseqs = np.zeros(shape = (window_size, len(sequences)), dtype=str)

  for (col_position, sequence) in enumerate(sequences):
    for i in range(start_point, end_point + 1):
      row_position = i - start_point
      subseqs[row_position, col_position] = sequence[i]
  

  return subseqs, window_size
  


def convert_sequences(sequences, positions, k, overlapping, filename=''):
  sequences, window_size = subsequences(sequences, positions, overlapping, k)

  sequences = clean_sequences(sequences, window_size)

  one_samples = one_hot_encoding(sequences)

  # kmer_emb = kmer_embedding(sequences, k, filename, overlapping)

  return one_samples


# creates and returns sets, i.e either training set or test set, which are consist of 
# samples and labels, <setname>_x and <setname>_y corresponding, where <setname>
# corresponds to either training set or test set
def create_sets(pos_sequences, neg_sequences, file_pos, file_neg, overlapping, positions=[], k=3, split=False):
  s = []

  set_x_pos = convert_sequences(pos_sequences, positions, k, overlapping, file_pos)

  set_x_neg = convert_sequences(neg_sequences, positions, k, overlapping, file_neg)

  set_x = np.concatenate((set_x_pos, set_x_neg))


  set_y_pos = np.ones((set_x_pos.shape[0],1), dtype=int)
  set_y_neg = np.zeros((set_x_neg.shape[0],1), dtype=int)

  set_y = np.concatenate((set_y_pos, set_y_neg)) 



  sample_dim = [set_x.shape[1], set_x.shape[2]]

  if split == True:
    train_x, val_x, train_y, val_y = train_test_split(set_x, set_y, shuffle=True, test_size=0.33)

    sample_dim = [train_x.shape[1], train_x.shape[2]]

    train_x, train_y = shuffle(train_x, train_y)
    val_x, val_y = shuffle(val_x, val_y)

    return [train_x, train_y, val_x, val_y, sample_dim]
  
  else:
    set_x, set_y = shuffle(set_x, set_y)

    return [set_x, set_y, sample_dim]



create_testing_set = create_training_set = create_sets


