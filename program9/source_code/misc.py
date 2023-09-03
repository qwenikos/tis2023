debug=False
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

def printd(*args):
    if debug:
        mes=" ".join(map(str,args))
        print ("--->",mes)

def read_fasta_file (input_file,start_point,end_point, num_samples=0):
  f = open(input_file,'r')

  lines = f.readlines()

  genes=[] 

  for line in lines:
    if line[0] == '>':
      continue 
    else:
      flag = 0
      for l in line:
        if l == 'N' or l == 'n':
          flag = 1
          break
      if len(line)<300: ##added by nikos to remove small seqs
        continue
      if flag == 1:
        continue
      seqPart=line[start_point:end_point].rstrip().upper()
      genes=genes+[seqPart]

  returned_genes = []

  for gene in genes:
    returned_genes.append(gene)
  
  if num_samples != 0:
    # returned_genes = rn.sample(returned_genes, num_samples)
    returned_genes = returned_genes[0:num_samples]  ##NIKOS TO RETURN EVERY TIME THE SAME SET
  f.close()

  return returned_genes

def subsequences(sequences):
  subseqs=np.array([list(sequence) for sequence in sequences])
  return subseqs

# removes the whitespaces characters at the beggining and 
# at the end of a sequence or skips a sequence which 
# contains the 'N' character
def clean_sequences(sequences):
  return sequences

# converts the sequences into one-hot encoding sequences
def one_hot_encoding (sequences):

  one_hot_samples = np.zeros(shape = (sequences.shape[0], 4, sequences.shape[1]), dtype=np.float32)

  col_position=0

  for (i, sequence) in enumerate(sequences):
    for (col_position, nucleotide) in enumerate(sequence):

      encoded_nuc = one_hot_conversion(nucleotide[0])

      for (row_position, one_hot) in enumerate(encoded_nuc):
        one_hot_samples[i, row_position, col_position] = one_hot


  return one_hot_samples



# returns the kmer-embedding vectors of the sequences, 
# which constracted by the GloVe algorithm
def kmer_embedding(sequences, k, filename, overlapping):

  c_matrix, len_embvec = coocurence_matrix(filename) 

  num_cols = num_of_kmers(k, sequences[0], overlapping)

  num_rows = len_embvec

  kmers_emb_samples = np.zeros(shape = (sequences.shape[0], num_rows, num_cols), dtype=np.float16)

  for (i, sequence) in enumerate(sequences):

    kmers = k_mers(sequence, k, overlapping)
    
    for (col_position, kmer) in enumerate(kmers):
      if kmer in c_matrix:
        emb_vector = c_matrix[kmer]
      else:                         ##handling out of vocabulary words or "unseen words"
        emb_vector = c_matrix["<unk>"] ## the last line in vector files

      for (row_position, num) in enumerate(emb_vector):

        kmers_emb_samples[i, row_position, col_position] = num
  return kmers_emb_samples

################################################################
###return a list of kmers and their one hot representatio
def np_generate_all_kmers_one_hot(k):
    nucleotides = ['A', 'T', 'C', 'G']
    kmers = ['']
    
    for a in range(k):

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
    
    for i, kmer in enumerate(kmers):
        onHot=np.asarray(one_hot_enc_kmers[:,i])

        kmersOnHotDict[kmer]=onHot

    return kmersOnHotDict,kmers,one_hot_enc_kmers

def kmers_one_hot_encoding (sequences,overlapping, k): ## add the k in args

  kmersOnHotDict,kmers,one_hot_enc_kmers=np_generate_all_kmers_one_hot(k)
 
  num_cols = num_of_kmers(k, sequences[0], overlapping) #num of kmers in sequence
  num_rows = len(kmers) ## size of on hot encoding
  
  kmers_emb_samples = np.zeros(shape = (sequences.shape[0], num_rows, num_cols), dtype=np.float16)

  for (i, sequence) in enumerate(sequences):

    kmers = k_mers(sequence, k, overlapping)

    for (col_position, kmer) in enumerate(kmers):
      if kmer in kmersOnHotDict:
        kmer_one_hot = kmersOnHotDict[kmer]

      for (row_position, num) in enumerate(kmer_one_hot):
 
        kmers_emb_samples[i, row_position, col_position] = num
      
  return kmers_emb_samples


def convert_sequences_to_kmers_one_hot(sequences, overlapping, k):
  sequences=np.array([list(sequence) for sequence in sequences]) ##added to convert list to nparray of nts
  
  one_samples = kmers_one_hot_encoding(sequences,overlapping, k) ##to impement
  return one_samples

def create_sets_kmers_one_hot(pos_sequences, neg_sequences,k,overlapping, split=False):
  s = []
  # set_x_pos = convert_sequences_to_one_hot(pos_sequences)
  set_x_pos = convert_sequences_to_kmers_one_hot(pos_sequences, overlapping, k)
  
  set_x_neg = convert_sequences_to_kmers_one_hot(neg_sequences, overlapping, k)


  set_x = np.concatenate((set_x_pos, set_x_neg)) 

  set_y_pos = np.ones((set_x_pos.shape[0],1), dtype=int)
  set_y_neg = np.zeros((set_x_neg.shape[0],1), dtype=int)

  set_y = np.concatenate((set_y_pos, set_y_neg)) 

  sample_dim = [set_x.shape[1], set_x.shape[2]]

  if split == True:

    train_x, val_x, train_y, val_y = train_test_split(set_x, set_y, shuffle=False, test_size=0.33)  ##np make shuffle==False

    sample_dim = [train_x.shape[1], train_x.shape[2]]

    train_x, train_y = shuffle(train_x, train_y)
    val_x, val_y = shuffle(val_x, val_y)

    return [train_x, train_y, val_x, val_y, sample_dim]
  
  else:
    set_x, set_y = shuffle(set_x, set_y)

    return [set_x, set_y, sample_dim]

################################################################

# converts a nucleotide into a 4-bit one-hot vector
def one_hot_conversion(nucleotide):
  one_hot_map = {
		"A": np.array([1, 0, 0, 0],dtype=np.float32), 
		"C": np.array([0, 1, 0, 0],dtype=np.float32), 
		"G": np.array([0, 0, 1, 0],dtype=np.float32), 
		"T": np.array([0, 0, 0, 1],dtype=np.float32),
    "": np.array([0, 0, 0, 0],dtype=np.float32)}

  return one_hot_map[nucleotide]

def coocurence_matrix(filename):
  f = open(filename, 'r')

  lines = f.readlines()
  coocurence_matrix = {}

  for line in lines:
    values = line.split()
    word = values[0]
    
    coefs = np.asarray(values[1:], dtype='float32')

    coocurence_matrix[word] = coefs
  embvec_size=len(coefs)

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
 

  else:  ##if overlapping
    shift = 1
    # count_kmers = len(sequence) - k +1
    count_kmers = num_of_kmers(k, sequence, overlapping)

  for i in range(0, count_kmers): #########

    kmer = ''
    for j in range(k):
      kmer += sequence[i*shift+j]

    kmers.append(kmer)

  return kmers

def convert_sequences_to_one_hot(sequences):
  sequences=np.array([list(sequence) for sequence in sequences]) ##added to convert list to nparray of nts

  one_samples = one_hot_encoding(sequences)
  
  return one_samples


def convert_sequences_to_embedding(sequences, k, overlapping, filename=''):
  ##here sequences are lists
  sequences=np.array([list(sequence) for sequence in sequences]) ##added to convert list to nparray of nts
  
  kmer_emb = kmer_embedding(sequences, k, filename, overlapping)
  return kmer_emb


  # creates and returns sets, i.e either training set or test set, which are consist of 
  # samples and labels, <setname>_x and <setname>_y corresponding, where <setname>
  # corresponds to either training set or test set
def create_sets_one_hot(pos_sequences, neg_sequences,split=False):
  s = []
  set_x_pos = convert_sequences_to_one_hot(pos_sequences)

  set_x_neg = convert_sequences_to_one_hot(neg_sequences)

  set_x = np.concatenate((set_x_pos, set_x_neg)) 

  set_y_pos = np.ones((set_x_pos.shape[0],1), dtype=int)
  set_y_neg = np.zeros((set_x_neg.shape[0],1), dtype=int)

  set_y = np.concatenate((set_y_pos, set_y_neg)) 

  sample_dim = [set_x.shape[1], set_x.shape[2]]

  if split == True:
    ##train_x, val_x, train_y, val_y = train_test_split(set_x, set_y, shuffle=True, test_size=0.33) 
    train_x, val_x, train_y, val_y = train_test_split(set_x, set_y, shuffle=False, test_size=0.33)  ##np make shuffle==False

    sample_dim = [train_x.shape[1], train_x.shape[2]]

    train_x, train_y = shuffle(train_x, train_y)
    val_x, val_y = shuffle(val_x, val_y)

    return [train_x, train_y, val_x, val_y, sample_dim]
  
  else:
    set_x, set_y = shuffle(set_x, set_y)

    return [set_x, set_y, sample_dim]


def create_sets_emb(pos_sequences, neg_sequences, file_pos, file_neg, overlapping, k=3, split=False):
  s = []
  ##here pos_sequences and neg_sequences are lists
  
  set_x_pos = convert_sequences_to_embedding(pos_sequences, k, overlapping, file_pos)
 
  set_x_neg = convert_sequences_to_embedding(neg_sequences, k, overlapping, file_neg)

  set_x = np.concatenate((set_x_pos, set_x_neg)) ######ERROR why concat 

  set_y_pos = np.ones((set_x_pos.shape[0],1), dtype=int)
  set_y_neg = np.zeros((set_x_neg.shape[0],1), dtype=int)

  set_y = np.concatenate((set_y_pos, set_y_neg)) 

  # sample_dim = [set_x.shape[1], set_x.shape[2],1] ## np why the 1 at the end
  sample_dim = [set_x.shape[1], set_x.shape[2]] 

  if split == True:
    ##train_x, val_x, train_y, val_y = train_test_split(set_x, set_y, shuffle=True, test_size=0.33) 
    train_x, val_x, train_y, val_y = train_test_split(set_x, set_y, shuffle=False, test_size=0.33)  ##np make shuffle==False

    # sample_dim = [train_x.shape[1], train_x.shape[2], 1] ## np why the 1 at the end
    sample_dim = [train_x.shape[1], train_x.shape[2]] 
    train_x, train_y = shuffle(train_x, train_y)
    val_x, val_y = shuffle(val_x, val_y)

    return [train_x, train_y, val_x, val_y, sample_dim]
  
  else:
    set_x, set_y = shuffle(set_x, set_y)

    return [set_x, set_y, sample_dim]


create_testing_set_one_hot      = create_training_set_one_hot      = create_sets_one_hot
create_testing_set_emb          = create_training_set_emb          = create_sets_emb
create_testing_set_kmer_one_hot = create_training_set_kmer_one_hot = create_sets_kmers_one_hot
