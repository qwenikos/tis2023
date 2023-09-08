import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = "-1" ## tell to use cpu
import sys
import argparse
import numpy as np

from misc import k_mers, read_fasta_file



k=5
start_point = 0
end_point = 300


overlapping="overlapping" ##choices=['overlapping', 'non-overlapping']

train_pos="../datasets/training/positive/positive_trainingSet_Flank-100.fa"
train_pos="../datasets/training/positive/both_positive_trainingSet_Flank-100.fa"
train_pos_sequences = read_fasta_file(train_pos,start_point,end_point)

sequences=np.array([list(sequence) for sequence in train_pos_sequences])

outFileName = 'corpus_text.txt'

f = open(outFileName,'w')

for sequence in sequences:

    kmers = k_mers(sequence, k, overlapping)

    for kmer in kmers:
        f.write(kmer)
        f.write(' ')

f.write('\n')

f.close()

#########################GLOBE##################################

CORPUS="corpus_text.txt"
VOCAB_FILE="vocab.txt"
COOCCURRENCE_FILE="cooccurrence.bin"
COOCCURRENCE_SHUF_FILE="cooccurrence.shuf.bin"
BUILDDIR="../glove/build"
SAVE_FILE="vectors"
VERBOSE="2"
MEMORY="4.0"
VOCAB_MIN_COUNT="0" #5
VECTOR_SIZE="100"  ##36
MAX_ITER="100" #300
WINDOW_SIZE="7" ## 3
BINARY="2"
NUM_THREADS="5"
X_MAX="30000"
outputFile=f"{k}-mer_emb.txt"

cmd1=f"{BUILDDIR}/vocab_count -min-count {VOCAB_MIN_COUNT} -verbose {VERBOSE} < {CORPUS} > {VOCAB_FILE}"
print (cmd1)
returned_value = os.system(cmd1)  # returns the exit code in unix
print('returned value:', returned_value)

cmd2=f"{BUILDDIR}/cooccur -memory {MEMORY} -vocab-file {VOCAB_FILE} -verbose {VERBOSE} -window-size {WINDOW_SIZE} < {CORPUS} > {COOCCURRENCE_FILE}"
print (cmd2)
returned_value = os.system(cmd2)  # returns the exit code in unix
print('returned value:', returned_value)

cmd3=f"{BUILDDIR}/shuffle -memory {MEMORY} -verbose {VERBOSE} < {COOCCURRENCE_FILE} > {COOCCURRENCE_SHUF_FILE}"
print (cmd3)
returned_value = os.system(cmd3)  # returns the exit code in unix
print('returned value:', returned_value)

cmd4=f"{BUILDDIR}/glove -save-file {SAVE_FILE} -threads {NUM_THREADS} -input-file {COOCCURRENCE_SHUF_FILE} -x-max {X_MAX} -iter {MAX_ITER} -vector-size {VECTOR_SIZE} -binary {BINARY} -vocab-file {VOCAB_FILE} -verbose {VERBOSE}"
print (cmd4)
returned_value = os.system(cmd4)  # returns the exit code in unix
print('returned value:', returned_value)

cmd5=f"mv {SAVE_FILE}.txt {outputFile}"
print (cmd5)
returned_value = os.system(cmd5)  # returns the exit code in unix
print('returned value:', returned_value)
