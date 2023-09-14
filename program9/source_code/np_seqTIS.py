##it use the 3_4 model
print ("\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ START $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n")
debug=True
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = "-1" ## tell to use cpu

from misc import read_fasta_file, create_sets_seq_one_hot,create_sets_kmer_one_hot,create_sets_kmer_emb

from models import create_cnn,cnn,classification

from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, ReduceLROnPlateau
from keras.models import Model,load_model
from keras.layers import Input
from keras.optimizers import SGD
from keras import metrics
from math import sqrt
import os
import datetime

import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, concatenate, Dropout,LeakyReLU,BatchNormalization
import numpy as np

#######################   General variable ###########################################################
#### data files

test_pos="../datasets/testing/positive/positive_testingSet_Flank-100.fa"
##>ENSG00000003989|8|P52569|1|17538777|17570561|ENST00000004531
##TGGTAAAGATCCCTACTTATCTATACGATTTAATTACTTTTTTTTCCATCAGTCCCAAAACAGAAAGAGCAGATGTCTCACCACGAAACTAGCAACTGGAATGAAGATAGAAACAAGTGGTTATAACTCAGACAAACTAATTTGTCGAGGGTTTATTGGAACACCTGCCCCACCGGTTTGCGACAGCAAGTTTCTCCTGTCGCCTTCGTCAGACGTCAGAATGATTCCTTGCAGAGCCGCGCTGACCTTTGCCCGATGTCTGATCCGGAGAAAAATCGTGACCCTGGACAGTCTAGAAGACACCAAATTATGCCGCTGCTTATCCACCATGGACCTCATTGCCCTGGGCGTTGGAAGCACCCTTGGGGCCGGGGTTTATGTCCTCGCTGGGGAGGTGGCCAAGGCAGACTCGGGCCCCAGCATCGTGGTGTCCTTCCTCATTGCTGCCCTGGCTTCAGTGATGGCTGGCCTCTGCTATGCCGAATTTGGGGCCCGTGTTCCCAAGACGGGGTCTGCATATTTGTACACCTACGTGACTGTCGGAGAGCTGTGGGCCTTCATCACTGGCTGGAATCTCATTTTATCGTATGTGATAGGTACATCAAGTGTTGCAAGAGCCTGGAGTGGCACCTTTGATGAACTTCTTAGCAAACAGATTGGTCAGTTTTTGAGGACATACTTCAGAATGAATTACACTGGTCTTGCAGAATATCCCGATTTTTTTGCTGTGTGCCTTATATTACTTCTAGCAGGTCTTTTGTCTTTTGGAGTAAAAGAGTCTGCTTGGGTGAATAAAGTCTTCACAGCTGTTAATATTCTCGTCCTTCTGTTTGTGATGGTTGCTGGGTTTGTGAAAGGAAATGTGGCAAACTGGAAGATTAGTGAAGAGTTTCTCAAAAATATATCAGCAAGTGCCAGAGAGCCACCTTCTGAAAACGGAACAAGTATCTATGGGGCTGGTGGCTTTATGCCTTATGGCTTTACGGGAACGTTGGCTGGTGCTGCAACTTGCTTTTATGCCTTTGTGGGATTTGACTGCATTGCAACAACTGGTGAAGAAGTTCGGAATCCCCAGAAAGCTATTCCCATTGGAATTGTGACGTCTTTGCTTGTTTGCTTTATGGCCTATTTTGGGGTCTCTGCAGCTTTAACACTTATGATGCCGTACTACCTCCTCGATGAAAAAAGCCCCCTTCCTGTAGCGTTTGAATATGTGGGATGGGGTCCTGCCAAATATGTCGTCGCAGCTGGTTCTCTCTGCGCCTTGTCAACAAGTCTTCTTGGATCCATTTTCCCAATGCCTCGTGTAATCTATGCTATGGCGGAGGATGGGTTGCTTTTCAAATGTCTAGCTCAAATCAATTCCAAAACGAAGACACCAATAATTGCTACTTTATCATCGGGTGCAGTGGCAGCTTTGATGGCCTTTCTGTTTGACCTGAAGGCGCTTGTGGACATGATGTCCATTGGCACACTCATGGCCTACTCTCTGGTGGCAGCCTGTGTTCTCATCCTCAGGTACCAGCCTGGCTTATCTTACGACCAGCCCAAATGTTCTCCTGAGAAAGATGGTCTGGGATCGTCTCCCAGGGTAACCTCGAAGAGTGAGTCCCAGGTCACCATGCTGCAGAGACAGGGCTTCAGCATGCGGACCCTCTTCTGCCCCTCCCTTCTGCCAACACAGCAGTCAGCTTCTCTCGTGAGCTTTCTGGTAGGATTCCTAGCTTTCCTCGTGTTGGGCCTGAGTGTCTTGACCACTTACGGAGTTCATGCCATCACCAGGCTGGAGGCCTGGAGCCTCGCTCTCCTCGCGCTGTTTCTTGTTCTCTTCGTTGCCATCGTTCTCACCATCTGGAGGCAGCCCCAGAATCAGCAAAAAGTAGCCTTCATGGTTCCATTCTTACCATTTTTGCCAGCGTTCAGCATCTTGGTGAACATTTACTTGATGGTCCAGTTAAGTGCAGACACTTGGGTCAGATTCAGCATTTGGATGGCAATTGGCTTCCTGATTTACTTTTCTTATGGCATTAGACACAGCCTGGAGGGTCATCTGAGAGATGAAAACAATGAAGAAGATGCTTATCCAGACAACGTTCATGCAGCAGCAGAAGAAAAATCTGCCATTCAAGCAAATGACCATCACCCAAGAAATCTCAGTTCACCTTTCATATTCCATGAAAAGACAAGTGAATTCTAA

test_neg="../datasets/testing/negative/negative_testingSet_Flank-100.fa"
##>REGION_GT_50000_1_sorf_1:chr1:102467:102867:0:+
##CATTCTCATATGACAGATTTCAGATGGCATTCTTATTTCCCTGATTTCTTTTTGAGATAGCTTGCATTTCCCTCCTCTATATAAAGCCACCGTTTATCAAATGCCTACATGGACCAAGCAGTCCACAAGGGCTTCACAGACAGTTTTACTAAACTCATGCCAAAACTTTCAGGTTTTATACCTACCTTATAGATAAAGAAATTGAAGCTTATAGAGTTTAAGTAATGTTCCCAAAGCCTCGTGGCTAGTAATTCAAACCTAATTTCTGCCTACTCCAAAGTCTATTTTTCCTTATGATACTCTACTGCCTCTCCATGGATAAAGACAGAGATCACATATTAATAAAATTTGCACAAAGTCGGCAAATTGTTGAAAGGGAAGGCTAAGATGATTAATAAAA


##dataset creation

k1=3
k2=5

num_tr_data =100
num_te_data =100
start_point = 0 ##def 60-120
end_point   = 300


model_type   ='cnn' 
flt         = 25
kernel_size = 5
lr          = 0.001
batch_size  = 64
epochs      = 50 
overlapping = 'overlapping'  ##default='non-overlapping', choices=['overlapping', 'non-overlapping'], help="if the kmers are overlapping")
######################################################################################################

test_pos_sequences = read_fasta_file(test_pos,start_point,end_point, num_te_data) ##num_tr_data <>0 then return num_tr RANDOM samples. return a list
test_neg_sequences = read_fasta_file(test_neg,start_point,end_point, num_te_data)

print ("&&&&")

test_x_seq_hot , test_y_seq_hot , _ = create_sets_seq_one_hot(test_pos_sequences, test_neg_sequences)
print ("&&&&")
k=k1
file_pos=str(k) + '-mer_emb.txt'  
file_neg=str(k) + '-mer_emb.txt' ## to check here must be right .need the same file for pos and neg

test_x_kmer_hot_1, test_y_kmer_hot_1, _ = create_sets_kmer_one_hot(test_pos_sequences, test_neg_sequences,overlapping=overlapping,k=k)
test_x_kmer_emb_1, test_y_kmer_emb_1, _ = create_sets_kmer_emb(test_pos_sequences, test_neg_sequences, file_pos=file_pos, file_neg=file_neg, overlapping=overlapping, k=k)
print ("&&&&")
k=k2
file_pos=str(k) + '-mer_emb.txt'  
file_neg=str(k) + '-mer_emb.txt' ## to check here must be right .need the same file for pos and neg

test_x_kmer_hot_2, test_y_kmer_hot_2, _ = create_sets_kmer_one_hot(test_pos_sequences, test_neg_sequences,overlapping=overlapping,k=k)
test_x_kmer_emb_2, test_y_kmer_emb_2, _ = create_sets_kmer_emb(test_pos_sequences, test_neg_sequences, file_pos=file_pos, file_neg=file_neg, overlapping=overlapping, k=k)
print ("&&&&")

test_data_x = [test_x_seq_hot,test_x_kmer_hot_1 ,test_x_kmer_emb_1,test_x_kmer_hot_2 ,test_x_kmer_emb_2]
test_data_y = test_y_seq_hot

# tresults = model.evaluate(test_data_x,test_data_y, batch_size = batch_size, verbose = 1, sample_weight = None)	
# print  (tresults)

# Load the saved model
pred_model="results/np_integrate_4_branch_3_5_2023-09-08_13-35-37_start_point-0_end_point-300_k1-3_k3-5.h5"
loaded_model = load_model(pred_model)

# Now, you can use the loaded model to make predictions
predictions = loaded_model.predict(test_data_x)
print (predictions)
print (predictions.dtype)
predictions=np.round(predictions)
print (predictions)
print (test_data_y)
