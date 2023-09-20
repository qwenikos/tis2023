##it use the 3_4 model
print ("\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ START $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n")
debug=True
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = "-1" ## tell to use cpu

from misc import read_fasta_file, create_sets_seq_one_hot,create_sets_kmer_one_hot,create_sets_kmer_emb
from misc import convert_sequences_to_one_hot,convert_sequences_to_kmers_one_hot,convert_sequences_to_embedding

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

input_fasta="../datasets/testing/positive/positive_testingSet_Flank-100.fa"



#### variables

k1=3
k2=5

input_data_count =100

start_point = 0 ##def 60-120
end_point   = 300


model_type   ='cnn' 
flt         = 25
kernel_size = 5
lr          = 0.001
batch_size  = 64
epochs      = 50 
overlapping = 'overlapping'  
######################################################################################################

input_sequences = read_fasta_file(input_fasta,start_point,end_point, input_data_count) ##num_tr_data <>0 then return num_tr RANDOM samples. return a list

print ("&& 1 &&")
#create_sets_seq_one_hot(test_pos_sequences, test_neg_sequences)
seq_one_hot = convert_sequences_to_one_hot(input_sequences)

print ("&& 2  &&")
k=k1
file_pos=str(k) + '-mer_emb.txt'  
#create_sets_kmer_one_hot
#create_sets_kmer_emb
kmer_one_hot_1 = convert_sequences_to_kmers_one_hot(input_sequences,overlapping=overlapping,k=k)
kmer_emb_1     = convert_sequences_to_embedding(input_sequences, k=k, overlapping=overlapping,file_pos=file_pos)


print ("&&  3  &&")
k=k2
file_pos=str(k) + '-mer_emb.txt'  

kmer_one_hot_2  = convert_sequences_to_kmers_one_hot(input_sequences,overlapping=overlapping,k=k)
kmer_emb_2 = convert_sequences_to_embedding(input_sequences, k=k, overlapping=overlapping,file_pos=file_pos)


print ("&&  4  &&")
input_data_x = [seq_one_hot,kmer_one_hot_1 ,kmer_emb_1,kmer_one_hot_2 ,kmer_emb_2]



pred_model="results/np_integrate_4_branch_3_5_2023-09-08_13-35-37_start_point-0_end_point-300_k1-3_k3-5.h5"
loaded_model = load_model(pred_model)

# Now, you can use the loaded model to make predictions
predictions = loaded_model.predict(input_data_x)
print (predictions)
print (predictions.dtype)
predictions=np.round(predictions)
print (predictions.sum())

