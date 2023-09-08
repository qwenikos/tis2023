print ("\n$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ START $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n")
debug=True
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = "-1" ## tell to use cpu

from misc import read_fasta_file, create_sets_seq_one_hot,create_sets_kmer_one_hot,create_sets_kmer_emb

from models import create_cnn,cnn,classification

from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, ReduceLROnPlateau
from keras.models import Model
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
train_pos="../datasets/training/positive/positive_trainingSet_Flank-100.fa"
train_pos="../datasets/training/positive/both_positive_trainingSet_Flank-100.fa"
## >ENSG00000004776|19|O14558|-1|35754566|35757029|ENST00000004982
## CGGTCCCAGTGTCCCTGGGCAGCCCTCCGAGGGGCCGGCACAGGGCGCACTATAAATGAGCGGCTGCGCACGCAGGGGCACTGCAACGCGGAGGAGCAGGATGGAGATCCCTGTGCCTGTGCAGCCGTCTTGGCTGCGCCGCGCCTCGGCCCCGTTGCCCGGACTTTCGGCGCCCGGACGCCTCTTTGACCAGCGCTTCGGCGAGGGGCTGCTGGAGGCCGAGCTGGCTGCGCTCTGCCCCACCACGCTCGCCCCCTACTACCTGCGCGCACCCAGCGTGGCGCTGCCCGTCGCCCAGGTGCCGACGGACCCCGGCCACTTTTCGGTGCTGCTAGACGTGAAGCACTTCTCGCCGGAGGAAATTGCTGTCAAGGTGGTGGGCGAACACGTGGAGGTGCACGCGCGCCACGAGGAGCGCCCGGATGAGCACGGATTCGTCGCGCGCGAGTTCCACCGTCGCTACCGCCTGCCGCCTGGCGTGGATCCGGCTGCCGTGACGTCCGCGCTGTCCCCCGAGGGCGTCCTGTCCATCCAGGCCGCACCAGCGTCGGCCCAGGCCCCACCGCCAGCCGCAGCCAAGTAG

train_neg="../datasets/training/negative/negative_trainingSet_Flank-100.fa"
train_neg="../datasets/training/negative/both_negative_trainingSet_Flank-100.fa"
## >REGION_GT_20000_LT_50000_1:chr1:38501:38901:0:+
## TAGTGTGAGAGAAGAGAGATAAATTGAGAAAGAGACTGGTTTTTAAACTGTTAAAATTGAATCAGGACTTGATGATTTTGAAAATTGTCAGTCTCCCCACATGGAAAAAGATGCTGAAATTAACAAATGGCTTCTGAGCATGTGGCATAGGGTGTAACTGTACAGTCTTTTGTGATTATGCATAAAGATCAAAGGATGGGAGTAGCAATGAGTCACACAGAGGTCTGTTGCAAGAGATTACAAGGGTGTACCATGCAGAACCTCTCCACCAAACCTTAGGGCCCTTGGGAAGCTTCAGTGAGTTACCCTGGGGGCCATCTTGGCAGGAGCTGAAGGTAGAAAGGTAGAGTTTATCTCTAAAAGATTCATGGGTATGGCTCTTGACAAATCGACTATGAGC

test_pos="../datasets/testing/positive/positive_testingSet_Flank-100.fa"
##>ENSG00000003989|8|P52569|1|17538777|17570561|ENST00000004531
##TGGTAAAGATCCCTACTTATCTATACGATTTAATTACTTTTTTTTCCATCAGTCCCAAAACAGAAAGAGCAGATGTCTCACCACGAAACTAGCAACTGGAATGAAGATAGAAACAAGTGGTTATAACTCAGACAAACTAATTTGTCGAGGGTTTATTGGAACACCTGCCCCACCGGTTTGCGACAGCAAGTTTCTCCTGTCGCCTTCGTCAGACGTCAGAATGATTCCTTGCAGAGCCGCGCTGACCTTTGCCCGATGTCTGATCCGGAGAAAAATCGTGACCCTGGACAGTCTAGAAGACACCAAATTATGCCGCTGCTTATCCACCATGGACCTCATTGCCCTGGGCGTTGGAAGCACCCTTGGGGCCGGGGTTTATGTCCTCGCTGGGGAGGTGGCCAAGGCAGACTCGGGCCCCAGCATCGTGGTGTCCTTCCTCATTGCTGCCCTGGCTTCAGTGATGGCTGGCCTCTGCTATGCCGAATTTGGGGCCCGTGTTCCCAAGACGGGGTCTGCATATTTGTACACCTACGTGACTGTCGGAGAGCTGTGGGCCTTCATCACTGGCTGGAATCTCATTTTATCGTATGTGATAGGTACATCAAGTGTTGCAAGAGCCTGGAGTGGCACCTTTGATGAACTTCTTAGCAAACAGATTGGTCAGTTTTTGAGGACATACTTCAGAATGAATTACACTGGTCTTGCAGAATATCCCGATTTTTTTGCTGTGTGCCTTATATTACTTCTAGCAGGTCTTTTGTCTTTTGGAGTAAAAGAGTCTGCTTGGGTGAATAAAGTCTTCACAGCTGTTAATATTCTCGTCCTTCTGTTTGTGATGGTTGCTGGGTTTGTGAAAGGAAATGTGGCAAACTGGAAGATTAGTGAAGAGTTTCTCAAAAATATATCAGCAAGTGCCAGAGAGCCACCTTCTGAAAACGGAACAAGTATCTATGGGGCTGGTGGCTTTATGCCTTATGGCTTTACGGGAACGTTGGCTGGTGCTGCAACTTGCTTTTATGCCTTTGTGGGATTTGACTGCATTGCAACAACTGGTGAAGAAGTTCGGAATCCCCAGAAAGCTATTCCCATTGGAATTGTGACGTCTTTGCTTGTTTGCTTTATGGCCTATTTTGGGGTCTCTGCAGCTTTAACACTTATGATGCCGTACTACCTCCTCGATGAAAAAAGCCCCCTTCCTGTAGCGTTTGAATATGTGGGATGGGGTCCTGCCAAATATGTCGTCGCAGCTGGTTCTCTCTGCGCCTTGTCAACAAGTCTTCTTGGATCCATTTTCCCAATGCCTCGTGTAATCTATGCTATGGCGGAGGATGGGTTGCTTTTCAAATGTCTAGCTCAAATCAATTCCAAAACGAAGACACCAATAATTGCTACTTTATCATCGGGTGCAGTGGCAGCTTTGATGGCCTTTCTGTTTGACCTGAAGGCGCTTGTGGACATGATGTCCATTGGCACACTCATGGCCTACTCTCTGGTGGCAGCCTGTGTTCTCATCCTCAGGTACCAGCCTGGCTTATCTTACGACCAGCCCAAATGTTCTCCTGAGAAAGATGGTCTGGGATCGTCTCCCAGGGTAACCTCGAAGAGTGAGTCCCAGGTCACCATGCTGCAGAGACAGGGCTTCAGCATGCGGACCCTCTTCTGCCCCTCCCTTCTGCCAACACAGCAGTCAGCTTCTCTCGTGAGCTTTCTGGTAGGATTCCTAGCTTTCCTCGTGTTGGGCCTGAGTGTCTTGACCACTTACGGAGTTCATGCCATCACCAGGCTGGAGGCCTGGAGCCTCGCTCTCCTCGCGCTGTTTCTTGTTCTCTTCGTTGCCATCGTTCTCACCATCTGGAGGCAGCCCCAGAATCAGCAAAAAGTAGCCTTCATGGTTCCATTCTTACCATTTTTGCCAGCGTTCAGCATCTTGGTGAACATTTACTTGATGGTCCAGTTAAGTGCAGACACTTGGGTCAGATTCAGCATTTGGATGGCAATTGGCTTCCTGATTTACTTTTCTTATGGCATTAGACACAGCCTGGAGGGTCATCTGAGAGATGAAAACAATGAAGAAGATGCTTATCCAGACAACGTTCATGCAGCAGCAGAAGAAAAATCTGCCATTCAAGCAAATGACCATCACCCAAGAAATCTCAGTTCACCTTTCATATTCCATGAAAAGACAAGTGAATTCTAA

test_neg="../datasets/testing/negative/negative_testingSet_Flank-100.fa"
##>REGION_GT_50000_1_sorf_1:chr1:102467:102867:0:+
##CATTCTCATATGACAGATTTCAGATGGCATTCTTATTTCCCTGATTTCTTTTTGAGATAGCTTGCATTTCCCTCCTCTATATAAAGCCACCGTTTATCAAATGCCTACATGGACCAAGCAGTCCACAAGGGCTTCACAGACAGTTTTACTAAACTCATGCCAAAACTTTCAGGTTTTATACCTACCTTATAGATAAAGAAATTGAAGCTTATAGAGTTTAAGTAATGTTCCCAAAGCCTCGTGGCTAGTAATTCAAACCTAATTTCTGCCTACTCCAAAGTCTATTTTTCCTTATGATACTCTACTGCCTCTCCATGGATAAAGACAGAGATCACATATTAATAAAATTTGCACAAAGTCGGCAAATTGTTGAAAGGGAAGGCTAAGATGATTAATAAAA


##dataset creation

k1=3
k2=5

num_tr_data =17000
num_te_data =10000
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


train_pos_sequences = read_fasta_file(train_pos, start_point,end_point, num_tr_data) ##num_tr_data <>0 then return num_tr RANDOM samples.
train_neg_sequences = read_fasta_file(train_neg, start_point,end_point, num_tr_data)

test_pos_sequences = read_fasta_file(test_pos,start_point,end_point, num_te_data) ##num_tr_data <>0 then return num_tr RANDOM samples. return a list
test_neg_sequences = read_fasta_file(test_neg,start_point,end_point, num_te_data)


train_x_seq_hot , train_y_seq_hot , val_x_seq_hot , val_y_seq_hot , sample_dim_seq_hot  = create_sets_seq_one_hot(train_pos_sequences, train_neg_sequences, split=True)
test_x_seq_hot , test_y_seq_hot , _ = create_sets_seq_one_hot(test_pos_sequences, test_neg_sequences)

k=k1
file_pos=str(k) + '-mer_emb.txt'  
file_neg=str(k) + '-mer_emb.txt' ## to check here must be right .need the same file for pos and neg

train_x_kmer_hot_1, train_y_kmer_hot_1, val_x_kmer_hot_1, val_y_kmer_hot_1, sample_dim_kmer_hot_1 = create_sets_kmer_one_hot(train_pos_sequences, train_neg_sequences,overlapping=overlapping,k=k, split=True)
train_x_kmer_emb_1, train_y_kmer_emb_1, val_x_kmer_emb_1, val_y_kmer_emb_1, sample_dim_kmer_emb_1 = create_sets_kmer_emb(train_pos_sequences, train_neg_sequences, file_pos=file_pos, file_neg=file_neg, overlapping=overlapping, k=k, split=True)

test_x_kmer_hot_1, test_y_kmer_hot_1, _ = create_sets_kmer_one_hot(test_pos_sequences, test_neg_sequences,overlapping=overlapping,k=k)
test_x_kmer_emb_1, test_y_kmer_emb_1, _ = create_sets_kmer_emb(test_pos_sequences, test_neg_sequences, file_pos=file_pos, file_neg=file_neg, overlapping=overlapping, k=k)

k=k2
file_pos=str(k) + '-mer_emb.txt'  
file_neg=str(k) + '-mer_emb.txt' ## to check here must be right .need the same file for pos and neg

train_x_kmer_hot_2, train_y_kmer_hot_2, val_x_kmer_hot_2, val_y_kmer_hot_2, sample_dim_kmer_hot_2 = create_sets_kmer_one_hot(train_pos_sequences, train_neg_sequences,overlapping=overlapping,k=k, split=True)
train_x_kmer_emb_2, train_y_kmer_emb_2, val_x_kmer_emb_2, val_y_kmer_emb_2, sample_dim_kmer_emb_2 = create_sets_kmer_emb(train_pos_sequences, train_neg_sequences, file_pos=file_pos, file_neg=file_neg, overlapping=overlapping, k=k, split=True)

test_x_kmer_hot_2, test_y_kmer_hot_2, _ = create_sets_kmer_one_hot(test_pos_sequences, test_neg_sequences,overlapping=overlapping,k=k)
test_x_kmer_emb_2, test_y_kmer_emb_2, _ = create_sets_kmer_emb(test_pos_sequences, test_neg_sequences, file_pos=file_pos, file_neg=file_neg, overlapping=overlapping, k=k)


input1=Input(shape = (sample_dim_seq_hot[0], sample_dim_seq_hot[1]))
out1 = cnn(input1,kernel_size,flt)

input2=Input(shape = (sample_dim_kmer_hot_1[0], sample_dim_kmer_hot_1[1]))
out2 = cnn(input2,kernel_size,flt)

input3=Input(shape = (sample_dim_kmer_emb_1[0], sample_dim_kmer_emb_1[1]))
out3 = cnn(input3,kernel_size,flt)

input4=Input(shape = (sample_dim_kmer_hot_2[0], sample_dim_kmer_hot_2[1]))
out4 = cnn(input4,kernel_size,flt)

input5=Input(shape = (sample_dim_kmer_emb_2[0], sample_dim_kmer_emb_2[1]))
out5 = cnn(input5,kernel_size,flt)

x = concatenate([out1, out2, out3, out4, out5])

# out = Dense(128, activation='relu')(merged)
print ("x.shape",x.shape)


out = Dense(units = sqrt(x.shape[1]))(x)
out = LeakyReLU()(out)
out = BatchNormalization()(out)
out = Dropout(rate = 0.2, noise_shape = None, seed = None)(out)
# # #out = Dense(units = 50, kernel_initializer = "he_normal", kernel_regularizer = regularizers.l2(0.00001))(out)
out = Dense(units = sqrt(out.shape[1]))(out)
out = LeakyReLU()(out)
out = BatchNormalization()(out)
out = Dropout(rate = 0.2, noise_shape = None, seed = None)(out)

out = Dense(units = 1, activation = "sigmoid")(out) 
 
model = tf.keras.Model(inputs=[input1,input2,input3,input4,input5], outputs=out)
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.compile(loss = "binary_crossentropy", optimizer='adam', metrics = ["accuracy",metrics.Precision(), metrics.Recall()])
model.summary()


mcp = ModelCheckpoint(filepath = 'results' + "/CNNonRaw_" + str(os.getpid()) + ".hdf5",verbose = 0,save_best_only = True)

# earlystopper = EarlyStopping(monitor = 'val_loss', patience = 10,min_delta = 0,verbose = 1,mode = 'auto')
earlystopper = EarlyStopping(monitor = 'val_accuracy', patience = 10,min_delta = 0.0001,verbose = 1,mode = 'auto')
csv_logger = CSVLogger('results' + "/CNNonRaw_" + str(os.getpid()) + ".log.csv", append=True, separator='\t')
	
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,patience=5, cooldown=1,min_lr=0.00001)


train_data_x=[train_x_seq_hot, train_x_kmer_hot_1, train_x_kmer_emb_1,train_x_kmer_hot_2, train_x_kmer_emb_2]
train_data_y=train_y_seq_hot
val_data_x=[val_x_seq_hot, val_x_kmer_hot_1, val_x_kmer_emb_1, val_x_kmer_hot_2, val_x_kmer_emb_2]
val_data_y=val_y_seq_hot


model.fit(train_data_x, train_data_y, validation_data=(val_data_x,val_data_y),shuffle=True, epochs=epochs, batch_size=batch_size, callbacks = [earlystopper, csv_logger, mcp, reduce_lr], verbose=2)

test_data_x = [test_x_seq_hot,test_x_kmer_hot_1 ,test_x_kmer_emb_1,test_x_kmer_hot_2 ,test_x_kmer_emb_2]
test_data_y = test_y_seq_hot

tresults = model.evaluate(test_data_x,test_data_y, batch_size = batch_size, verbose = 1, sample_weight = None)	
print  (tresults)

script_filename = os.path.splitext(os.path.basename(__file__))[0]
current_datetime = datetime.datetime.now()
date_time_str = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
output_filename = f"{script_filename}_{date_time_str}_start_point-{str(start_point)}_end_point-{str(end_point)}_k1-{str(k1)}_k3-{str(k2)}.h5"
model.save("results/"+output_filename)
output_filename_txt = f"{script_filename}_{date_time_str}_start_point-{str(start_point)}_end_point-{str(end_point)}_k1-{str(k1)}_k3-{str(k2)}.txt"
outFileTxt=open("results/"+output_filename_txt,"w")
outFileTxt.write(f"loss\t{tresults[0]}\naccuracy\t{tresults[1]}\nprecision \t{tresults[2]}\nrecall\t{tresults[3]}\n")
# outFileTxt.write(str(tresults))
outFileTxt.close()