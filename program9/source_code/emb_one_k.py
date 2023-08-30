debug=True

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = "-1" ## tell to use cpu

from misc import read_fasta_file,create_training_set_emb, create_testing_set_emb

from train import train_model
from models import create_cnn,cnn,classification

from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, ReduceLROnPlateau
from keras.models import Model
from keras.layers import Input
from keras.optimizers import SGD
import numpy as np

############################################
def printd(*args):
    if debug:
        mes=" ".join(map(str,args))
        print ("--->",mes)
#############################################

#######################   General variable ###########################################################
#### data files
train_pos="../datasets/training/positive/positive_trainingSet_Flank-100.fa"
## >ENSG00000004776|19|O14558|-1|35754566|35757029|ENST00000004982
## CGGTCCCAGTGTCCCTGGGCAGCCCTCCGAGGGGCCGGCACAGGGCGCACTATAAATGAGCGGCTGCGCACGCAGGGGCACTGCAACGCGGAGGAGCAGGATGGAGATCCCTGTGCCTGTGCAGCCGTCTTGGCTGCGCCGCGCCTCGGCCCCGTTGCCCGGACTTTCGGCGCCCGGACGCCTCTTTGACCAGCGCTTCGGCGAGGGGCTGCTGGAGGCCGAGCTGGCTGCGCTCTGCCCCACCACGCTCGCCCCCTACTACCTGCGCGCACCCAGCGTGGCGCTGCCCGTCGCCCAGGTGCCGACGGACCCCGGCCACTTTTCGGTGCTGCTAGACGTGAAGCACTTCTCGCCGGAGGAAATTGCTGTCAAGGTGGTGGGCGAACACGTGGAGGTGCACGCGCGCCACGAGGAGCGCCCGGATGAGCACGGATTCGTCGCGCGCGAGTTCCACCGTCGCTACCGCCTGCCGCCTGGCGTGGATCCGGCTGCCGTGACGTCCGCGCTGTCCCCCGAGGGCGTCCTGTCCATCCAGGCCGCACCAGCGTCGGCCCAGGCCCCACCGCCAGCCGCAGCCAAGTAG

train_neg="../datasets/training/negative/negative_trainingSet_Flank-100.fa"
## >REGION_GT_20000_LT_50000_1:chr1:38501:38901:0:+
## TAGTGTGAGAGAAGAGAGATAAATTGAGAAAGAGACTGGTTTTTAAACTGTTAAAATTGAATCAGGACTTGATGATTTTGAAAATTGTCAGTCTCCCCACATGGAAAAAGATGCTGAAATTAACAAATGGCTTCTGAGCATGTGGCATAGGGTGTAACTGTACAGTCTTTTGTGATTATGCATAAAGATCAAAGGATGGGAGTAGCAATGAGTCACACAGAGGTCTGTTGCAAGAGATTACAAGGGTGTACCATGCAGAACCTCTCCACCAAACCTTAGGGCCCTTGGGAAGCTTCAGTGAGTTACCCTGGGGGCCATCTTGGCAGGAGCTGAAGGTAGAAAGGTAGAGTTTATCTCTAAAAGATTCATGGGTATGGCTCTTGACAAATCGACTATGAGC

test_pos="../datasets/testing/positive/positive_testingSet_Flank-100.fa"
##>ENSG00000003989|8|P52569|1|17538777|17570561|ENST00000004531
##TGGTAAAGATCCCTACTTATCTATACGATTTAATTACTTTTTTTTCCATCAGTCCCAAAACAGAAAGAGCAGATGTCTCACCACGAAACTAGCAACTGGAATGAAGATAGAAACAAGTGGTTATAACTCAGACAAACTAATTTGTCGAGGGTTTATTGGAACACCTGCCCCACCGGTTTGCGACAGCAAGTTTCTCCTGTCGCCTTCGTCAGACGTCAGAATGATTCCTTGCAGAGCCGCGCTGACCTTTGCCCGATGTCTGATCCGGAGAAAAATCGTGACCCTGGACAGTCTAGAAGACACCAAATTATGCCGCTGCTTATCCACCATGGACCTCATTGCCCTGGGCGTTGGAAGCACCCTTGGGGCCGGGGTTTATGTCCTCGCTGGGGAGGTGGCCAAGGCAGACTCGGGCCCCAGCATCGTGGTGTCCTTCCTCATTGCTGCCCTGGCTTCAGTGATGGCTGGCCTCTGCTATGCCGAATTTGGGGCCCGTGTTCCCAAGACGGGGTCTGCATATTTGTACACCTACGTGACTGTCGGAGAGCTGTGGGCCTTCATCACTGGCTGGAATCTCATTTTATCGTATGTGATAGGTACATCAAGTGTTGCAAGAGCCTGGAGTGGCACCTTTGATGAACTTCTTAGCAAACAGATTGGTCAGTTTTTGAGGACATACTTCAGAATGAATTACACTGGTCTTGCAGAATATCCCGATTTTTTTGCTGTGTGCCTTATATTACTTCTAGCAGGTCTTTTGTCTTTTGGAGTAAAAGAGTCTGCTTGGGTGAATAAAGTCTTCACAGCTGTTAATATTCTCGTCCTTCTGTTTGTGATGGTTGCTGGGTTTGTGAAAGGAAATGTGGCAAACTGGAAGATTAGTGAAGAGTTTCTCAAAAATATATCAGCAAGTGCCAGAGAGCCACCTTCTGAAAACGGAACAAGTATCTATGGGGCTGGTGGCTTTATGCCTTATGGCTTTACGGGAACGTTGGCTGGTGCTGCAACTTGCTTTTATGCCTTTGTGGGATTTGACTGCATTGCAACAACTGGTGAAGAAGTTCGGAATCCCCAGAAAGCTATTCCCATTGGAATTGTGACGTCTTTGCTTGTTTGCTTTATGGCCTATTTTGGGGTCTCTGCAGCTTTAACACTTATGATGCCGTACTACCTCCTCGATGAAAAAAGCCCCCTTCCTGTAGCGTTTGAATATGTGGGATGGGGTCCTGCCAAATATGTCGTCGCAGCTGGTTCTCTCTGCGCCTTGTCAACAAGTCTTCTTGGATCCATTTTCCCAATGCCTCGTGTAATCTATGCTATGGCGGAGGATGGGTTGCTTTTCAAATGTCTAGCTCAAATCAATTCCAAAACGAAGACACCAATAATTGCTACTTTATCATCGGGTGCAGTGGCAGCTTTGATGGCCTTTCTGTTTGACCTGAAGGCGCTTGTGGACATGATGTCCATTGGCACACTCATGGCCTACTCTCTGGTGGCAGCCTGTGTTCTCATCCTCAGGTACCAGCCTGGCTTATCTTACGACCAGCCCAAATGTTCTCCTGAGAAAGATGGTCTGGGATCGTCTCCCAGGGTAACCTCGAAGAGTGAGTCCCAGGTCACCATGCTGCAGAGACAGGGCTTCAGCATGCGGACCCTCTTCTGCCCCTCCCTTCTGCCAACACAGCAGTCAGCTTCTCTCGTGAGCTTTCTGGTAGGATTCCTAGCTTTCCTCGTGTTGGGCCTGAGTGTCTTGACCACTTACGGAGTTCATGCCATCACCAGGCTGGAGGCCTGGAGCCTCGCTCTCCTCGCGCTGTTTCTTGTTCTCTTCGTTGCCATCGTTCTCACCATCTGGAGGCAGCCCCAGAATCAGCAAAAAGTAGCCTTCATGGTTCCATTCTTACCATTTTTGCCAGCGTTCAGCATCTTGGTGAACATTTACTTGATGGTCCAGTTAAGTGCAGACACTTGGGTCAGATTCAGCATTTGGATGGCAATTGGCTTCCTGATTTACTTTTCTTATGGCATTAGACACAGCCTGGAGGGTCATCTGAGAGATGAAAACAATGAAGAAGATGCTTATCCAGACAACGTTCATGCAGCAGCAGAAGAAAAATCTGCCATTCAAGCAAATGACCATCACCCAAGAAATCTCAGTTCACCTTTCATATTCCATGAAAAGACAAGTGAATTCTAA

test_neg="../datasets/testing/negative/negative_testingSet_Flank-100.fa"
##>REGION_GT_50000_1_sorf_1:chr1:102467:102867:0:+
##CATTCTCATATGACAGATTTCAGATGGCATTCTTATTTCCCTGATTTCTTTTTGAGATAGCTTGCATTTCCCTCCTCTATATAAAGCCACCGTTTATCAAATGCCTACATGGACCAAGCAGTCCACAAGGGCTTCACAGACAGTTTTACTAAACTCATGCCAAAACTTTCAGGTTTTATACCTACCTTATAGATAAAGAAATTGAAGCTTATAGAGTTTAAGTAATGTTCCCAAAGCCTCGTGGCTAGTAATTCAAACCTAATTTCTGCCTACTCCAAAGTCTATTTTTCCTTATGATACTCTACTGCCTCTCCATGGATAAAGACAGAGATCACATATTAATAAAATTTGCACAAAGTCGGCAAATTGTTGAAAGGGAAGGCTAAGATGATTAATAAAA



##dataset creation
num_tr_data =1000      
num_te_data =1000
start_point = 0 ##def 60-120 works well for 0 200
end_point   = 200
model_type   ='cnn' 

flt         = 25
lr          = 0.001
batch_size  = 64
kernel_size = 5
epochs      = 50 
layers =15 ##  default=15, help="Number of gated convolution layers")
branch = 3 ##  default='3', help="Number of branches of the model")
k = [3,4]
vector_size=30
vector_size = 'vector_size_' + str(vector_size)
overlapping = 'non-overlapping'  ##default='non-overlapping', choices=['overlapping', 'non-overlapping'], help="if the kmers are overlapping")
made_by = 'positive' ##default='positive', choices=['positive', 'positive_negative'], help="the vectors are made only by the positive dataset or
params_tuning="no"
params={}
params['k'] = k
params['lr'] = lr     
params['batch_size'] = batch_size
params['kernel_size'] = kernel_size       
params['layers'] = layers
params['epochs'] = epochs
params['flt'] = flt
params['branch'] = branch

file = '../datasets/kmer_embedding/' + made_by + '/' + overlapping + '/' + vector_size + '/training/'

######################################################################################################
train_x = {}
train_y = []

val_x = {}
val_y = []

test_x = {}
test_y = []
sample_dim = {}







######### new #######

train_pos_sequences = read_fasta_file(train_pos, start_point,end_point, num_tr_data,) ##num_tr_data <>0 then return num_tr RANDOM samples.
train_neg_sequences = read_fasta_file(train_neg, start_point,end_point, num_tr_data)
print (len(train_neg_sequences))
for i in k:
    file_pos = file + 'positive/' + str(i) + '-mer_emb.txt'
    file_neg = file + 'negative/' + str(i) + '-mer_emb.txt' 
    train_x[i], train_y, val_x[i], val_y, sample_dim[i] = create_training_set_emb(train_pos_sequences, train_neg_sequences, positions=[start_point, end_point], file_pos=file_pos, file_neg=file_neg, overlapping=overlapping, k=i, split=True)
    print("sample_dim",sample_dim)

print ("np.shape(train_x[3])",np.shape(train_x[3]))
print ("np.shape(train_x[4])",np.shape(train_x[4]))

params['sample_dim'] = sample_dim    




test_pos_sequences = read_fasta_file(test_pos,start_point,end_point, num_te_data) ##num_tr_data <>0 then return num_tr RANDOM samples. return a list
test_neg_sequences = read_fasta_file(test_neg,start_point,end_point, num_te_data)

for i in k:
    file_pos = file + 'positive/' + str(i) + '-mer_emb.txt'
    file_neg = file + 'negative/' + str(i) + '-mer_emb.txt'


    test_x[i], test_y, _ = create_testing_set_emb(test_pos_sequences[0:num_te_data], test_neg_sequences[0:num_te_data], positions=[start_point, end_point], file_pos=file_pos, file_neg=file_neg, overlapping=overlapping, k=i)


###############################33 TRAINING################################
#model, best_params, results = train_model(model_type, train_x, train_y, val_x, val_y, test_x, test_y, params, params_tuning)

mcp = ModelCheckpoint(filepath = 'results' + "/CNNonRaw_" + str(os.getpid()) + ".hdf5",
				verbose = 0,
				save_best_only = True)


earlystopper = EarlyStopping(monitor = 'val_loss', 
					patience = 10,
					min_delta = 0,
					verbose = 1,
					mode = 'auto')

csv_logger = CSVLogger('results' + "/CNNonRaw_" + str(os.getpid()) + ".log.csv", 
				append=True, 
				separator='\t')
	
reduce_lr = ReduceLROnPlateau(monitor='val_loss', 
								factor=0.2,
                              	patience=5, 
								cooldown=1,
								min_lr=0.00001)

train_x = [v for k, v in train_x.items()] ## first place 0,1,2 depending the k, second is the samplesNum and then a 66X1 fro the embendings
val_x = [v for k, v in val_x.items()]
test_x = [v for k, v in test_x.items()]


# print ("train_x",train_x[0][0])
# print ("train_x",len(train_x))
# print ("val_x",len(val_x))
# print ("test_x",len(test_x))
# print ("train_x",np.shape(train_x[0]))

# print ("train_x",np.shape(train_x[0][0]))
# print ("val_x",np.shape(val_x[0][0]))
# print ("test_x",np.shape(test_x[0][0]))
# print()

print (sample_dim)
# print(sample_dim[i][0])
# print(sample_dim[i][1])
# print(sample_dim[i][2])

# model = create_model(model_type, sample_dim=params['sample_dim'], kernel_size=params['kernel_size'], flt=params['flt'], lr=params['lr'], layers=params['layers'], k=params['k'])
sequence_input = []
size = len(k)
k=3

print (sample_dim[k][0], sample_dim[k][1])

    
sequence_input=(Input(shape = (sample_dim[k][0], sample_dim[k][1]))) 
    # res = feature_extraction(model_type, sequence_input[j], kernel_size, flt, layers)

out = cnn(sequence_input)
    
#     ### end feature extraction
# print(out.shape)
# concatenated.append(out)

# out = tf.concat([i for i in concatenated], axis=1)

out = classification(out)

model = Model(inputs=sequence_input, outputs=out)

sgd = SGD(learning_rate = lr, decay = 1e-6, momentum = 0.9, nesterov = True)

model.compile(loss = "binary_crossentropy", optimizer=sgd, metrics = ["accuracy"])

    
model.fit(train_x[k-3], train_y, validation_data = (val_x[k-3], val_y), shuffle=True, epochs=params['epochs'], batch_size=params['batch_size'], callbacks = [earlystopper, csv_logger, mcp, reduce_lr], verbose=1)

# model.save('results/saved_model.h5')

print("\n\t\t\t\tEvaluation: [loss, acc]\n")
exit()

tresults = model.evaluate(test_x, test_y,
                                batch_size = params['batch_size'],
                                verbose = 1,
                                sample_weight = None)

print(params)
print(tresults)