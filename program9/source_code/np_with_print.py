debug=True

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

os.environ['CUDA_VISIBLE_DEVICES'] = "-1" ## tell to use cpu

from misc import read_fasta_file, create_training_set_one_hot, create_testing_set_one_hot,create_training_set_emb,create_testing_set_emb

from train import train_model,train_model_one_hot
from models import create_cnn,cnn,classification

from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, ReduceLROnPlateau
from keras.models import Model
from keras.layers import Input
from keras.optimizers import SGD
import numpy as np

#######################   General variable ###########################################################
##dataset creation
num_tr_data =100
num_te_data =100
start_point = 0 ##def 60-120
end_point   = 200

## training params
# model_type='gated_cnn'      ## choices=['cnn', 'gated_cnn', 'hybrid'], help="if the kmers are overlapping")
model_type   ='cnn' 

##kmer embendings params
vector_size ="30"
vector_size = "vector_size_"+ vector_size
overlapping = 'non-overlapping' ##choices=['overlapping', 'non-overlapping'], help="if the kmers are overlapping")
made_by     ='positive'  ## choices=['positive', 'positive_negative']

flt=25
kernel_size=5
lr= 0.001
batch_size=64
epochs= 50 

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

############################################
def printd(*args):
    if debug:
        mes=" ".join(map(str,args))
        print ("--->",mes)
#############################################

train_x_hot = {}
train_y_hot = []
val_x_hot = {}
val_y_hot = []
test_x_hot = {}
test_y_hot = []
sample_dim_hot = {}

train_x_emb = {}
train_y_emb = []
val_x_emb = {}
val_y_emb = []
test_x_emb = {}
test_y_emb = []
sample_dim_emb = {}
######################################################################################################


###################################   create TRAIN ONE HOT data   #############################################
#######################################################################################################
printd("-------------create_training_set_one_hot--------------")

train_pos_sequences = read_fasta_file(train_pos, start_point,end_point, num_tr_data,) ##num_tr_data <>0 then return num_tr RANDOM samples.
train_neg_sequences = read_fasta_file(train_neg, start_point,end_point, num_tr_data)
printd ("len(train_pos_sequences)",len(train_pos_sequences))
printd ("len(train_neg_sequences)",len(train_neg_sequences))

train_x_hot, train_y_hot, val_x_hot, val_y_hot, sample_dim_hot = create_training_set_one_hot(train_pos_sequences, train_neg_sequences, split=True)
print("+++++++++")
printd ("sample_dim_hot",sample_dim_hot)
printd ("type(train_x_hot)",type(train_x_hot))
printd ("train_x_hot.shape",train_x_hot.shape)
printd ("train_y_hot.shape",train_y_hot.shape)
printd ("val_x_hot.shape",val_x_hot.shape)
printd ("val_y_hot.shape)",val_y_hot.shape)

printd ("------------------------")
printd ("----create TEST data----")
test_pos_sequences = read_fasta_file(test_pos,start_point,end_point, num_te_data) ##num_tr_data <>0 then return num_tr RANDOM samples. return a list
test_neg_sequences = read_fasta_file(test_neg,start_point,end_point, num_te_data)

printd ("len(test_pos_sequences)",len(test_pos_sequences))
printd ("len(test_neg_sequences)",len(test_neg_sequences))
test_x_hot, test_y_hot, _ = create_testing_set_one_hot(test_pos_sequences, test_neg_sequences)

printd ("len(test_x_hot)",len(test_x_hot))
printd ("len(test_y_hot)",len(test_y_hot))

printd ("----train the model----")



###############################33 TRAINING################################
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




  
printd('input_shape', sample_dim_hot[0], sample_dim_hot[1])

sequence_input=Input(shape = (sample_dim_hot[0], sample_dim_hot[1]))

out = cnn(sequence_input)

printd("out_shape",out.shape)

out = classification(out)
print("out_classification_shape",out.shape)

model = Model(inputs=sequence_input, outputs=out)
printd(model.summary())


sgd = SGD(learning_rate = lr, decay = 1e-6, momentum = 0.9, nesterov = True)

model.compile(loss = "binary_crossentropy", optimizer=sgd, metrics = ["accuracy"])



train_x_hot=list(train_x_hot)
train_x_hot=[train_x_hot]

val_x_hot=list(val_x_hot)
val_x_hot=[val_x_hot]

test_x_hot=list(test_x_hot)
test_x_hot=[test_x_hot]

print (np.shape(train_x_hot))
print (np.shape(train_y_hot))
print (np.shape(test_x_hot))
print (np.shape(test_y_hot))
print (np.shape(val_x_hot))
print (np.shape(val_y_hot))

model.fit(train_x_hot, train_y_hot, validation_data = (val_x_hot, val_y_hot), shuffle=True, epochs=epochs, batch_size=batch_size, callbacks = [earlystopper, csv_logger, mcp, reduce_lr], verbose=2)
exit()
model.save('results/saved_model.h5')

print("\n\t\t\t\tEvaluation: [loss, acc]\n")




tresults = model.evaluate(test_x, test_y, batch_size = batch_size, verbose = 1, sample_weight = None)
		

print(tresults)
exit()




#####################################   create EMBEDDING data   ############################################
#######################################################################################################
file = '../datasets/kmer_embedding/' + made_by + '/' + overlapping + '/' + vector_size + '/training/'
for i in k:
    file_pos = file + 'positive/' + str(i) + '-mer_emb.txt'
    file_neg = file + 'negative/' + str(i) + '-mer_emb.txt' 
    ##create_training_set In code to change from 1hot to kmers

    print("-------------create_training_set_embendings--------------")
    train_x_emb[i], train_y_emb, val_x_emb[i], val_y_emb, sample_dim_emb[i] = create_training_set_emb(train_pos_sequences, train_neg_sequences, file_pos=file_pos, file_neg=file_neg, overlapping=overlapping, k=i, split=True)
    print (sample_dim_emb[i])
    print ("len(train_x_emb[i])",len(train_x_emb[i]))
    print ("len(train_y_emb)",len(train_y_emb))
    print ("len(val_x_emb[i])",len(val_x_emb[i]))
    print ("len(val_y_emb)",len(val_y_emb))


# file = '../datasets/kmer_embedding/' + made_by + '/' + overlapping + '/' + vector_size + '/testing/' ##use the positive


for i in k:
    file_pos = file + 'positive/' + str(i) + '-mer_emb.txt'
    file_neg = file + 'negative/' + str(i) + '-mer_emb.txt' 
    ##create_training_set In code to change from 1hot to kmers
    test_x_emb[i], test_y_emb, _ = create_testing_set_emb(test_pos_sequences, test_neg_sequences, file_pos=file_pos, file_neg=file_neg, overlapping=overlapping, k=i)

    print ("len(test_x_emb[i])",len(test_x_emb[i]))
    print ("len(test_y_emb)",len(test_y_emb))



#######################################    train the model  ###########################################
#######################################################################################################







print ("----END----")