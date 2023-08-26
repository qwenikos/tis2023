
debug=True

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from misc import read_fasta_file, create_training_set_one_hot, create_testing_set_one_hot,create_training_set_emb,create_testing_set_emb

from train import train_model,train_model_one_hot

from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, ReduceLROnPlateau

#######################   General variable ###########################################################
##dataset creation
num_tr_data =130
num_te_data =130
start_point = 0 ##def 60-120
end_point   = 200

## training params
# model_type='gated_cnn'      ## choices=['cnn', 'gated_cnn', 'hybrid'], help="if the kmers are overlapping")
model_type   ='cnn' 

params_tuning="no" ##choices=['yes', 'no'], default='no', help="Enables the hyperparameter tuning") if yes the member must be given as list of values eg params['lr'] =[0.001,0.002]

##kmer embendings params
vector_size ="30"
vector_size = "vector_size_"+ vector_size
overlapping = 'non-overlapping' ##choices=['overlapping', 'non-overlapping'], help="if the kmers are overlapping")
made_by     ='positive'  ## choices=['positive', 'positive_negative']

# k          = ['3,4'] ##what kmers create
k           = [3] ## because i use on hot encoding to avoid loop\

### params for training embendings
params = {'lr': [], 'k': [], 'batch_size': [], 'kernel_size': [], 'layers':[], 'sample_dim':[], 'mntm':[], 'epochs': [], 'flt': [], 'branch': []}
params_tuning == 'no'
params['lr'] = 0.001
params['batch_size'] = 64 
params['kernel_size'] = 5          
params['layers'] = 15  
params['epochs'] = 50 
params['flt'] = 25
params['branch'] = 3
params['k'] = k ## because i use on hot encoding to avoid loop\


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
printd (sample_dim_hot)
# print(train_x_hot[0].shape)
# exit()
print (type(train_x_hot),train_x_hot.shape)
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

params['sample_dim'] = sample_dim_hot  
printd ("train_x_hot.shape",train_x_hot.shape)

###############################33 TRAINING
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

print(type(train_x_hot), type(train_x_hot[0]), len(train_x_hot[0]))
train_x_hot=list(train_x_hot)
train_x_hot=[train_x_hot]

model = create_model_one_hot(model_type, sample_dim=params['sample_dim'], kernel_size=params['kernel_size'], flt=params['flt'], lr=params['lr'], layers=params['layers'])
		
model.fit(train_x, train_y, validation_data = (val_x, val_y), shuffle=True, epochs=params['epochs'], batch_size=params['batch_size'], callbacks = [earlystopper, csv_logger, mcp, reduce_lr], verbose=1)

model.save('results/saved_model.h5')

print("\n\t\t\t\tEvaluation: [loss, acc]\n")

# print(test_x[3].shape, test_y[3].shape, params['sample_dim'][3])
# print(test_x[4].shape, test_y[4].shape, params['sample_dim'][4])
# print(test_x[5].shape, test_y[5].shape, params['sample_dim'][5])


tresults = model.evaluate(test_x, test_y,
										batch_size = params['batch_size'],
										verbose = 1,
										sample_weight = None)
		
print(params)
print(tresults)
exit()
# model, best_params, results = train_model_one_hot(model_type, train_x_hot, train_y_hot, val_x_hot, val_y_hot, test_x_hot, test_y_hot, params, params_tuning)
train_model_one_hot



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




# model, best_params, results = train_model(model_type, train_x_emb, train_y_emb, val_x_emb, val_y_emb, test_x_emb, test_y_emb, params, params_tuning)


print ("----END----")