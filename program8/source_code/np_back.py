
from misc import read_fasta_file, create_training_set, create_testing_set
from train import train_model

#######################   General variable #########################
num_tr_data=1000
num_te_data=1000
####################################################################

print ("----create train data----")
train_pos="../datasets/training/positive/positive_trainingSet_Flank-100.fa"
## >ENSG00000004776|19|O14558|-1|35754566|35757029|ENST00000004982
## CGGTCCCAGTGTCCCTGGGCAGCCCTCCGAGGGGCCGGCACAGGGCGCACTATAAATGAGCGGCTGCGCACGCAGGGGCACTGCAACGCGGAGGAGCAGGATGGAGATCCCTGTGCCTGTGCAGCCGTCTTGGCTGCGCCGCGCCTCGGCCCCGTTGCCCGGACTTTCGGCGCCCGGACGCCTCTTTGACCAGCGCTTCGGCGAGGGGCTGCTGGAGGCCGAGCTGGCTGCGCTCTGCCCCACCACGCTCGCCCCCTACTACCTGCGCGCACCCAGCGTGGCGCTGCCCGTCGCCCAGGTGCCGACGGACCCCGGCCACTTTTCGGTGCTGCTAGACGTGAAGCACTTCTCGCCGGAGGAAATTGCTGTCAAGGTGGTGGGCGAACACGTGGAGGTGCACGCGCGCCACGAGGAGCGCCCGGATGAGCACGGATTCGTCGCGCGCGAGTTCCACCGTCGCTACCGCCTGCCGCCTGGCGTGGATCCGGCTGCCGTGACGTCCGCGCTGTCCCCCGAGGGCGTCCTGTCCATCCAGGCCGCACCAGCGTCGGCCCAGGCCCCACCGCCAGCCGCAGCCAAGTAG

train_neg="../datasets/training/negative/negative_trainingSet_Flank-100.fa"
## >REGION_GT_20000_LT_50000_1:chr1:38501:38901:0:+
## TAGTGTGAGAGAAGAGAGATAAATTGAGAAAGAGACTGGTTTTTAAACTGTTAAAATTGAATCAGGACTTGATGATTTTGAAAATTGTCAGTCTCCCCACATGGAAAAAGATGCTGAAATTAACAAATGGCTTCTGAGCATGTGGCATAGGGTGTAACTGTACAGTCTTTTGTGATTATGCATAAAGATCAAAGGATGGGAGTAGCAATGAGTCACACAGAGGTCTGTTGCAAGAGATTACAAGGGTGTACCATGCAGAACCTCTCCACCAAACCTTAGGGCCCTTGGGAAGCTTCAGTGAGTTACCCTGGGGGCCATCTTGGCAGGAGCTGAAGGTAGAAAGGTAGAGTTTATCTCTAAAAGATTCATGGGTATGGCTCTTGACAAATCGACTATGAGC


print ("----read input Fasta Files----")
train_pos_sequences = read_fasta_file(train_pos, num_tr_data) ##num_tr_data <>0 then return num_tr RANDOM samples.
train_neg_sequences = read_fasta_file(train_neg, num_tr_data)
print ("len(train_pos_sequences)",len(train_pos_sequences))
print ("len(train_neg_sequences)",len(train_neg_sequences))

made_by     = "positive" ## take values the vectors are made only by the positive dataset or by both choices=['positive', 'positive_negative']
overlapping = "non-overlapping" ##help="if the kmers are overlapping choices=['overlapping', 'non-overlapping']"
vector_size = "30" ## length of the kmer embedding vector
file = '../datasets/kmer_embedding/' + made_by + '/' + overlapping + '/' + vector_size + '/training/'

train_x = {}
val_x = {}
val_y = []
train_y = []
sample_dim = {}

start_point = 60
end_point   = 120
k           = ['3,4'] ##what kmers create
k           = ['3'] ## because i use on hot encoding to avoid loop
# print (train_pos_sequences[0])
for i in k:
    file_pos = file + 'positive/' + str(i) + '-mer_emb.txt'
    file_neg = file + 'negative/' + str(i) + '-mer_emb.txt' 
    ##create_training_set In code to change from 1hot to kmers
    train_x[i], train_y, val_x[i], val_y, sample_dim[i] = create_training_set(train_pos_sequences, train_neg_sequences, positions=[start_point, end_point], file_pos=file_pos, file_neg=file_neg, overlapping=overlapping, k=i, split=True)
    print (sample_dim[i])
    print ("len(train_x[i])",len(train_x[i]))
    print ("len(train_y)",len(train_y))
    print ("len(val_x[i])",len(val_x[i]))
    print ("len(val_y)",len(val_y))


print ("----create test data----")
test_pos="../datasets/testing/positive/positive_testingSet_Flank-100.fa"
##>ENSG00000003989|8|P52569|1|17538777|17570561|ENST00000004531
##TGGTAAAGATCCCTACTTATCTATACGATTTAATTACTTTTTTTTCCATCAGTCCCAAAACAGAAAGAGCAGATGTCTCACCACGAAACTAGCAACTGGAATGAAGATAGAAACAAGTGGTTATAACTCAGACAAACTAATTTGTCGAGGGTTTATTGGAACACCTGCCCCACCGGTTTGCGACAGCAAGTTTCTCCTGTCGCCTTCGTCAGACGTCAGAATGATTCCTTGCAGAGCCGCGCTGACCTTTGCCCGATGTCTGATCCGGAGAAAAATCGTGACCCTGGACAGTCTAGAAGACACCAAATTATGCCGCTGCTTATCCACCATGGACCTCATTGCCCTGGGCGTTGGAAGCACCCTTGGGGCCGGGGTTTATGTCCTCGCTGGGGAGGTGGCCAAGGCAGACTCGGGCCCCAGCATCGTGGTGTCCTTCCTCATTGCTGCCCTGGCTTCAGTGATGGCTGGCCTCTGCTATGCCGAATTTGGGGCCCGTGTTCCCAAGACGGGGTCTGCATATTTGTACACCTACGTGACTGTCGGAGAGCTGTGGGCCTTCATCACTGGCTGGAATCTCATTTTATCGTATGTGATAGGTACATCAAGTGTTGCAAGAGCCTGGAGTGGCACCTTTGATGAACTTCTTAGCAAACAGATTGGTCAGTTTTTGAGGACATACTTCAGAATGAATTACACTGGTCTTGCAGAATATCCCGATTTTTTTGCTGTGTGCCTTATATTACTTCTAGCAGGTCTTTTGTCTTTTGGAGTAAAAGAGTCTGCTTGGGTGAATAAAGTCTTCACAGCTGTTAATATTCTCGTCCTTCTGTTTGTGATGGTTGCTGGGTTTGTGAAAGGAAATGTGGCAAACTGGAAGATTAGTGAAGAGTTTCTCAAAAATATATCAGCAAGTGCCAGAGAGCCACCTTCTGAAAACGGAACAAGTATCTATGGGGCTGGTGGCTTTATGCCTTATGGCTTTACGGGAACGTTGGCTGGTGCTGCAACTTGCTTTTATGCCTTTGTGGGATTTGACTGCATTGCAACAACTGGTGAAGAAGTTCGGAATCCCCAGAAAGCTATTCCCATTGGAATTGTGACGTCTTTGCTTGTTTGCTTTATGGCCTATTTTGGGGTCTCTGCAGCTTTAACACTTATGATGCCGTACTACCTCCTCGATGAAAAAAGCCCCCTTCCTGTAGCGTTTGAATATGTGGGATGGGGTCCTGCCAAATATGTCGTCGCAGCTGGTTCTCTCTGCGCCTTGTCAACAAGTCTTCTTGGATCCATTTTCCCAATGCCTCGTGTAATCTATGCTATGGCGGAGGATGGGTTGCTTTTCAAATGTCTAGCTCAAATCAATTCCAAAACGAAGACACCAATAATTGCTACTTTATCATCGGGTGCAGTGGCAGCTTTGATGGCCTTTCTGTTTGACCTGAAGGCGCTTGTGGACATGATGTCCATTGGCACACTCATGGCCTACTCTCTGGTGGCAGCCTGTGTTCTCATCCTCAGGTACCAGCCTGGCTTATCTTACGACCAGCCCAAATGTTCTCCTGAGAAAGATGGTCTGGGATCGTCTCCCAGGGTAACCTCGAAGAGTGAGTCCCAGGTCACCATGCTGCAGAGACAGGGCTTCAGCATGCGGACCCTCTTCTGCCCCTCCCTTCTGCCAACACAGCAGTCAGCTTCTCTCGTGAGCTTTCTGGTAGGATTCCTAGCTTTCCTCGTGTTGGGCCTGAGTGTCTTGACCACTTACGGAGTTCATGCCATCACCAGGCTGGAGGCCTGGAGCCTCGCTCTCCTCGCGCTGTTTCTTGTTCTCTTCGTTGCCATCGTTCTCACCATCTGGAGGCAGCCCCAGAATCAGCAAAAAGTAGCCTTCATGGTTCCATTCTTACCATTTTTGCCAGCGTTCAGCATCTTGGTGAACATTTACTTGATGGTCCAGTTAAGTGCAGACACTTGGGTCAGATTCAGCATTTGGATGGCAATTGGCTTCCTGATTTACTTTTCTTATGGCATTAGACACAGCCTGGAGGGTCATCTGAGAGATGAAAACAATGAAGAAGATGCTTATCCAGACAACGTTCATGCAGCAGCAGAAGAAAAATCTGCCATTCAAGCAAATGACCATCACCCAAGAAATCTCAGTTCACCTTTCATATTCCATGAAAAGACAAGTGAATTCTAA

test_neg="../datasets/testing/negative/negative_testingSet_Flank-100.fa"
##>REGION_GT_50000_1_sorf_1:chr1:102467:102867:0:+
##CATTCTCATATGACAGATTTCAGATGGCATTCTTATTTCCCTGATTTCTTTTTGAGATAGCTTGCATTTCCCTCCTCTATATAAAGCCACCGTTTATCAAATGCCTACATGGACCAAGCAGTCCACAAGGGCTTCACAGACAGTTTTACTAAACTCATGCCAAAACTTTCAGGTTTTATACCTACCTTATAGATAAAGAAATTGAAGCTTATAGAGTTTAAGTAATGTTCCCAAAGCCTCGTGGCTAGTAATTCAAACCTAATTTCTGCCTACTCCAAAGTCTATTTTTCCTTATGATACTCTACTGCCTCTCCATGGATAAAGACAGAGATCACATATTAATAAAATTTGCACAAAGTCGGCAAATTGTTGAAAGGGAAGGCTAAGATGATTAATAAAA

num_te_data=1000
print ("----read input Fasta Files----")

test_pos_sequences = read_fasta_file(test_pos, num_te_data) ##num_tr_data <>0 then return num_tr RANDOM samples.
test_neg_sequences = read_fasta_file(test_neg, num_te_data)
print ("len(test_pos_sequences)",len(test_pos_sequences))
print ("len(test_neg_sequences)",len(test_neg_sequences))


print ("----create_test_set----")

made_by     = "positive" ## take values the vectors are made only by the positive dataset or by both choices=['positive', 'positive_negative']
overlapping = "non-overlapping" ##help="if the kmers are overlapping choices=['overlapping', 'non-overlapping']"
vector_size = "30" ## length of the kmer embedding vector
# file = '../datasets/kmer_embedding/' + made_by + '/' + overlapping + '/' + vector_size + '/testing/' ##use the positive

test_x = {}
# val_x = {}
# val_y = []
# train_y = []


start_point = 60
end_point   = 120
k           = ['3,4'] ##what kmers create
k           = ['3'] ## because i use on hot encoding to avoid loop
# print (test_pos_sequences[0])
for i in k:
    file_pos = file + 'positive/' + str(i) + '-mer_emb.txt'
    file_neg = file + 'negative/' + str(i) + '-mer_emb.txt' 
    ##create_training_set In code to change from 1hot to kmers
    test_x[i], test_y, _ = create_testing_set(test_pos_sequences, test_neg_sequences, positions=[start_point, end_point], file_pos=file_pos, file_neg=file_neg, overlapping=overlapping, k=i)

    print ("len(test_x[i])",len(test_x[i]))
    print ("len(test_y)",len(test_y))

print ("----train model----")


best_params = {}
results = []


vector_size="30"
vector_size = "vector_size_"+ vector_size

overlapping = 'non-overlapping' ##choices=['overlapping', 'non-overlapping'], help="if the kmers are overlapping")
made_by ='positive'             ## choices=['positive', 'positive_negative'], 
model_type='cnn'      ## choices=['cnn', 'gated_cnn', 'hybrid'], help="if the kmers are overlapping")
params_tuning="no" ##choices=['yes', 'no'], default='no', help="Enables the hyperparameter tuning") if yes the member must be given as list of values eg params['lr'] =[0.001,0.002]

params = {'lr': [], 'k': [], 'batch_size': [], 'kernel_size': [], 'layers':[], 'sample_dim':[], 'mntm':[], 'epochs': [], 'flt': [], 'branch': []}


params_tuning == 'no'
params['lr'] = 0.001
params['batch_size'] = 64 
params['kernel_size'] = 5          
params['layers'] = 15  
params['epochs'] = 30 
params['flt'] = 25
params['branch'] = 3
params['k'] = ['3']
params['sample_dim'] = sample_dim   




model, best_params, results = train_model(model_type, train_x, train_y, val_x, val_y, test_x, test_y, params, params_tuning)
print ("----END----")

