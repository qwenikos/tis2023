import numpy as np
import sys
import json
import os
import argparse
from misc import read_fasta_file, create_training_set, create_testing_set
from train import train_cnn, train_gated_cnn, train_model
import tensorflow as tf
import random as rn
import warnings

warnings.filterwarnings("ignore")

# fix random seed for reproducibility
seed = 2023
tf.random.set_seed(seed)

np.random.seed(20)

# Necessary for starting core Python generated random numbers in a well-defined state.
rn.seed(1984)

def print_model(s):
    
    file = '../results/' + str(os.getpid()) + '.txt'

    with open(file,'a') as f:
        print(s, file=f)


def main():

    parser = argparse.ArgumentParser(description="""Let us create a user contact.""")

    parser.add_argument("--params_tuning", choices=['yes', 'no'], default='no', help="Enables the hyperparameter tuning")

    parser.add_argument("--lr", default=0.001, help="learning rate")
    parser.add_argument("--batch_size", default=64, help="Batch size")
    parser.add_argument("--kernel_size", default=5, help="Size of the kernel")
    parser.add_argument("--layers", default=15, help="Number of gated convolution layers")
    parser.add_argument("--epochs", default=30, help="Number of epochs")
    parser.add_argument("--flt", default=25, help="Number of filters")
    parser.add_argument("--branch", default='3', help="Number of branches of the model")
    parser.add_argument("--vector_size", default='30', help="length of the kmer embedding vector")
    parser.add_argument("--overlapping", default='non-overlapping', choices=['overlapping', 'non-overlapping'], help="if the kmers are overlapping")
    parser.add_argument("--made_by", default='positive', choices=['positive', 'positive_negative'], help="the vectors are made only by the positive dataset or by both, negative and positive")
    parser.add_argument("--train_pos", default='../datasets/training/positive/positive_trainingSet_Flank-100.fa', help="if the kmers are overlapping")
    parser.add_argument("--train_neg", default='../datasets/training/negative/negative_trainingSet_Flank-100.fa', help="if the kmers are overlapping")
    parser.add_argument("--test_pos", default='../datasets/testing/positive/positive_testingSet_Flank-100.fa', help="if the kmers are overlapping")
    parser.add_argument("--test_neg", default='../datasets/testing/negative/negative_testingSet_Flank-100.fa', help="if the kmers are overlapping")
    parser.add_argument("--type", default='hybrid', choices=['cnn', 'gated_cnn', 'hybrid'], help="if the kmers are overlapping")
    parser.add_argument("--num_tr_data", default=0, help="Number of filters")
    parser.add_argument("--num_te_data", default=0, help="Number of filters")


    parser.add_argument("--subregion", default='60,120', help="Starting and finishing positions of the selected subregion of a gene")
    parser.add_argument("--k", default='3,4', help="k value for k-mers")
    
    args = parser.parse_args()

    params_tuning = args.params_tuning
    model_type = args.type

    num_tr_data = int(args.num_tr_data)
    num_te_data = int(args.num_te_data)

    train_pos = args.train_pos
    train_neg = args.train_neg
    test_pos = args.test_pos
    test_neg = args.test_neg

    subregion = args.subregion.split(",")

    if len(subregion) < 2 or subregion[1] == '':
        sys.exit('The given arguments for the flag --subregion are incorrect')
    else:
        start_point = int(subregion[0])
        end_point = int(subregion[1])

    params = {'lr': [], 'k': [], 'batch_size': [], 'kernel_size': [], 'layers':[], 'sample_dim':[], 'mntm':[], 'epochs': [], 'flt': [], 'branch': []}
    
    k = [int(i) for i in args.k.split(",")]
    params['k'] = k

    vector_size = 'vector_size_' + args.vector_size
    overlapping = args.overlapping
    made_by = args.made_by

    if params_tuning == 'yes':
        if type(args.lr) == str:
            params['lr'] = [float(i) for i in args.lr.split(",")]
        else:
            params['lr'] = [args.lr]

        if type(args.batch_size) == str:
            params['batch_size'] = [int(i) for i in args.batch_size.split(",")]
        else:
            params['batch_size'] = args.batch_size  

        if type(args.kernel_size) == str:
            params['kernel_size'] = [int(i) for i in args.kernel_size.split(",")]
        else:
            params['kernel_size'] = args.kernel_size          

        if type(args.layers) == str:
            params['layers'] = [int(i) for i in args.layers.split(",")]
        else:
            params['layers'] = args.layers  

        if type(args.epochs) == str:
            params['epochs'] = [int(i) for i in args.epochs.split(",")]
        else:
            params['epochs'] = args.epochs  

        if type(args.flt) == str:
            params['flt'] = [int(i) for i in args.flt.split(",")]
        else:
            params['flt'] = args.flt
        
        params['branch'] = [int(args.branch)]
        params['k'] = k
         
    else:
        params['lr'] = float(args.lr)       
        params['batch_size'] = int(args.batch_size) 
        params['kernel_size'] = int(args.kernel_size)        
        params['layers'] = int(args.layers)  
        params['epochs'] = int(args.epochs)  
        params['flt'] = int(args.flt)
        params['branch'] = int(args.branch)


    train_pos_sequences = read_fasta_file(train_pos, num_tr_data)
    train_neg_sequences = read_fasta_file(train_neg, num_tr_data)


    #tha exw ena for to opoio tha dimioyrgei sets analoga me to k value kai tha apothikevontai ola sto train_x dictionary

    train_x = {}
    val_x = {}
    val_y = []
    train_y = []
    sample_dim = {}
    file = '../datasets/kmer_embedding/' + made_by + '/' + overlapping + '/' + vector_size + '/training/'

    if num_tr_data == 0:
        for i in k:
            file_pos = file + 'positive/' + str(i) + '-mer_emb.txt'
            file_neg = file + 'negative/' + str(i) + '-mer_emb.txt' 

            train_x[i], train_y, val_x[i], val_y, sample_dim[i] = create_training_set(train_pos_sequences, train_neg_sequences, positions=[start_point, end_point], file_pos=file_pos, file_neg=file_neg, overlapping=overlapping, k=i, split=True)

    else:
        for i in k:
            file_pos = file + 'positive/' + str(i) + '-mer_emb.txt'
            file_neg = file + 'negative/' + str(i) + '-mer_emb.txt' 

            train_x[i], train_y, val_x[i], val_y, sample_dim[i] = create_training_set(train_pos_sequences, train_neg_sequences, positions=[start_point, end_point], file_pos=file_pos, file_neg=file_neg, overlapping=overlapping, k=i, split=True)
        
    # if params_tuning == 'yes':
    #     params['sample_dim'] = [sample_dim]
    # else:
    params['sample_dim'] = sample_dim

    test_pos_sequences = read_fasta_file(test_pos, num_te_data)


    test_neg_sequences = read_fasta_file(test_neg, num_te_data)

    test_x = {}
    # file = '../datasets/kmer_embedding/' + made_by + '/' + overlapping + '/' + vector_size + '/testing/'

    if num_te_data == 0:
        for i in k:
            file_pos = file + 'positive/' + str(i) + '-mer_emb.txt'
            file_neg = file + 'negative/' + str(i) + '-mer_emb.txt'


            test_x[i], test_y, _ = create_testing_set(test_pos_sequences, test_neg_sequences, positions=[start_point, end_point], file_pos=file_pos, file_neg=file_neg, overlapping=overlapping, k=i)

    else: 
        for i in k:
            file_pos = file + 'positive/' + str(i) + '-mer_emb.txt'
            file_neg = file + 'negative/' + str(i) + '-mer_emb.txt'


            test_x[i], test_y, _ = create_testing_set(test_pos_sequences[0:num_te_data], test_neg_sequences[0:num_te_data], positions=[start_point, end_point], file_pos=file_pos, file_neg=file_neg, overlapping=overlapping, k=i)

    best_params = {}
    results = []
    
    model, best_params, results = train_model(model_type, train_x, train_y, val_x, val_y, test_x, test_y, params, params_tuning)

    print('Name of program: ', os.getpid(), '\n')
    file = '../results/' + str(os.getpid()) + '.txt'

    # if os.path.isfile(file):
	# 	os.remove(file)

    f = open(file,'w')

    f.write('Given parameters')

    f.write('\n')

    f.write(json.dumps(params))
    # f.write('%s:%s\n' % (key, str(value)) for key, value in params)

    f.write('\n\n')

    f.write('Best parameters')

    f.write('\n')

    f.write(json.dumps(best_params))

    f.write('\n\n')

    f.write('Results')

    f.write('\n')

    f.write('loss = %s, accuracy = %s\n' % (str(results[0]), str(results[1])))

    f.write('Model_type: %s, Subregion: %s, Num_of_tr: %s, Num_of_te: %s, Vector_size: %s,  Overlapping: %s, made_by: %s \n\n' % (model_type, subregion, num_tr_data, num_te_data, vector_size, overlapping, made_by))

    f.write('Model Architecture\n')

    f.close()

    model.summary(print_fn=print_model)



if __name__ == '__main__':
	main()

