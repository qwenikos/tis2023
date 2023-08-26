from tensorflow.keras.models import load_model
from misc import read_fasta_file, create_testing_set
import os 
import numpy as np

def test_model():

    test_pos='../datasets/testing/positive/positive_testingSet_Flank-100.fa'
    test_neg='../datasets/testing/negative/negative_testingSet_Flank-100.fa'
    
    test_pos_sequences = read_fasta_file(test_pos)
    test_neg_sequences = read_fasta_file(test_neg)

    test_x = {}
    file = '../datasets/kmer_embedding/positive/non-overlapping/vector_size_50/testing/'

    k = [3,4,5]

    for i in k:
        file_pos = file + 'positive/' + str(i) + '-mer_emb.txt'
        file_neg = file + 'negative/' + str(i) + '-mer_emb.txt'


        test_x[i], test_y, _ = create_testing_set(test_pos_sequences, test_neg_sequences, positions=[60,120], file_pos=file_pos, file_neg=file_neg, overlapping='non-overlapping', k=i)

    
    model = load_model('results/CNNonRaw.hdf5')

    prediction_scores = model.predict(test_x)
    
    fh_predictions = open(str(os.getpid())+'.predictions.txt', 'w')
    prediction_scores = model.predict(test_x)
    np.savetxt(fh_predictions, prediction_scores, delimiter="\t")
    fh_predictions.close()

    




if __name__ == '__main__':
	test_model()




# model_type='hybrid'
# sample_dim=[60,120]
# kernel_size=3
# flt=50
# layers=3
# lr=0.01
# k=[3,4,5]

# model = create_model(model_type, sample_dim, kernel_size, flt, layers, lr, k)



# loaded_model = load_model(file)

# loaded_model.evaluate

# tresults = best_cnn.evaluate(test_x, test_y,
#                             batch_size = batch_size,
#                             verbose = 1,
#                             sample_weight = None)

# print(tresults)

    # return [tresults]