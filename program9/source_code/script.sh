#!/bin/bash

FUNCTION=corpus_text                  # train or corpus_text or test
VECTOR_SIZE=50                  # lenght of the k-mers vectors
OVERLAPPING=non-overlapping         # overlapping or non-overlapping
DATASET=training                # training or testing
MADE_BY=positive                # kmers are made only by the positive datasets or by the positive and negative: positive or positive_negative

SUBREGION=60,150
K=3                             # k value of k-mers: 

if [ "$FUNCTION" == "create_corpus" ];
then

    file_pos="../datasets/kmer_embedding/${MADE_BY}/${OVERLAPPING}/vector_size_${VECTOR_SIZE}/"
    file_neg="../datasets/kmer_embedding/${MADE_BY}/${OVERLAPPING}/vector_size_${VECTOR_SIZE}/"

    if [ ! -d $file_pos ]; then
        mkdir ${file_pos}
    fi

    file_pos+="${DATASET}/"
    file_neg+="${DATASET}/"

    if [ ! -d $file_pos ]; then
        mkdir ${file_pos}
    fi

    file_pos+="positive/"
    file_neg+="negative/"

    if [ ! -d $file_pos ]; then
        mkdir ${file_pos}
    fi

    if [ ! -d $file_neg ]; then
        mkdir ${file_neg}
    fi

    file_pos+="/${K}-mer_emb.txt"
    file_neg+="/${K}-mer_emb.txt"

    if [ -f $file_pos ]; then
        rm ${file_pos}
    fi

    if [ -f $file_neg ]; then
        rm ${file_neg}
    fi

    touch ${file_pos}
    touch ${file_neg}


    file_pos="../program9/datasets/kmer_embedding/${MADE_BY}/${OVERLAPPING}/vector_size_${VECTOR_SIZE}/${DATASET}/"
    file_neg=$file_pos
    

    file_pos+="positive/${K}-mer_emb.txt"
    file_neg+="negative/${K}-mer_emb.txt"


    if [ "$MADE_BY" == "positive" ];
    then
        file="../datasets/${DATASET}/positive/positive_${DATASET}Set_Flank-100.fa"

        python3 create_corpus.py --k ${K} --overlapping ${OVERLAPPING} --file ${file} --subregion ${SUBREGION}

        mv corpus_text.txt ../../glove/ && cd ../../glove

        make clean && make && ./demo.sh && cp vectors.txt ${file_pos} && cp vectors.txt ${file_neg} && rm vectors.txt

        cd ../program9/source_code/


    else
        file="../datasets/${DATASET}/positive/positive_${DATASET}Set_Flank-100.fa"

        python3 create_corpus.py --k ${K} --overlapping ${OVERLAPPING} --file ${file} --subregion ${SUBREGION}

        mv corpus_text.txt ../../glove/ && cd ../../glove

        make clean && make && ./demo.sh && mv vectors.txt ${file_pos}

        cd ../program9/source_code/

        file="../datasets/${DATASET}/negative/negative_${DATASET}Set_Flank-100.fa"

        python3 create_corpus.py --k ${K} --overlapping ${OVERLAPPING} --file ${file} --subregion ${SUBREGION}

        mv corpus_text.txt ../../glove/ && cd ../../glove

        make clean && make && ./demo.sh && mv vectors.txt ${}
    
    fi
