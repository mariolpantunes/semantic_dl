#!/bin/bash


AGR_CORPUS=dataset/aggregated_corpus
PROC_SETENCES=dataset/processed_setences.json
TESTDIR=dataset/test/
TRAINDIR=dataset/train/

VECTOR_SIZES=(43, 50, 78, 88, 112, 125)
WINDOW_SIZES=(3, 3, 5, 5, 7, 7)

RESULTDIR=experiment_results

mkdir -p "${RESULTDIR}"

if [ ! -e $AGR_CORPUS ]; then
    echo "Generating Aggregated Corpus for fastText and GloVe"
    python utils/pre_process_glove.py --dataset $TRAINDIR --destFile $AGR_CORPUS
fi

if [ ! -e $PROC_SETENCES ]; then
    echo "Generating Aggregated Corpus for Word2Vec and TF-IDF"
    python utils/pre_process_w2v.py --dataset $TRAINDIR --destFile $PROC_SETENCES
fi

for i in {0..2}; do # {1..10} 
    echo "\n\n Running Experiment $i of 6 \n \n"

    #echo "Running fastText"
    #(cd fasttext && ./run.sh -r ../"${RESULTDIR}"/fasttext/"${VECTOR_SIZES[i]}"_"${WINDOW_SIZES[i]}"/ -t ../$TESTDIR -c ../$AGR_CORPUS -d ${VECTOR_SIZES[i]} -w ${WINDOW_SIZES[i]})

    echo "Running glove"
    (cd glove && ./run.sh -r ../"${RESULTDIR}"/glove/"${VECTOR_SIZES[i]}"_"${WINDOW_SIZES[i]}"/ -t ../$TESTDIR -c ../$AGR_CORPUS -d ${VECTOR_SIZES[i]} -w ${WINDOW_SIZES[i]})

    #echo "Running word2vec"
    #(cd word2vec && ./run.sh -r ../"${RESULTDIR}"/word2vec/"${VECTOR_SIZES[i]}"_"${WINDOW_SIZES[i]}"/ -t ../$TESTDIR -c ../$PROC_SETENCES -d ${VECTOR_SIZES[i]} -w ${WINDOW_SIZES[i]})
#
    #echo "Running tf-idf"
    #(cd tf-idf && ./run.sh -r ../"${RESULTDIR}"/fasttext/"${VECTOR_SIZES[i]}"_"${WINDOW_SIZES[i]}"/ -t ../$TESTDIR -c ../$PROC_SETENCES -n ${VECTOR_SIZES[i]})
done


