#!/bin/bash


AGR_CORPUS=dataset/aggregated_corpus
PROC_SETENCES=dataset/processed_setences.json
SPACY_CORPUS=dataset/aggregated_corpus.jsonl
BERT_CORPUS=dataset/train_corpus_bert.txt
BERT_DEV_CORPUS=dataset/test_corpus_bert.txt
SUP_TRAIN=dataset/train_sim/rg65.csv
TESTDIR=dataset/test/
TRAINDIR=dataset/train/

VECTOR_SIZES=(43 50 78 88 112 125)
WINDOW_SIZES=(3 3 5 5 7 7)
BERT_VECTORS=(36 48 72 84 108 120)

RESULTDIR=experiment_results

mkdir -p "${RESULTDIR}"

if hash python 2>/dev/null; then
    PYTHON=python
else
    PYTHON=python3
fi

if [ ! -e $AGR_CORPUS ]; then
    echo "Generating Aggregated Corpus for fastText and GloVe"
    python utils/pre_process_glove.py --dataset $TRAINDIR --destFile $AGR_CORPUS
fi

if [ ! -e $PROC_SETENCES ]; then
    echo "Generating Aggregated Corpus for Word2Vec and TF-IDF"
    python utils/pre_process_w2v.py --dataset $TRAINDIR --destFile $PROC_SETENCES
fi
if [ ! -e $SPACY_CORPUS ]; then
    echo "Generating Aggregated Corpus for Spacy"
    python utils/pre_process_spacy.py --dataset $TRAINDIR --destFile $SPACY_CORPUS
fi

echo "Running models trained from scratch"

for i in {0..0}; do # {1..10} 
    echo ""
    echo "Running Experiment $(( $i + 1 )) of 6"
    echo ""

    #echo "Running fastText"
    #(cd fasttext && ./run.sh -r ../"${RESULTDIR}"/fasttext/"${VECTOR_SIZES[i]}"_"${WINDOW_SIZES[i]}"/ -t ../$TESTDIR -c ../$AGR_CORPUS -d ${VECTOR_SIZES[i]} -w ${WINDOW_SIZES[i]} -o 1)

    #echo "Running glove"
    #(cd glove && ./run.sh -r ../"${RESULTDIR}"/glove/"${VECTOR_SIZES[i]}"_"${WINDOW_SIZES[i]}"/ -t ../$TESTDIR -c ../$AGR_CORPUS -d ${VECTOR_SIZES[i]} -w ${WINDOW_SIZES[i]} -o 1)

    #echo "Running word2vec"
    #(cd word2vec && ./run.sh -r ../"${RESULTDIR}"/word2vec/"${VECTOR_SIZES[i]}"_"${WINDOW_SIZES[i]}"/ -t ../$TESTDIR -c ../$PROC_SETENCES -d ${VECTOR_SIZES[i]} -w ${WINDOW_SIZES[i]} -o 1)

    #echo "Running tf-idf"
    #(cd tf-idf && ./run.sh -r ../"${RESULTDIR}"/tf-idf/"${VECTOR_SIZES[i]}"_"${WINDOW_SIZES[i]}"/ -t ../$TESTDIR -c ../$PROC_SETENCES -n ${VECTOR_SIZES[i]} -o 1)

    #echo "Running spacy"
    #(cd spacy && ./run.sh -r ../"${RESULTDIR}"/spacy/"${VECTOR_SIZES[i]}"_"${WINDOW_SIZES[i]}" -t ../$TESTDIR -c ../$SPACY_CORPUS --config configs/config_"${VECTOR_SIZES[i]}"_"${WINDOW_SIZES[i]}"-cfg -o 1)

    echo "Running bert"
    (cd sbert && ./run.sh -r ../"${RESULTDIR}"/bert/"${BERT_VECTORS[i]}" -t ../"${TESTDIR}" -c ../"${BERT_CORPUS}" -p ../"${TRAINDIR}" -d ../"${BERT_DEV_CORPUS}" -v ${BERT_VECTORS[i]} -s ../"${SUP_TRAIN}" -o 1)
done

echo "Running pre-trained models"

echo "Running bert"
(cd sbert && ./run.sh -r ../"${RESULTDIR}"/bert/pretrained -t ../"${TESTDIR}" -o 2)

echo "Running spacy"
(cd spacy && ./run.sh -r ../"${RESULTDIR}"/spacy/pretrained -t ../"${TESTDIR}" -o 2)

echo "Running fastText"
(cd fasttext && ./run.sh -r ../"${RESULTDIR}"/fasttext/pretrained -t ../"${TESTDIR}" -o 2)

echo "Running glove"
(cd glove && ./run.sh -r ../"${RESULTDIR}"/glove/pretrained -t ../"${TESTDIR}" -o 2)

echo "Running word2vec"
(cd word2vec && ./run.sh -r ../"${RESULTDIR}"/word2vec/pretrained -t ../$TESTDIR -o 2)


echo "Running online models"

echo "Running bert"
(cd sbert && ./run.sh -r ../"${RESULTDIR}"/bert/online/ -t ../"${TESTDIR}" -c ../"${BERT_CORPUS}" -d ../"${BERT_DEV_CORPUS}" -v ${BERT_VECTORS[i]} -s ../"${SUP_TRAIN}" -o 3)
    


