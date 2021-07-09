#!/bin/bash

echo "Generating tok2vec vectors"
python -m spacy pretrain config.cfg ./tok2vec_vectors --paths.raw_text ../dataset/aggregated_corpus.jsonl

echo "Building model with no further training"
python -m spacy assemble config.cfg ./model --paths.init_tok2vec ./tok2vec_vectors/model0.bin

echo "Evaluating the model with the tok2vec pretraining"
python eval.py -m "${RESULTDIR}"/vectors.txt -p $TESTDIR -d "${RESULTDIR}"/results.txt