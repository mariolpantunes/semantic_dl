#!/bin/bash

set -e

while [[ $# -gt 0 ]]; do
  key="$1"

  case $key in
    -r|--resultdir)
      RESULTDIR="$2"
      shift # past argument
      shift # past value
      ;;
    -t|--testdir)
      TESTDIR="$2"
      shift # past argument
      shift # past value
      ;;
    -c|--corpus)
      CORPUS="$2"
      shift # past argument
      shift # past value
      ;;
    --config)
      CONFIGPATH="$2"
      shift # past argument
      shift # past value
      ;;
  esac
done

mkdir -p "${RESULTDIR}"/pretrain/
mkdir -p "${RESULTDIR}"/model/

echo "Generating tok2vec vectors"
#python -m spacy pretrain $CONFIGPATH "${RESULTDIR}"/pretrain/ --paths.raw_text $CORPUS

echo "Building model with no further training"
python -m spacy assemble $CONFIGPATH "${RESULTDIR}"/model/ --paths.init_tok2vec "${RESULTDIR}"/pretrain/model800.bin

echo "Evaluating the model with the tok2vec pretraining"
python eval.py -m "${RESULTDIR}"/model/ -p $TESTDIR -d "${RESULTDIR}"/results.txt