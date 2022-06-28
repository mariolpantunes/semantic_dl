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
    -o|--option)
      OPTION="$2"
      shift # past argument
      shift # past value
      ;;
  esac
done

mkdir -p "${RESULTDIR}"/pretrain/
mkdir -p "${RESULTDIR}"/model/

if [ $OPTION -eq 1 ]
then

echo "Generating tok2vec vectors"
python -m spacy pretrain $CONFIGPATH "${RESULTDIR}"/pretrain/ --paths.raw_text $CORPUS

echo "Building model with no further training"
python -m spacy assemble $CONFIGPATH "${RESULTDIR}"/model/ --paths.init_tok2vec "${RESULTDIR}"/pretrain/model19.bin

echo "Evaluating the model with the tok2vec pretraining"
python eval.py -m "${RESULTDIR}"/model/ -p $TESTDIR -d "${RESULTDIR}"/results.txt

fi

if [ $OPTION -eq 2 ]
then

echo "Evaluating the model with the tok2vec pretraining"
python eval.py -m en_core_web_lg -p $TESTDIR -d "${RESULTDIR}"/results.txt

fi