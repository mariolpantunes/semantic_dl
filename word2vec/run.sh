#!/bin/bash

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
    -d|--dim)
      VECTOR_SIZE="$2"
      shift # past argument
      shift # past value
      ;;
    -w|--window_size)
      WINDOW_SIZE="$2"
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

mkdir -p "${RESULTDIR}"

python word2vec.py --train_input $CORPUS --test_input $TESTDIR --outputFolder $RESULTDIR --vector_size $VECTOR_SIZE --window_size $WINDOW_SIZE

if [ $OPTION -eq 1 ]
then

echo "Training model from scratch"
python train.py --train_input $CORPUS --outputFolder "${RESULTDIR}"/model/ --vector_size $VECTOR_SIZE --window_size $WINDOW_SIZE

echo "Evaluating model"
python eval.py --test_input $TESTDIR --outputFolder "${RESULTDIR}"/results.txt -w "${RESULTDIR}"/model/w2v.model

fi

if [ $OPTION -eq 2 ]

echo "Evaluating pretrained model"
python eval.py --test_input $TESTDIR --outputFolder "${RESULTDIR}"/results.txt -w pretrained

fi
