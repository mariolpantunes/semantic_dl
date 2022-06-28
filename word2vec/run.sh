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
NANOSECONDS=1000000000

mkdir -p "${RESULTDIR}"/model

if [ $OPTION -eq 1 ]
then

echo "Training model from scratch"
start=`date +%s%N`
python train.py --train_input $CORPUS --outputFolder "${RESULTDIR}"/model/ --vector_size $VECTOR_SIZE --window_size $WINDOW_SIZE
end=`date +%s%N`
time=`expr $end - $start`

echo "Train:  $(echo "scale=5; $time/$NANOSECONDS" | bc -l )" >> "${RESULTDIR}"/results.txt

echo "Evaluating model"
python eval.py --test_input $TESTDIR --outputFile "${RESULTDIR}"/results.txt -w "${RESULTDIR}"/model/w2v.model

fi

if [ $OPTION -eq 2 ]
then

echo "Evaluating pretrained model"
python eval.py --test_input $TESTDIR --outputFile "${RESULTDIR}"/results.txt -w pretrained

fi
