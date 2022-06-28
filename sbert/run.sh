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
    -p|--path)
      TRAINPATH="$2"
      shift # past argument
      shift # past value
      ;;
    -c|--corpus)
      CORPUS="$2"
      shift # past argument
      shift # past value
      ;;
    -d|--devcorpus)
      DEVCORPUS="$2"
      shift # past argument
      shift # past value
      ;;
    -v|--vector)
      VECTOR_SIZE="$2"
      shift # past argument
      shift # past value
      ;;
    -s|--sup_train)
      SUP_TRAIN="$2"
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

mkdir -p "${RESULTDIR}"

if [ $OPTION -eq 1 ]
then

mkdir -p "${RESULTDIR}"/model
mkdir -p "${RESULTDIR}"/pretrain

echo "Pretraining the model"

start=`date +%s%N`
python from_scratch/pre_train.py -v $VECTOR_SIZE -p $TRAINPATH -m "${RESULTDIR}"/pretrain/ -t $CORPUS -d $DEVCORPUS
end=`date +%s%N`
time=`expr $end - $start`

echo "Pretrain:  $(echo "scale=5; $time/$NANOSECONDS" | bc -l )" >> "${RESULTDIR}"/results.txt

echo "Training the model"
start=`date +%s%N`
python train.py -d "${RESULTDIR}"/model/ -m "${RESULTDIR}"/pretrain/ -t $SUP_TRAIN
end=`date +%s%N`
time=`expr $end - $start`

echo "Train:  $(echo "scale=5; $time/$NANOSECONDS" | bc -l )" >> "${RESULTDIR}"/results.txt

echo "Evaluating the model"
python eval.py -m "${RESULTDIR}"/model/ -p $TESTDIR -d "${RESULTDIR}"/results.txt

fi

if [ $OPTION -eq 2 ]
then

python eval.py -m roberta-base -p $TESTDIR -d "${RESULTDIR}"/results.txt

fi


if [ $OPTION -eq 3 ]
then

mkdir -p "${RESULTDIR}"/model
mkdir -p "${RESULTDIR}"/pretrain


start=`date +%s%N`
python pre-trained/pre_train.py -n roberta-base -m "${RESULTDIR}"/pretrain/ -t $CORPUS -d $DEVCORPUS
end=`date +%s%N`
time=`expr $end - $start`

echo "Pretrain:  $(echo "scale=5; $time/$NANOSECONDS" | bc -l )" >> "${RESULTDIR}"/results.txt

start=`date +%s%N`
python train.py -d "${RESULTDIR}"/model/ -m "${RESULTDIR}"/pretrain/ -t $SUP_TRAIN
end=`date +%s%N`
time=`expr $end - $start`

echo "Train:  $(echo "scale=5; $time/$NANOSECONDS" | bc -l )" >> "${RESULTDIR}"/results.txt

python eval.py -m "${RESULTDIR}"/model/ -p $TESTDIR -d "${RESULTDIR}"/results.txt

fi