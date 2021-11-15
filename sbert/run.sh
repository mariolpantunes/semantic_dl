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

mkdir -p "${RESULTDIR}"

if [ $OPTION -eq 1 ]
then

mkdir -p "${RESULTDIR}"/model
mkdir -p "${RESULTDIR}"/pretrain

python from_scratch/pre_train.py -v $VECTOR_SIZE -p $TRAINPATH -m "${RESULTDIR}"/pretrain/ -t $CORPUS -d $DEVCORPUS

python train.py -d "${RESULTDIR}"/model/ -m "${RESULTDIR}"/pretrain/ -t $SUP_TRAIN

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

python pre-trained/pre_train.py -n roberta-base -m "${RESULTDIR}"/pretrain/ -t $CORPUS -d $DEVCORPUS

python train.py -d "${RESULTDIR}"/model/ -m "${RESULTDIR}"/pretrain/ -t $SUP_TRAIN

python eval.py -m "${RESULTDIR}"/model/ -p $TESTDIR -d "${RESULTDIR}"/results.txt

fi