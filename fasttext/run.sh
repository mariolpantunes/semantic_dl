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
    -d|--dim)
      DIM="$2"
      shift # past argument
      shift # past value
      ;;
    -w|--window_size)
      WS="$2"
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

QUERIES="${RESULTDIR}"/queries.txt
THREADS=4

mkdir -p "${RESULTDIR}"

if [ $OPTION -eq 1 ]
then

echo "Training Model"

python generate_queries.py -p $TESTDIR -d $QUERIES

./fasttext skipgram -input $CORPUS -output "${RESULTDIR}"/model -dim $DIM -ws $WS -minCount 1 -thread $THREADS

cat $QUERIES | ./fasttext print-word-vectors "${RESULTDIR}"/model.bin > "${RESULTDIR}"/vectors.txt

echo "Evaluating Model"
python eval.py -m "${RESULTDIR}"/model.vec -p $TESTDIR -d "${RESULTDIR}"/results.txt

fi

if [ $OPTION -eq 2 ]
then

echo "Evaluating Model"
python eval.py -m ./pre-trained/pretrained.vec -p $TESTDIR -d "${RESULTDIR}"/results.txt

fi