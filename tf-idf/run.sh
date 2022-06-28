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
    -n|--n_topics)
      NTOPICS="$2"
      shift # past argument
      shift # past value
      ;;
  esac
done
NANOSECONDS=1000000000

mkdir -p "${RESULTDIR}"


start=`date +%s%N`
python train.py --train_input $CORPUS -o $RESULTDIR -n $NTOPICS
end=`date +%s%N`
time=`expr $end - $start`

echo "Train:  $(echo "scale=5; $time/$NANOSECONDS" | bc -l )" >> "${RESULTDIR}"/results.txt

python eval.py --train_input $CORPUS --test_input $TESTDIR -o $RESULTDIR -l "${RESULTDIR}"/lsi.model -t "${RESULTDIR}"/tf-idf.model
