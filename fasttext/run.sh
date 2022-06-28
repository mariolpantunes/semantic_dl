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
THREADS=10
NANOSECONDS=1000000000

mkdir -p "${RESULTDIR}"

if [ $OPTION -eq 1 ]; then

echo "Training Model"

start=`date +%s%N`
./fasttext skipgram -input $CORPUS -output "${RESULTDIR}"/model -dim $DIM -ws $WS -minCount 1 -thread $THREADS
end=`date +%s%N`
time=`expr $end - $start`

echo "Time to train the model:  $(echo "scale=5; $time/$NANOSECONDS" | bc -l )" >> "${RESULTDIR}"/results.txt

echo "Evaluating Model"

start=`date +%s%N`
python generate_queries.py -p $TESTDIR -d $QUERIES
end=`date +%s%N`
time=`expr $end - $start`

echo "Time to preprocess testing data: $(echo "scale=5; $time/$NANOSECONDS" | bc -l )" >> "${RESULTDIR}"/results.txt

start=`date +%s%N`
cat $QUERIES | ./fasttext print-word-vectors "${RESULTDIR}"/model.bin > "${RESULTDIR}"/vectors.txt
end=`date +%s%N`
time=`expr $end - $start`

echo "Time to generate queries: $(echo "scale=5; $time/$NANOSECONDS" | bc -l )" >> "${RESULTDIR}"/results.txt

python eval.py -m "${RESULTDIR}"/model.vec -p $TESTDIR -d "${RESULTDIR}"/results.txt

fi

if [ $OPTION -eq 2 ]; then

echo "Evaluating Model"
python eval.py -m ./pre-trained/pretrained.vec -p $TESTDIR -d "${RESULTDIR}"/results.txt

fi

if [ $OPTION -eq 3 ]; then

echo "Training Model"

start=`date +%s%N`
./fasttext skipgram -input $CORPUS -output "${RESULTDIR}"/model -dim 300 -minCount 1 -thread $THREADS -pretrainedVectors ./pre-trained/pretrained.vec
end=`date +%s%N`
time=`expr $end - $start`

echo "Time to train the model: $(echo "scale=5; $time/$NANOSECONDS" | bc -l )" >> "${RESULTDIR}"/results.txt

echo "Evaluating Model"

start=`date +%s%N`
python generate_queries.py -p $TESTDIR -d $QUERIES
end=`date +%s%N`
time=`expr $end - $start`

echo "Time to preprocess testing data: $(echo "scale=5; $time/$NANOSECONDS" | bc -l )" >> "${RESULTDIR}"/results.txt

start=`date +%s%N`
cat $QUERIES | ./fasttext print-word-vectors "${RESULTDIR}"/model.bin > "${RESULTDIR}"/vectors.txt
end=`date +%s%N`
time=`expr $end - $start`

echo "Time to generate queries: $(echo "scale=5; $time/$NANOSECONDS" | bc -l )" >> "${RESULTDIR}"/results.txt

python eval.py -m "${RESULTDIR}"/model.vec -p $TESTDIR -d "${RESULTDIR}"/results.txt

fi