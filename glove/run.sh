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
  esac
done

mkdir -p "${RESULTDIR}"

NUM_THREADS=4

VOCAB_FILE="${RESULTDIR}"vocab.txt
COOCCURRENCE_FILE="${RESULTDIR}"cooccurrence.bin
COOCCURRENCE_SHUF_FILE="${RESULTDIR}"cooccurrence.shuf.bin
SAVE_FILE="${RESULTDIR}"vectors

VERBOSE=2
MEMORY=4.0
VOCAB_MIN_COUNT=1
MAX_ITER=20
BINARY=2
X_MAX=10

echo "$ ./vocab_count -min-count $VOCAB_MIN_COUNT -verbose $VERBOSE < $CORPUS > $VOCAB_FILE"
./vocab_count -min-count $VOCAB_MIN_COUNT -verbose $VERBOSE < $CORPUS > $VOCAB_FILE

echo "$ ./cooccur -memory $MEMORY -vocab-file $VOCAB_FILE -verbose $VERBOSE -window-size $WINDOW_SIZE < $CORPUS > $COOCCURRENCE_FILE"
./cooccur -memory $MEMORY -vocab-file $VOCAB_FILE -verbose $VERBOSE -window-size $WINDOW_SIZE < $CORPUS > $COOCCURRENCE_FILE

echo "$ ./shuffle -memory $MEMORY -verbose $VERBOSE < $COOCCURRENCE_FILE > $COOCCURRENCE_SHUF_FILE"
./shuffle -memory $MEMORY -verbose $VERBOSE < $COOCCURRENCE_FILE > $COOCCURRENCE_SHUF_FILE

echo "$ ./glove -save-file $SAVE_FILE -threads $NUM_THREADS -input-file $COOCCURRENCE_SHUF_FILE -x-max $X_MAX -iter $MAX_ITER -vector-size $VECTOR_SIZE -binary $BINARY -vocab-file $VOCAB_FILE -verbose $VERBOSE"
./glove -save-file $SAVE_FILE -threads $NUM_THREADS -input-file $COOCCURRENCE_SHUF_FILE -x-max $X_MAX -iter $MAX_ITER -vector-size $VECTOR_SIZE -binary $BINARY -vocab-file $VOCAB_FILE -verbose $VERBOSE

echo "$ python evaluation.py"
python evaluation.py --vocab_file $VOCAB_FILE --vectors_file "${SAVE_FILE}".txt -p $TESTDIR -d "${RESULTDIR}"/results.txt
