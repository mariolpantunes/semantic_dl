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

python word2vec.py --train_input $CORPUS --test_input $TESTDIR --outputFolder $RESULTDIR --vector_size $VECTOR_SIZE --window_size $WINDOW_SIZE
