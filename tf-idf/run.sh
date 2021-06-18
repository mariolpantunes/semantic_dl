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

mkdir -p "${RESULTDIR}"

python tf-idf.py --train_input $CORPUS --test_input $TESTDIR -o $RESULTDIR -n $NTOPICS
