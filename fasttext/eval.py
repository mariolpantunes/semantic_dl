#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
from tracemalloc import start
import pandas as pd
import glob
import io

from numpy import dot
from numpy.linalg import norm

from scipy.stats import pearsonr
import time

parser = argparse.ArgumentParser()
parser.add_argument(
    '--model',
    '-m',
    dest='modelPath',
    action='store',
    required=True,
    help='path to model'
)
parser.add_argument(
    '--path',
    '-p',
    dest='testFolder',
    action='store',
    required=True,
    help='path to folder containing test files'
)
parser.add_argument(
    '--destFile',
    '-d',
    dest='destFile',
    action='store',
    required=True,
    help='File that stores the results'
)

args = parser.parse_args()

def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    fin.readline()
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = [float(x) for x in  tokens[1:]]
    
    return data

result = open(args.destFile, 'a')

start_time = time.time()
vectors = load_vectors(args.modelPath)
result.write("Vector loading: " + str(time.time()-start_time) + "\n")

'''
Read Files to test for similarities
'''
test_files = glob.glob(args.testFolder+'*.csv')

test_dataset = []

print('Loading Test Datasets.')
for f in test_files:
    dataset = pd.read_csv(f, header=None).values
    test_dataset.append(dataset)

'''
Testing the model.
'''
print('Testing the trained model.')

total_time = 0
for d in range(0, len(test_dataset)):
    predictions = []
    result.write("---------- " + str(test_files[d]) + " ----------\n")

    for pair in test_dataset[d]:
        if pair[0] in vectors and pair[1] in vectors:
            startTime = time.time()
            term_1 = vectors[pair[0]]
            term_2 = vectors[pair[1]]

            sim = dot(term_1, term_2)/(norm(term_1)*norm(term_2))

            total_time += (time.time() - startTime)

            predictions.append(sim)
            result.write(str(sim) + "\n")
        else:
            print("Missing one of the words in the model: ", pair[0], pair[1])
            predictions.append(None)
            result.write("None\n")

    test_removed = [ x for i, x in enumerate(test_dataset[d][:, 2]) if predictions[i]]
    predictions_removed = [ x for x in predictions if x] 
            
    print("Pearson Correlation Coefficient: ", pearsonr(test_removed, predictions_removed)[0])
    result.write("Pearson Correlation Coefficient: "+ str(pearsonr(test_removed, predictions_removed)[0])+"\n")
    result.write("--------------------\n")

result.write("Evaluation time: " + str(total_time) + "\n")