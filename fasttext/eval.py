#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import argparse
import pandas as pd
import glob

from numpy import dot
from numpy.linalg import norm

from scipy.stats import pearsonr


#Pearson Correlation Coefficient:  0.682878515021699
#Pearson Correlation Coefficient:  0.6428373536030053


def compat_splitting(line):
    return line.decode('utf8').split()

parser = argparse.ArgumentParser(description='Process some integers.')
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

vectors = {}
fin = open(args.modelPath, 'rb')
for _, line in enumerate(fin):
    try:
        tab = compat_splitting(line)
        vec = np.array(tab[1:], dtype=float)
        word = tab[0]
        if np.linalg.norm(vec) == 0:
            continue
        if not word in vectors:
            vectors[word] = vec
    except ValueError:
        continue
    except UnicodeDecodeError:
        continue
fin.close()

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

result = open(args.destFile, 'w')

for d in range(0, len(test_dataset)):
    predictions = []
    result.write("---------- " + str(test_files[d]) + " ----------\n")

    for pair in test_dataset[d]:
        if pair[0] in vectors and pair[1] in vectors:

            term_1 = vectors[pair[0]]
            term_2 = vectors[pair[1]]

            sim = dot(term_1, term_2)/(norm(term_1)*norm(term_2))
            predictions.append(sim)
            result.write(str(sim) + "\n")
        else:
            print("Missing one of the words in the model: ", pair[0], pair[1])
            predictions.append(0.5)
            result.write("0.5\n")
            
    print("Pearson Correlation Coefficient: ", pearsonr(predictions, test_dataset[d][:, 2])[0])
    result.write("Pearson Correlation Coefficient: "+ str(pearsonr(predictions, test_dataset[d][:, 2])[0])+"\n")
    result.write("--------------------\n")

