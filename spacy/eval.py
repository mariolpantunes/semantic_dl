#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import pandas as pd
import glob

from numpy import dot
from numpy.linalg import norm

from scipy.stats import pearsonr

import spacy

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

nlp = spacy.load(args.modelPath)

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
        doc1 = nlp(pair[0])
        doc2 = nlp(pair[1])
        if doc1[0].has_vector and doc2[0].has_vector:
            #Try wiht doc1[0].similarity(doc2[0])
            term_1 = doc1[0].vector
            term_2 = doc2[0].vector
            
            sim = dot(term_1, term_2)/(norm(term_1)*norm(term_2))
            predictions.append(sim)
            result.write(str(sim) + "\n")
        else:
            print("Missing one of the words in the model: ", pair[0], pair[1])
            predictions.append(None)
            result.write("None\n")

    test_removed = [ x for i, x in enumerate(test_dataset[d][:, 2]) if predictions[i]]
    predictions_removed = [ x for x in predictions if x] 

    print("Pearson Correlation Coefficient: ", pearsonr(predictions_removed, test_removed)[0])
    result.write("Pearson Correlation Coefficient: "+ str(pearsonr(predictions_removed, test_removed)[0])+"\n")
    result.write("--------------------\n")

