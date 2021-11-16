import argparse
import pandas as pd
import numpy as np 
from numpy import dot
from numpy.linalg import norm
import glob

from scipy.stats import pearsonr

parser = argparse.ArgumentParser()
parser.add_argument(
    '--vectors_file', 
    required=True,
    action='store',
    dest='vectors_file',
    help='File with the word vectors'
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

with open(args.vectors_file, 'r') as f:
    vectors = {}
    for line in f:
        vals = line.rstrip().split(' ')
        vectors[vals[0]] = [float(x) for x in vals[1:]]

vocab_size = len(vectors)
vocab = {w: idx for idx, w in enumerate(vectors.keys())}
ivocab = {idx: w for idx, w in enumerate(vectors.keys())}

vector_dim = len(vectors[ivocab[0]])
W = np.zeros((vocab_size, vector_dim))
for word, v in vectors.items():
    if word == '<unk>':
        continue
    W[vocab[word], :] = v

'''
Read Files to test for similarities
'''
test_files = glob.glob(args.testFolder+'*.csv')

test_dataset = []

print('Loading Test Datasets.')
for f in test_files:
    dataset = pd.read_csv(f, header=None).values
    test_dataset.append(dataset)

result = open(args.destFile, 'w')

for d in range(0, len(test_dataset)):
    predictions = []

    result.write("---------- " + str(test_files[d]) + " ----------\n")
    for pair in test_dataset[d]:
        if pair[0] in vocab and pair[1] in vocab:

            term_1 = W[vocab[pair[0]]]
            term_2 = W[vocab[pair[1]]]
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
