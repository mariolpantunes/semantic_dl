import argparse
import pandas as pd
import numpy as np 
from numpy import dot
from numpy.linalg import norm

from scipy.stats import pearsonr

test_files = ["../en-mc-30.csv", "../en-iot-30.csv"]

test_dataset = []


parser = argparse.ArgumentParser()
parser.add_argument('--vocab_file', default='vocab.txt', type=str)
parser.add_argument('--vectors_file', default='vectors.txt', type=str)
args = parser.parse_args()

with open(args.vocab_file, 'r') as f:
    words = [x.rstrip().split(' ')[0] for x in f.readlines()]
with open(args.vectors_file, 'r') as f:
    vectors = {}
    for line in f:
        vals = line.rstrip().split(' ')
        vectors[vals[0]] = [float(x) for x in vals[1:]]

vocab_size = len(words)
vocab = {w: idx for idx, w in enumerate(words)}
ivocab = {idx: w for idx, w in enumerate(words)}

vector_dim = len(vectors[ivocab[0]])
W = np.zeros((vocab_size, vector_dim))
for word, v in vectors.items():
    if word == '<unk>':
        continue
    W[vocab[word], :] = v

# normalize each word vector to unit length
W_norm = np.zeros(W.shape)
d = (np.sum(W ** 2, 1) ** (0.5))
W_norm = (W.T / d).T


'''
Read Files to test for similarities
'''
print('Loading Test Datasets.')
for f in test_files:
    dataset = pd.read_csv(f, header=None).values
    test_dataset.append(dataset)

for d in test_dataset:
    predictions = []
    for pair in d:
        if pair[0] in vocab and pair[1] in vocab:

            term_1 = W_norm[vocab[pair[0]]]
            term_2 = W_norm[vocab[pair[1]]]

            predictions.append(dot(term_1, term_2)/(norm(term_1)*norm(term_2)))
        else:
            print("Missing one of the words in the model: ", pair[0], pair[1])
            predictions.append(0.5)
    print("Pearson Correlation Coefficient: ", pearsonr(predictions, d[:, 2])[0])