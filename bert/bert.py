import nltk
nltk.download('punkt')

import tarfile
import numpy as np
import pandas as pd
from numpy import dot
from numpy.linalg import norm

from nltk.tokenize import word_tokenize
from bert_embedding import BertEmbedding

import torch
from pytorch_transformers import *

import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import pearsonr

def get_similarity(word1, word2):
  bert = BertEmbedding(model="bert_12_768_12", dataset_name="wiki_multilingual")

  token_1 = word_tokenize(word1)
  token_2 = word_tokenize(word2)

  bert_embedding_1  = bert.embedding(sentences = token_1)
  bert_embedding_2  = bert.embedding(sentences = token_2)

  word_embeding_1 = bert_embedding_1[0][1][0]
  word_embeding_2 = bert_embedding_2[0][1][0]

  return dot(word_embeding_1, word_embeding_2)/(norm(word_embeding_1)*norm(word_embeding_2))

test_files = ["../en-mc-30.csv", "../en-iot-30.csv"]

test_dataset = []

print('Loading Test Datasets.')
for f in test_files:
    dataset = pd.read_csv(f, header=None).values
    test_dataset.append(dataset)

# Test Model
print('Testing the trained model.')
for d in test_dataset:
    predictions = []
    for pair in d:
      predictions.append(get_similarity(pair[0], pair[1]))

    print("Pearson Correlation Coefficient: ", pearsonr(predictions, d[:, 2])[0])

  