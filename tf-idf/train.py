
import argparse
import glob
import pandas as pd
import json

from numpy import dot
from numpy.linalg import norm

from gensim.models import TfidfModel, LsiModel
from gensim.corpora import Dictionary
from scipy.stats import pearsonr

import time

import sys
sys.path.insert(1, "../utils/")

from tokenizer import tokenizer

parser = argparse.ArgumentParser()

parser.add_argument(
    '--train_input',
    dest='setences_path',
    action='store',
    required=True,
    help='File containing the train examples'
)

parser.add_argument(
    '--outputFolder',
    '-o',
    dest='results_path',
    action='store',
    required=True,
    help='Path to store results'
)

parser.add_argument(
    '--n_topics',
    '-n',
    dest='n_topics',
    action='store',
    required=True,
    help='Number of LSI topics'
)

args = parser.parse_args()

print('Loading previously generated tokens.')
setences_tokens = json.load(open(args.setences_path))


'''
Loading/ Training the model.
'''

# train model
print('Training new model.')
startTime = time.time()
dct = Dictionary(setences_tokens)
corpus = [dct.doc2bow(line) for line in setences_tokens]

model_tfidf = TfidfModel(corpus, id2word=dct)
tfidf_corpus =  model_tfidf[corpus]

model_lsi = LsiModel(tfidf_corpus, id2word=dct, num_topics=args.n_topics)

# store model
model_tfidf.save(args.results_path + 'tf-idf.model')
model_lsi.save(args.results_path + 'lsi.model')



