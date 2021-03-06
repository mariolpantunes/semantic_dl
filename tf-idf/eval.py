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
    '-t',
    '--tf',
    action='store',
    default=None,
    dest='model_tfidf_path',
    help='File with the tf-idf model'
)

parser.add_argument(
    '-l',
    '--lsi',
    default=None,
    action='store',
    dest='model_lsi_path',
    help='File with the lsi model'
)

parser.add_argument(
    '--train_input',
    dest='setences_path',
    action='store',
    required=True,
    help='File containing the train examples'
)

parser.add_argument(
    '--test_input',
    dest='testFolder',
    action='store',
    required=True,
    help='path to folder containing test files'
)

parser.add_argument(
    '--outputFolder',
    '-o',
    dest='results_path',
    action='store',
    required=True,
    help='Path to store results'
)

args = parser.parse_args()

'''
Read Files to test for similarities
'''
test_files = glob.glob(args.testFolder+'*.csv')
test_dataset = []

print('Loading Test Datasets.')
for f in test_files:
    dataset = pd.read_csv(f, header=None).values
    test_dataset.append(dataset)


result = open(args.results_path + 'results.txt', 'a')

# load model
print('Loading previously trained model.')
startTime = time.time()

setences_tokens = json.load(open(args.setences_path))

dct = Dictionary(setences_tokens)

model_tfidf = TfidfModel.load(args.model_tfidf_path)

model_lsi = LsiModel.load(args.model_lsi_path)
result.write("Loading time: " + str(time.time() - startTime) + "\n")


'''
Testing the model.
'''
print('Testing the trained model.')

total_time = 0
for d in range(0, len(test_dataset)):
    predictions = []
    result.write("---------- " + str(test_files[d]) + " ----------\n")
    for pair in test_dataset[d]:
        startTime = time.time()
        if pair[0] in dct.token2id and pair[1] in dct.token2id:

            term_1 = pd.DataFrame(model_lsi[model_tfidf[dct.doc2bow(tokenizer(pair[0]))]], columns=['dim','val']) 
            term_2 = pd.DataFrame(model_lsi[model_tfidf[dct.doc2bow(tokenizer(pair[1]))]], columns=['dim','val'])

            sim = dot(term_1['val'], term_2['val'])/(norm(term_1['val'])*norm(term_2['val']))
            total_time += time.time() - startTime
            predictions.append(sim)
            result.write(str(sim) + "\n")
        else:
            total_time += time.time() - startTime
            print("Missing one of the words in the model: ", pair[0], pair[1])
            predictions.append(0.5)
            result.write("0.5\n")
            
    print("Pearson Correlation Coefficient: ", pearsonr(predictions, test_dataset[d][:, 2])[0])
    result.write("Pearson Correlation Coefficient: "+ str(pearsonr(predictions, test_dataset[d][:, 2])[0])+"\n")
    result.write("--------------------\n")

result.write("Evaluation time: " + str(total_time) + "\n")

result.close()