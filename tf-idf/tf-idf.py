
import argparse
import glob
import pandas as pd
import json

from numpy import dot
from numpy.linalg import norm

from gensim.models import TfidfModel, LsiModel
from gensim.corpora import Dictionary
from scipy.stats import pearsonr

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
Read Files to test for similarities
'''
test_files = glob.glob(args.testFolder+'*.csv')
test_dataset = []

print('Loading Test Datasets.')
for f in test_files:
    dataset = pd.read_csv(f, header=None).values
    test_dataset.append(dataset)

'''
Loading/ Training the model.
'''
if args.model_tfidf_path is not None and args.model_lsi_path is not None:
    # load model
    print('Loading previously trained model.')
    dct = Dictionary(setences_tokens)

    model_tfidf = TfidfModel.load(args.model_tfidf_path)

    model_lsi = LsiModel.load(args.model_lsi_path)

else:
    # train model
    print('Training new model.')
    dct = Dictionary(setences_tokens)
    corpus = [dct.doc2bow(line) for line in setences_tokens]

    model_tfidf = TfidfModel(corpus, id2word=dct)
    tfidf_corpus =  model_tfidf[corpus]

    model_lsi = LsiModel(tfidf_corpus, id2word=dct, num_topics=args.n_topics)

    # store model
    model_tfidf.save(args.results_path + 'tf-idf.model')
    model_lsi.save(args.results_path + 'lsi.model')

'''
Testing the model.
'''
print('Testing the trained model.')
result = open(args.results_path + 'results.txt', 'w')

for d in range(0, len(test_dataset)):
    predictions = []
    result.write("---------- " + str(test_files[d]) + " ----------\n")
    for pair in test_dataset[d]:
        if pair[0] in dct.token2id and pair[1] in dct.token2id:

            term_1 = pd.DataFrame(model_lsi[model_tfidf[dct.doc2bow(tokenizer(pair[0]))]], columns=['dim','val']) 
            term_2 = pd.DataFrame(model_lsi[model_tfidf[dct.doc2bow(tokenizer(pair[1]))]], columns=['dim','val'])

            sim = dot(term_1['val'], term_2['val'])/(norm(term_1['val'])*norm(term_2['val']))

            predictions.append(sim)
            result.write(str(sim) + "\n")
        else:
            print("Missing one of the words in the model: ", pair[0], pair[1])
            predictions.append(0.5)
            result.write("0.5\n")
            
    print("Pearson Correlation Coefficient: ", pearsonr(predictions, test_dataset[d][:, 2])[0])
    result.write("Pearson Correlation Coefficient: "+ str(pearsonr(predictions, test_dataset[d][:, 2])[0])+"\n")
    result.write("--------------------\n")

