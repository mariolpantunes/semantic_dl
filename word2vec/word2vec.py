import glob
import pandas as pd
import json
import argparse

from gensim.models import Word2Vec
from scipy.stats import pearsonr

parser = argparse.ArgumentParser()

parser.add_argument(
    '-w',
    '--w2v',
    action='store',
    default=None,
    dest='model_path',
    help='File with the word2vec model'
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
    '--vector_size',
    '-v',
    dest='vector_size',
    action='store',
    required=True,
    help='Path to store results'
)
parser.add_argument(
    '--window_size',
    dest='window_size',
    action='store',
    required=True,
    help='Path to store results'
)

args = parser.parse_args()

print('Loading previously generated tokens.')
setences_tokens = json.load(open(args.setences_path))

'''
Read Files to test for similarities
'''
print('Loading Test Datasets.')
test_files = glob.glob(args.testFolder+'*.csv')
test_dataset = []

for f in test_files:
    dataset = pd.read_csv(f, header=None).values
    test_dataset.append(dataset)

'''
Loading/ Training the model.
'''
if args.model_path is not None:
    # load model
    print('Loading previously trained model.')
    model = Word2Vec.load(args.model_path)
else:
    # train model
    print('Training new model.')
    model = Word2Vec(setences_tokens, vector_size=int(args.vector_size), window=int(args.window_size), min_count=1, workers=4, sg=1, hs=0, negative=15, seed=17)

    # store model
    model.save(args.results_path + 'w2v.model')

words_seen_by_model = set(model.wv.index_to_key)


'''
Testing the model.
'''
print('Testing the trained model.')
result = open(args.results_path + 'results.txt', 'w')

for d in range(0, len(test_dataset)):
    predictions = []
    result.write("---------- " + str(test_files[d]) + " ----------\n")
    for pair in test_dataset[d]:
        if pair[0] in words_seen_by_model and pair[1] in words_seen_by_model:
            sim = model.wv.similarity(pair[0], pair[1])
            predictions.append(sim)
            result.write(str(sim) + "\n")
        else:
            print("Missing one of the words in the model: ", pair[0], pair[1])
            predictions.append(0.5)
            result.write("0.5\n")

    print("Pearson Correlation Coefficient: ", pearsonr(predictions, test_dataset[d][:, 2])[0])
    result.write("Pearson Correlation Coefficient: "+ str(pearsonr(predictions, test_dataset[d][:, 2])[0])+"\n")
    result.write("--------------------\n")
