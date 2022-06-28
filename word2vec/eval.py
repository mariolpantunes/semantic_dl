import glob
import pandas as pd
import argparse
import time

from gensim.models import Word2Vec
import gensim.downloader as api
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
    '--test_input',
    dest='testFolder',
    action='store',
    required=True,
    help='path to folder containing test files'
)

parser.add_argument(
    '--outputFile',
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
print('Loading Test Datasets.')
test_files = glob.glob(args.testFolder+'*.csv')
test_dataset = []

for f in test_files:
    dataset = pd.read_csv(f, header=None).values
    test_dataset.append(dataset)

result = open(args.results_path, 'a')

'''
Loading/ Training the model.
'''
# load model
print('Loading previously trained model.')
if args.model_path == "pretrained":
    startTime = time.time()
    model = api.load("word2vec-google-news-300")
    result.write("Loading time: " + str(time.time() - startTime) + "\n")
else:
    startTime = time.time()
    model = Word2Vec.load(args.model_path).wv
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
        if pair[0] in model and pair[1] in model:
            sim = model.similarity(pair[0], pair[1])
            total_time += time.time() - startTime
            predictions.append(sim)
            result.write(str(sim) + "\n")
        else:
            total_time += time.time()-startTime
            print("Missing one of the words in the model: ", pair[0], pair[1])
            predictions.append(None)
            result.write("None\n")

    test_removed = [ x for i, x in enumerate(test_dataset[d][:, 2]) if predictions[i]]
    predictions_removed = [ x for x in predictions if x] 

    print("Pearson Correlation Coefficient: ", pearsonr(predictions_removed, test_removed)[0])
    result.write("Pearson Correlation Coefficient: "+ str(pearsonr(predictions_removed, test_removed)[0])+"\n")
    result.write("--------------------\n")

result.write("Evaluation time: " + str(total_time) + "\n")

result.close()
