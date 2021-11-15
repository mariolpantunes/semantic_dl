import glob
import pandas as pd
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

'''
Loading/ Training the model.
'''
# load model
print('Loading previously trained model.')
if args.model_path == "pretrained":
    model = Word2Vec.load(args.model_path)
else:
    model = Word2Vec.load(args.model_path).wv

'''
Testing the model.
'''
print('Testing the trained model.')
result = open(args.results_path, 'w')

for d in range(0, len(test_dataset)):
    predictions = []
    result.write("---------- " + str(test_files[d]) + " ----------\n")
    for pair in test_dataset[d]:
        if pair[0] in model and pair[1] in model:
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