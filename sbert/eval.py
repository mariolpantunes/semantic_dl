import argparse
import pandas as pd
from numpy import dot
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer

import glob

from scipy.stats import pearsonr

parser = argparse.ArgumentParser()
parser.add_argument(
    '-m',
    '--modelPath',
    action='store',
    required=True,
    dest='modelPath',
    help='Folder with the objective model'
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

model = SentenceTransformer(args.modelPath)

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

        term_1 = model.encode(pair[0])
        term_2 = model.encode(pair[1])
        
        sim = dot(term_1, term_2)/(norm(term_1)*norm(term_2))
        predictions.append(sim)
        result.write(str(sim) + "\n")
        
    print("Pearson Correlation Coefficient: ", pearsonr(predictions, test_dataset[d][:, 2])[0])
    result.write("Pearson Correlation Coefficient: "+ str(pearsonr(predictions, test_dataset[d][:, 2])[0])+"\n")
    result.write("--------------------\n")
