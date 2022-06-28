import argparse
import pandas as pd
from numpy import dot
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer
import time

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
result = open(args.destFile, 'a')


startTime = time.time()
model = SentenceTransformer(args.modelPath)
result.write("Model loading: " + str(time.time()-startTime) + "\n")

'''
Read Files to test for similarities
'''
test_files = glob.glob(args.testFolder+'*.csv')

test_dataset = []

print('Loading Test Datasets.')
for f in test_files:
    dataset = pd.read_csv(f, header=None).values
    test_dataset.append(dataset)


total_time = 0
for d in range(0, len(test_dataset)):
    predictions = []

    result.write("---------- " + str(test_files[d]) + " ----------\n")
    for pair in test_dataset[d]:
        startTime = time.time()
        term_1 = model.encode(pair[0])
        term_2 = model.encode(pair[1])
        
        sim = dot(term_1, term_2)/(norm(term_1)*norm(term_2))
        total_time += time.time() - startTime
        predictions.append(sim)
        result.write(str(sim) + "\n")
        
    print("Pearson Correlation Coefficient: ", pearsonr(predictions, test_dataset[d][:, 2])[0])
    result.write("Pearson Correlation Coefficient: "+ str(pearsonr(predictions, test_dataset[d][:, 2])[0])+"\n")
    result.write("--------------------\n")

result.write("Evaluation time: " + str(total_time) + "\n")

result.close()
