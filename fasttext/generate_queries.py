import argparse
import pandas as pd
import glob


parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument(
    '--path',
    '-p',
    dest='testFolder',
    action='store',
    required=True,
    help='path to folder containing test files'
)
parser.add_argument(
    '--dest',
    '-d',
    dest='queryFile',
    action='store',
    required=True,
    help='destination file'
)

args = parser.parse_args()

test_files = glob.glob(args.testFolder+'*.csv')
test_dataset = []

for f in test_files:
    dataset = pd.read_csv(f, header=None).values
    test_dataset.append(dataset)
words = set()

for d in test_dataset:
    predictions = []
    for pair in d:
        words.add(pair[0])
        words.add(pair[1])

queries = open(args.queryFile, "w")
for word in words:
    queries.write(word + "\n")

