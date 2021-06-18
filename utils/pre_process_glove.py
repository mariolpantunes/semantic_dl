import argparse
import glob
import json
import string
import re

from tokenizer import tokenizer

parser = argparse.ArgumentParser()

parser.add_argument(
    '--dataset',
    dest='dataset',
    action='store',
    required=True,
    help='path to folder containing train files'
)
parser.add_argument(
    '--destFile',
    dest='destFile',
    action='store',
    required=True,
    help='File that stores the results'
)

args = parser.parse_args()

train_files = glob.glob(args.dataset+'*.csv')

# Read the files in the dataset and create setences
print('Generating tokens from files.')
# Text Mining Pipeline
aggregated_files = open(args.destFile, "w")

for f in train_files:
    with open(f, 'rt', newline='', encoding='utf-8') as f:
        snippets = f.readlines()
        for s in snippets:
            for token in tokenizer(s):
                aggregated_files.write(token+" ")
    aggregated_files.write("\n")

