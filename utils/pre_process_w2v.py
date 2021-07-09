import argparse
import glob
import json

from tokenizer import tokenizer

'''
Pre-processing for tf-idf and word2vec
'''
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

setences_tokens = []

for f in train_files:
    with open(f, 'rt', newline='', encoding='utf-8') as f:
        snippets = f.readlines()
        for s in snippets:
            setences_tokens.append(tokenizer(s))
            
json.dump(setences_tokens, open(args.destFile, 'w'))