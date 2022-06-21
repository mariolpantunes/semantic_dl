import argparse
import glob

from tokenizer import tokenizer
import time

parser = argparse.ArgumentParser()

parser.add_argument(
    '--dataset',
    dest='dataset',
    action='store',
    required=True,
    help='Path to folder containing train files'
)
parser.add_argument(
    '--destFile',
    dest='destFile',
    action='store',
    required=True,
    help='File that stores the results'
)

parser.add_argument(
    '-r',
    dest='resultFile',
    action='store',
    required=True,
    help='File that stores the execution time'
)

args = parser.parse_args()

train_files = glob.glob(args.dataset+'*.csv')

# Read the files in the dataset and create setences
print('Generating tokens from files.')

# Text Mining Pipeline
startTime = time.time()
aggregated_files = open(args.destFile, "w")


for f in train_files:
    with open(f, 'rt', newline='', encoding='utf-8') as f:
        snippets = f.readlines()
        for s in snippets:
            for token in tokenizer(s):
                aggregated_files.write(token+" ")
    aggregated_files.write("\n")

aggregated_files.close()

executionTime = (time.time() - startTime)

result_file = open(args.resultFile, "w")

result_file.write("Preprocessing time: " + str(executionTime))

result_file.close()

