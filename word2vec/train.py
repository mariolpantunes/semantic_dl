import json
import argparse

from gensim.models import Word2Vec

parser = argparse.ArgumentParser()

parser.add_argument(
    '--train_input',
    dest='setences_path',
    action='store',
    required=True,
    help='File containing the train examples'
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

# train model
print('Training new model.')
model = Word2Vec(setences_tokens, vector_size=int(args.vector_size), window=int(args.window_size), min_count=1, workers=4, sg=1, hs=0, negative=15, seed=17)

# store model
model.save(args.results_path + 'w2v.model')
