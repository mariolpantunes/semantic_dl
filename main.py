import nltk
import glob
import multiprocessing
import pandas as pd
import numpy as np
import json

from gensim.models import Word2Vec
from scipy.stats import pearsonr


train_files = glob.glob('dataset/*.csv')

test_files = ["en-iot-30.csv", "en-mc-30.csv"]

test_examples = []

setences_tokens = []


#Change to the actual path to save time and skip training
model_path = None
#model_path = "w2v.model"

#setences_path = "processed_setences.json"
setences_path = None


if setences_path is not None:
    print("Loading previously generated tokens.")
    setences_tokens = json.load(open("processed_setences.json"))
    
else:
    # Read the files in the dataset and create setences
    print("Generating tokens from files.")
    for f in train_files:
        with open(f, 'rt', newline='', encoding='utf-8') as f:
            snippets = f.readlines()
            # Text Mining Pipeline
            stop_words = set(nltk.corpus.stopwords.words('english'))
            for s in snippets:
                setences_tokens.append(
                    [w.lower() for w in nltk.word_tokenize(s) if not w in stop_words and w.isalpha() and len(w) > 2]
                    )

    json.dump(setences_tokens, open("processed_setences.json", "w"))

# Read Files to test for similarities
print("Loading Test Examples.")
for f in test_files:
    test_examples.append(pd.read_csv(f, header=None))

test_examples = pd.concat(test_examples, axis=0).values

if model_path is not None:
    # load model
    print("Loading previously trained model.")
    model = Word2Vec.load(model_path)
else:
    # train model
    print("Training new model.")
    model = Word2Vec(setences_tokens, vector_size=100, window=3, min_count=1, workers=4)

    # store model
    model.save('w2v.model')


words_seen_by_model = set(model.wv.index_to_key)

# Test Model
print("Testing the trained model.")
predictions = []
for example in test_examples:
    if example[0] in words_seen_by_model and example[1] in words_seen_by_model:
        predictions.append(model.wv.similarity(example[0], example[1]))
    else:
        print("Missing one of the words in the model: ", example[0], example[1])
        predictions.append(0.5)

print("Pearson Correlation Coefficient: ", pearsonr(predictions, test_examples[:, 2])[0])

print("Car and Automobile Similarity: ", model.wv.similarity('car', 'automobile'))

#word_vectors = 
#
#vocab_len = len(word_vectors)
#print(vocab_len)
#
#print(word_vectors.key_to_index)

#sims = model.wv.most_similar('car', topn=10)
#print(sims)

#sim = word_vectors.similarity('automobile', 'automobile')
#print(sim)

# test a pair
# 

#print(sim)

