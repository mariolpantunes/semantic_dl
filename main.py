import nltk
import glob
import pandas as pd
import json

from gensim.models import Word2Vec
from scipy.stats import pearsonr


train_files = glob.glob('dataset/*.csv')

test_files = ["en-mc-30.csv", "en-iot-30.csv"]

test_dataset = []

setences_tokens = []


#Change to the actual path to save time and skip training
model_path = None
#model_path = "w2v.model"

setences_path = "processed_setences.json"
#setences_path = None

if setences_path is not None:
    print('Loading previously generated tokens.')
    setences_tokens = json.load(open(setences_path))
    
else:
    # Read the files in the dataset and create setences
    print('Generating tokens from files.')
    # Text Mining Pipeline
    stop_words = set(nltk.corpus.stopwords.words('english'))
    for f in train_files:
        with open(f, 'rt', newline='', encoding='utf-8') as f:
            snippets = f.readlines()
            for s in snippets:
                setences_tokens.append([w.lower() for w in nltk.word_tokenize(s) if not w in stop_words and w.isalpha() and len(w) > 2])
    json.dump(setences_tokens, open('processed_setences.json', 'w'))


# Read Files to test for similarities
print('Loading Test Datasets.')
for f in test_files:
    dataset = pd.read_csv(f, header=None).values
    test_dataset.append(dataset)

if model_path is not None:
    # load model
    print('Loading previously trained model.')
    model = Word2Vec.load(model_path)
else:
    # train model
    print('Training new model.')
    model = Word2Vec(setences_tokens, vector_size=20, window=7, min_count=1, workers=10)

    # store model
    model.save('w2v.model')


words_seen_by_model = set(model.wv.index_to_key)


# Test Model
print('Testing the trained model.')
for d in test_dataset:
    predictions = []
    for pair in d:
        if pair[0] in words_seen_by_model and pair[1] in words_seen_by_model:
            predictions.append(model.wv.similarity(pair[0], pair[1]))
        else:
            print("Missing one of the words in the model: ", pair[0], pair[1])
            predictions.append(0.5)
    print("Pearson Correlation Coefficient: ", pearsonr(predictions, d[:, 2])[0])
