import nltk
import glob
import multiprocessing

from gensim.models import Word2Vec

# Read the files in the dataset and create setences

files = glob.glob('dataset/*.csv')

tokens = []

for f in files:
    with open(f, 'rt', newline='', encoding='utf-8') as f:
        snippets = f.readlines()
        # Text Mining Pipeline
        stop_words = set(nltk.corpus.stopwords.words('english'))
        for s in snippets:
            temp_tokens = nltk.word_tokenize(s)
            filtered_tokens = [w.lower() for w in temp_tokens if not w in stop_words and w.isalpha() and len(w) > 2]
            tokens.extend(filtered_tokens)

# train model
model = Word2Vec(tokens, vector_size=100, window=3, min_count=1, workers=4)

# load model
#model = Word2Vec.load('w2v.model')

# store model
model.save('w2v.model')

word_vectors = model.wv

vocab_len = len(word_vectors)
print(vocab_len)

print(word_vectors.key_to_index)

#sims = model.wv.most_similar('car', topn=10)
#print(sims)

#sim = word_vectors.similarity('automobile', 'automobile')
#print(sim)

# test a pair
# model.similarity('car', 'automobile')

#print(sim)