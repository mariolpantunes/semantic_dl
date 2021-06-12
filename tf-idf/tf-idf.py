
import nltk
import glob
import pandas as pd
import json
import string
import re
from numpy import dot
from numpy.linalg import norm
import spacy
from gensim.models import TfidfModel, LsiModel
from gensim.corpora import Dictionary
from scipy.stats import pearsonr

'''
function for data cleaning and processing
'''
def tokenizer(sentence):
    stop_words = set(nltk.corpus.stopwords.words('english'))
    #create list of punctuations and stopwords
    punctuations = string.punctuation
 
    #remove distracting single quotes
    sentence = re.sub('\'','',sentence)

    #remove digits adnd words containing digits
    sentence = re.sub('\w*\d\w*','',sentence)

    #replace extra spaces with single space
    sentence = re.sub(' +',' ',sentence)

    #remove unwanted lines starting from special charcters
    sentence = re.sub(r'\n: \'\'.*','',sentence)
    sentence = re.sub(r'\n!.*','',sentence)
    sentence = re.sub(r'^:\'\'.*','',sentence)
    
    #remove non-breaking new line characters
    sentence = re.sub(r'\n',' ',sentence)
    
    #remove punctunations
    sentence = re.sub(r'[^\w\s]',' ',sentence)
    
    #lower, strip and lemmatize
    tokens = [word.lower().strip() for word in  nltk.word_tokenize(sentence)]
    
    #remove stopwords, and exclude words less than 2 characters
    tokens = [word for word in tokens if word not in stop_words and word not in punctuations and len(word) > 2]
    
    #return tokens
    return tokens

train_files = glob.glob('../dataset/*.csv')

test_files = ["../en-mc-30.csv", "../en-iot-30.csv"]

test_dataset = []

setences_tokens = []

'''
Change to the actual path to save time and skip training and word processing
'''

model_path = None
#model_path = "td-idf.model"

setences_path = "processed_setences.json"
#setences_path = None
'''
----------------------------------------------------------
'''


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
                setences_tokens.append(tokenizer(s))
    json.dump(setences_tokens, open('processed_setences.json', 'w'))


'''
Read Files to test for similarities
'''
print('Loading Test Datasets.')
for f in test_files:
    dataset = pd.read_csv(f, header=None).values
    test_dataset.append(dataset)


'''
Loading/Training the model.
'''
if model_path is not None:
    # load model
    print('Loading previously trained model.')
    dct = Dictionary(setences_tokens)
    corpus = [dct.doc2bow(line) for line in setences_tokens]

    model_tfidf = TfidfModel.load(model_path)
    #model_lsi = LsiModel.load(model_path)
    tfidf_corpus =  model_tfidf[corpus]
    #lsi_corpus =  model_lsi[corpus]

else:
    # train model
    print('Training new model.')
    dct = Dictionary(setences_tokens)
    corpus = [dct.doc2bow(line) for line in setences_tokens]

    model_tfidf = TfidfModel(corpus, id2word=dct)
    tfidf_corpus =  model_tfidf[corpus]

    model_lsi = LsiModel(tfidf_corpus, id2word=dct, num_topics=300)
    #lsi_corpus =  model_lsi[corpus]

    # store model
    model_tfidf.save('tf-idf.model')
    model_lsi.save('lsi.model')

# Test Model
print('Testing the trained model.')
for d in test_dataset:
    predictions = []
    for pair in d:
        if pair[0] in dct.token2id and pair[1] in dct.token2id:

            term_1 = pd.DataFrame(model_lsi[model_tfidf[dct.doc2bow(tokenizer(pair[0]))]], columns=['dim','val']) 
            term_2 = pd.DataFrame(model_lsi[model_tfidf[dct.doc2bow(tokenizer(pair[1]))]], columns=['dim','val'])

            predictions.append(dot(term_1['val'], term_2['val'])/(norm(term_1['val'])*norm(term_2['val'])))
        else:
            print("Missing one of the words in the model: ", pair[0], pair[1])
            predictions.append(0.5)
    print("Pearson Correlation Coefficient: ", pearsonr(predictions, d[:, 2])[0])
