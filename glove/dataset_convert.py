
import nltk
import glob
import json
import string
import re

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

# Read the files in the dataset and create setences
print('Generating tokens from files.')
# Text Mining Pipeline
stop_words = set(nltk.corpus.stopwords.words('english'))
aggregated_files = open("aggregated_corpus", "w")

for f in train_files:
    with open(f, 'rt', newline='', encoding='utf-8') as f:
        snippets = f.readlines()
        for s in snippets:
            for token in tokenizer(s):
                aggregated_files.write(token+" ")

