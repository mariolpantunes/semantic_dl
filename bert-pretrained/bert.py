

import pandas as pd
from numpy import dot
from numpy.linalg import norm

import torch
from transformers import BertTokenizer, BertModel

from scipy.stats import pearsonr


'''
Obtains 0.50 & 0.04
'''

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def get_similarity(word1, word2):

  encoded_input = tokenizer([word1, word2], padding=True, truncation=True, max_length=64, return_tensors='pt')

  with torch.no_grad():

    model_output = model(**encoded_input)
  
    embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

  return dot(embeddings[0], embeddings[1])/(norm(embeddings[0])*norm(embeddings[1]))


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

test_files = ["../en-mc-30.csv", "../en-iot-30.csv"]

test_dataset = []

print('Loading Test Datasets.')
for f in test_files:
    dataset = pd.read_csv(f, header=None).values
    test_dataset.append(dataset)

# Test Model
print('Testing the trained model.')
results = []

for d in test_dataset:
    predictions = []
    for pair in d:
      predictions.append(get_similarity(pair[0], pair[1]))

    results.append(pearsonr(predictions, d[:, 2])[0])

print("Pearson Correlation Coefficient: ", results[0])
print("Pearson Correlation Coefficient: ", results[1])
  