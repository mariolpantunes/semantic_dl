

import pandas as pd
from numpy import dot
from numpy.linalg import norm

import torch
from transformers import AutoTokenizer, AutoModel

from scipy.stats import pearsonr


'''
Obtains 0.77 & 0.43
'''

def get_similarity(word1, word2):

  encoded_input = tokenizer([word1, word2], padding=True, truncation=True, max_length=64, return_tensors='pt')

  with torch.no_grad():
    model_output = model(**encoded_input)
    embeddings = model_output.pooler_output
    
  return dot(embeddings[0], embeddings[1])/(norm(embeddings[0])*norm(embeddings[1]))


tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/LaBSE")
model = AutoModel.from_pretrained("sentence-transformers/LaBSE")

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
  