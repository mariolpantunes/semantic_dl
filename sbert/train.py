from os import sep
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import argparse
import pandas as pd

parser = argparse.ArgumentParser()

parser.add_argument(
    '-d',
    '--destPath',
    action='store',
    required=True,
    dest='destPath',
    help='Destination model path'
)
parser.add_argument(
    '--modelPath',
    '-m',
    dest='modelPath',
    action='store',
    required=True,
    help='path to folder containing test files'
)
parser.add_argument(
    '--trainFile',
    '-t',
    dest='trainFile',
    action='store',
    required=True,
    help='File with all the training text'
)

args = parser.parse_args()

#Define the model. Either from scratch of by loading a pre-trained model
model = SentenceTransformer(args.modelPath)

training_file = pd.read_csv(args.trainFile, sep=";", header=None)

train_examples = [ InputExample(texts=[x[0], x[1]], label=x[3]) for x in training_file.to_numpy()]

#Define your train dataset, the dataloader and the train loss
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = losses.CosineSimilarityLoss(model)

#Tune the model
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=100)

model.save(args.destPath)