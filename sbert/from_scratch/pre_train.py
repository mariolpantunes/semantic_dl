
from transformers import RobertaTokenizerFast, RobertaForMaskedLM, RobertaConfig
from transformers import DataCollatorForLanguageModeling, DataCollatorForWholeWordMask
from transformers import Trainer, TrainingArguments
from tokenizers import ByteLevelBPETokenizer
import argparse
import os

parser = argparse.ArgumentParser()

parser.add_argument(
    '-v',
    '--vector_size',
    action='store',
    required=True,
    dest='vectorSize',
    help='Size of the embeddings'
)
parser.add_argument(
    '-p',
    '--path',
    required=True,
    action='store',
    dest='trainingPath',
    help='File with the training files'
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
parser.add_argument(
    '--devFile',
    '-d',
    dest='devFile',
    action='store',
    required=True,
    help='File with all the testing text'
)
args = parser.parse_args()

per_device_train_batch_size = 32

save_steps = 1000               #Save model every 1k steps
num_train_epochs = 5            #Number of epochs
use_fp16 = False                #Set to True, if your GPU supports FP16 operations
max_length = 100                #Max length for a text input
do_whole_word_mask = False      #If set to true, whole words are masked
mlm_prob = 0.15                 #Probability that a word is replaced by a [MASK] token


### Train tokenizer from scratch
paths = [ args.trainingPath + path for path in os.listdir(args.trainingPath) if ".csv" in path]

tokenizer = ByteLevelBPETokenizer()

tokenizer.train(files=paths, min_frequency=1, special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"])

tokenizer.save_model(args.modelPath)
#####################################

#### Initialize model and tokenizer for pretended task
config = RobertaConfig(
    max_position_embeddings=514,
    hidden_size=int(args.vectorSize),
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1
)

tokenizer = RobertaTokenizerFast.from_pretrained(args.modelPath, max_length=512)

model = RobertaForMaskedLM(config=config)

##### Load our training datasets

train_sentences = []
with open(args.trainFile, 'r', encoding='utf8') as fIn:
    for line in fIn:
        line = line.strip()
        if len(line) >= 10:
            train_sentences.append(line)

print("Train sentences:", len(train_sentences))

dev_sentences = []
with open(args.devFile, 'r', encoding='utf8') as fIn:
    for line in fIn:
        line = line.strip()
        if len(line) >= 10:
            dev_sentences.append(line)

print("Dev sentences:", len(dev_sentences))

#A dataset wrapper, that tokenizes our data on-the-fly
class TokenizedSentencesDataset:
    def __init__(self, sentences, tokenizer, max_length, cache_tokenization=False):
        self.tokenizer = tokenizer
        self.sentences = sentences
        self.max_length = max_length
        self.cache_tokenization = cache_tokenization

    def __getitem__(self, item):
        if not self.cache_tokenization:
            return self.tokenizer(self.sentences[item], add_special_tokens=True, truncation=True, max_length=self.max_length, return_special_tokens_mask=True)

        if isinstance(self.sentences[item], str):
            self.sentences[item] = self.tokenizer(self.sentences[item], add_special_tokens=True, truncation=True, max_length=self.max_length, return_special_tokens_mask=True)
        return self.sentences[item]

    def __len__(self):
        return len(self.sentences)

train_dataset = TokenizedSentencesDataset(train_sentences, tokenizer, max_length)
dev_dataset = TokenizedSentencesDataset(dev_sentences, tokenizer, max_length, cache_tokenization=True) if len(dev_sentences) > 0 else None


##### Training arguments

if do_whole_word_mask:
    data_collator = DataCollatorForWholeWordMask(tokenizer=tokenizer, mlm=True, mlm_probability=mlm_prob)
else:
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=mlm_prob)

training_args = TrainingArguments(
    output_dir=args.modelPath + "checkpoints/",
    overwrite_output_dir=True,
    num_train_epochs=num_train_epochs,
    evaluation_strategy="steps" if dev_dataset is not None else "no",
    per_device_train_batch_size=per_device_train_batch_size,
    eval_steps=save_steps,
    save_steps=save_steps,
    logging_steps=save_steps,
    save_total_limit=1,
    prediction_loss_only=True,
    fp16=use_fp16
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset
)

print("Save tokenizer to:", args.modelPath)
tokenizer.save_pretrained(args.modelPath)

trainer.train()

print("Save model to:", args.modelPath)
model.save_pretrained(args.modelPath)

print("Training done")