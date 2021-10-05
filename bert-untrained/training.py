from pathlib import Path

from tokenizers import ByteLevelBPETokenizer
from transformers import RobertaConfig
from tokenizers.processors import BertProcessing
from transformers import RobertaTokenizerFast
from transformers import RobertaForMaskedLM
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import pipeline

import torch

from torch.utils.data import Dataset

import os

class PersonalDataset(Dataset):
    def __init__(self, base_model, src_files):
        tokenizer = ByteLevelBPETokenizer(
            base_model + "/vocab.json",
           base_model + "/merges.txt",
        )
        tokenizer._tokenizer.post_processor = BertProcessing(
            ("</s>", tokenizer.token_to_id("</s>")),
            ("<s>", tokenizer.token_to_id("<s>")),
        )
        tokenizer.enable_truncation(max_length=512)

        self.examples = []

        for src_file in src_files:
            with open(src_file, 'r', encoding='utf-8') as fp:
                lines = fp.readlines()

                self.examples += [x.ids for x in tokenizer.encode_batch(lines)]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        # We’ll pad at the batch level.
        return torch.tensor(self.examples[i])

training_folder = "../dataset/train/"

### Train tokenizer from scratch
paths = [ training_folder + path for path in os.listdir(training_folder) if ".csv" in path]

tokenizer = ByteLevelBPETokenizer()

tokenizer.train(files=paths, min_frequency=1, special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"])

tokenizer.save_model("model")
#####################################


#### Initialize model and tokenizer for pretended task
config = RobertaConfig(
    max_position_embeddings=514,
    hidden_size=768,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1
)

tokenizer = RobertaTokenizerFast.from_pretrained("./model", max_length=512)

model = RobertaForMaskedLM(config=config)

#### Generate dataset for training ------- NEEDS TO BE UPDATED TO NEW VERSION --------

dataset = PersonalDataset("./model", paths[0:2])

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

training_args = TrainingArguments(
    output_dir="./trained_model",
    overwrite_output_dir=True,
    num_train_epochs=1000,
    per_gpu_train_batch_size=20,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

trainer.train()

trainer.save_model("./model")

nlp = pipeline(
    "feature-extraction",
    model="./model",
    tokenizer="./model",
)