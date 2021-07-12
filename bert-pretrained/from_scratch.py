from pathlib import Path

from tokenizers import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from transformers import RobertaConfig
from transformers import RobertaTokenizerFast
from transformers import RobertaForMaskedLM
from transformers import LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments


### Train tokenizer from scratch
paths = ["../dataset/aggregated_corpus"]

tokenizer = ByteLevelBPETokenizer()

tokenizer.train(files=paths, min_frequency=1, special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"])

tokenizer.save_model("EsperBERTo")

#tokenizer = ByteLevelBPETokenizer(
#    "./EsperBERTo/vocab.json",
#    "./EsperBERTo/merges.txt",
#)

tokenizer._tokenizer.post_processor = BertProcessing(
    ("</s>", tokenizer.token_to_id("</s>")),
    ("<s>", tokenizer.token_to_id("<s>")),
)
tokenizer.enable_truncation(max_length=512)
#####################################


#### Initialize model and tokenizer for pretended task
config = RobertaConfig(
    max_position_embeddings=514,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
)

tokenizer = RobertaTokenizerFast.from_pretrained("./EsperBERTo", max_len=512)

model = RobertaForMaskedLM(config=config)



#### Generate dataset for training
dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="../dataset/aggregated_corpus",
    block_size=128,
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

training_args = TrainingArguments(
    output_dir="./EsperBERTo",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_gpu_train_batch_size=64,
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

trainer.save_model("./EsperBERTo")