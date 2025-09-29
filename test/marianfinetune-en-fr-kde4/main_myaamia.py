import os
import logging
import pandas as pd
from datasets import Dataset, concatenate_datasets
from transformers import MarianTokenizer, MarianMTModel, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
import evaluate
import numpy as np

# -------------------------
# CONFIG
# -------------------------
model_name = "Helsinki-NLP/opus-mt-en-fr"   # ✅ replace with right pair if exists
save_dir = "./marian_finetuned_myaamia"
log_file = "./training.log"
eval_file = "./eval_results.tsv"

# -------------------------
# LOGGER
# -------------------------
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# -------------------------
# LOAD DATA
# -------------------------
def load_tsv(path):
    df = pd.read_csv(path, sep="\t")
    df = df.dropna(subset=["mya", "en"])  # Remove rows with missing translations
    df["mya"] = df["mya"].astype(str)
    df["en"] = df["en"].astype(str)
    return Dataset.from_pandas(df)


datasets = [
    load_tsv("../../data/basic_forms.tsv"),
    load_tsv("../../data/command_forms.tsv"),
    load_tsv("../../data/sentences.tsv"),
    load_tsv("../../data/story_book.tsv"),
]

full_dataset = concatenate_datasets(datasets)
full_dataset = full_dataset.train_test_split(test_size=0.3, seed=42)  # 70% train, 30% val

# -------------------------
# TOKENIZER
# -------------------------
tokenizer = MarianTokenizer.from_pretrained(model_name)

def preprocess(batch):
    model_inputs = tokenizer(batch["mya"], max_length=128, truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(batch["en"], max_length=128, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = full_dataset.map(preprocess, batched=True)

# -------------------------
# METRIC (BLEU or sacreBLEU)
# -------------------------
metric = evaluate.load("sacrebleu")

def postprocess_text(preds, labels):
    preds = [p.strip() for p in preds]
    labels = [[l.strip()] for l in labels]
    return preds, labels

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    # log to file
    with open(eval_file, "a") as f:
        f.write(str(result) + "\n")

    return result

# -------------------------
# MODEL & TRAINING
# -------------------------
model = MarianMTModel.from_pretrained(model_name)

training_args = Seq2SeqTrainingArguments(
    output_dir=save_dir,
    eval_strategy="epoch",               
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=1,
    num_train_epochs=3,
    predict_with_generate=True,
    logging_dir="./logs",
    logging_strategy="steps",
    logging_steps=5,
    fp16=True,
    report_to=["none"],  # disables wandb
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# -------------------------
# TRAIN
# -------------------------
trainer.train()
