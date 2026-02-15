import os
import logging
import pandas as pd
from datasets import Dataset, concatenate_datasets
from transformers import MarianTokenizer, MarianMTModel, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
import evaluate
import numpy as np
import nltk
import bert_score
from statistics import mean

# -------------------------
# CONFIG
# -------------------------
model_name = "Helsinki-NLP/opus-mt-mul-en" 
save_dir = "./marian_finetuned_myaamia"
log_file = "./training.log"
eval_file = "./marianmt_en_mul_eval_results.tsv"
nltk.download("wordnet", quiet=True)

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
    df = df.dropna(subset=["mia", "en"])  # Remove rows with missing translations
    df["mia"] = df["mia"].astype(str)
    df["en"] = df["en"].astype(str)
    return Dataset.from_pandas(df)


datasets = [
    load_tsv("../../../../../data/basic_forms.tsv"),
    load_tsv("../../../../../data/command_forms.tsv"),
    load_tsv("../../../../../data/sentences.tsv"),
    load_tsv("../../../../../data/story_book.tsv"),
]


full_dataset = concatenate_datasets(datasets)
full_dataset = full_dataset.train_test_split(test_size=0.3, seed=42)  # 70% train, 30% val

# -------------------------
# MODEL
# -------------------------
model = MarianMTModel.from_pretrained(model_name)

# -------------------------
# TOKENIZER
# -------------------------
tokenizer = MarianTokenizer.from_pretrained(model_name)

# Old code had a source and target direction issue with the model selected. Expected: english as souce and myaamia/french(pretrained model) as target
# def preprocess(batch):
#     model_inputs = tokenizer(batch["mya"], max_length=128, truncation=True)
#     with tokenizer.as_target_tokenizer():
#         labels = tokenizer(batch["en"], max_length=128, truncation=True)
#     model_inputs["labels"] = labels["input_ids"]
#     return model_inputs

# tokenized_dataset = full_dataset.map(preprocess, batched=True) 

new_token = ">>en<<" # Marian style target language token based on ISO 639-3 language codes

def preprocess(batch):
    # prepend the language token to source sentences
    src_texts = [f"{new_token} {text}" for text in batch["mia"]]
    tgt_texts = batch["en"]

    model_inputs = tokenizer(src_texts, max_length=128, truncation=True)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(tgt_texts, max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# def preprocess(batch):
#     # Source: English
#     model_inputs = tokenizer(batch["en"], max_length=128, truncation=True, padding="max_length")
    
#     # Target: Myaamia
#     with tokenizer.as_target_tokenizer():
#         labels = tokenizer([prefix + t for t in batch["mya"]], 
#                            max_length=128, truncation=True, padding="max_length")
    
#     model_inputs["labels"] = labels["input_ids"]
#     return model_inputs

tokenized_dataset = full_dataset.map(preprocess, batched=True)


# -------------------------
# METRIC (sacreBLEU)
# -------------------------
metric = evaluate.load("sacrebleu")
chrf_metric = evaluate.load("chrf")

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

    bleu_result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    bleu_score = bleu_result["score"]

    # chfr
    try:
        chrf_result = chrf_metric.compute(predictions=decoded_preds, references=decoded_labels, word_order=2)
        chrf_score = chrf_result["score"]
    except Exception as e:
        logging.warning(f"chrF++ calculation failed: {e}")
        chrf_score = 0.0

    # Combine results
    result = {
        "sacrebleu": round(bleu_score, 4),
        "chrf++": round(chrf_score, 4),
    }

    # Log to file
    with open(eval_file, "a") as f:
        f.write(str(result) + "\n")

    return result

# -------------------------
# TRAINING
# -------------------------

training_args = Seq2SeqTrainingArguments(
    output_dir=save_dir,
    eval_strategy="epoch",               
    save_strategy="epoch",
    learning_rate=2e-5,
    warmup_ratio=0.06,           
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=40,
    predict_with_generate=True,
    logging_dir="./logs",
    logging_strategy="steps",
    logging_steps=5,
    fp16=True,
    report_to="wandb"
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

# {'eval_loss': 1.1505186557769775, 'eval_sacrebleu': 68.2848, 'eval_chrf++': 74.5195, 'eval_runtime': 101.2087, 'eval_samples_per_second': 56.606, 'eval_steps_per_second': 3.547, 'epoch': 40.0}                                                    
# {'train_runtime': 6206.8132, 'train_samples_per_second': 86.144, 'train_steps_per_second': 5.388, 'train_loss': 0.44584113181909574, 'epoch': 40.0}  