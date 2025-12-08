import os
import logging
import pandas as pd
import numpy as np
import nltk
import evaluate
from statistics import mean
from datasets import Dataset, concatenate_datasets
from transformers import (
    MarianTokenizer, 
    MarianMTModel, 
    Seq2SeqTrainer, 
    Seq2SeqTrainingArguments, 
    DataCollatorForSeq2Seq
)

# -------------------------
# CONFIG
# -------------------------
model_name = "Helsinki-NLP/opus-mt-tc-bible-big-mul-mul"
save_dir = "./marian_finetuned_myaamia"
log_file = "./training.log"
eval_file = "./eval_results.tsv"

# Ensure NLTK data is present
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

# -------------------------
# LOGGER
# -------------------------
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
# Add console handler to see logs live
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger("").addHandler(console)

# -------------------------
# LOAD DATA
# -------------------------
def load_tsv(path):
    try:
        df = pd.read_csv(path, sep="\t")
        df = df.dropna(subset=["mia", "en"])
        df["mia"] = df["mia"].astype(str)
        df["en"] = df["en"].astype(str)
        return Dataset.from_pandas(df)
    except Exception as e:
        logging.warning(f"Could not load {path}: {e}")
        return None

datasets = [
    load_tsv("../../../../../data/basic_forms.tsv"),
    load_tsv("../../../../../data/command_forms.tsv"),
    load_tsv("../../../../../data/sentences.tsv"),
    load_tsv("../../../../../data/story_book.tsv"),
]
# Filter out None if any files failed
datasets = [ds for ds in datasets if ds is not None]
full_dataset = concatenate_datasets(datasets)
full_dataset = full_dataset.train_test_split(test_size=0.3, seed=42)

# -------------------------
# TOKENIZER & MODEL
# -------------------------
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# --- NEW TOKEN SETUP ---
tgt_token = ">>mia<<" 

# Add token if it doesn't exist
if tgt_token not in tokenizer.get_vocab():
    num_added = tokenizer.add_tokens([tgt_token])
    if num_added > 0:
        model.resize_token_embeddings(len(tokenizer))
        logging.info(f"Added new token '{tgt_token}' and resized model embeddings.")

# -------------------------
# PREPROCESSING
# -------------------------
def preprocess(batch):
    # 1. Add prefix to source (English)
    src_texts = [f"{tgt_token} {text}" for text in batch["en"]]
    tgt_texts = batch["mia"]

    # 2. Tokenize Source
    # REMOVED padding="max_length" -> Let DataCollator handle dynamic padding (Faster!)
    model_inputs = tokenizer(
        src_texts,
        truncation=True,
        max_length=128
    )

    # 3. Tokenize Target
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            tgt_texts, 
            truncation=True, 
            max_length=128
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = full_dataset.map(preprocess, batched=True)

# -------------------------
# METRICS
# -------------------------
metric_bleu = evaluate.load("sacrebleu")
metric_chrf = evaluate.load("chrf")

# Initialize Eval File
with open(eval_file, "w") as f:
    f.write("bleu\tchrf++\n")

def postprocess_text(preds, labels):
    preds = [p.strip() for p in preds]
    labels = [[l.strip()] for l in labels]
    return preds, labels

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    if isinstance(preds, tuple):
        preds = preds[0]
    
    # --- CRITICAL FIX: Handle -100 for ignored loss indices ---
    # Convert -100 to pad_token_id so tokenizer doesn't crash
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    # BLEU
    bleu_result = metric_bleu.compute(predictions=decoded_preds, references=decoded_labels)
    
    # ChrF++
    try:
        chrf_result = metric_chrf.compute(predictions=decoded_preds, references=decoded_labels, word_order=2)
        chrf_score = chrf_result["score"]
    except Exception:
        chrf_score = 0.0

    result = {
        "bleu": round(bleu_result["score"], 2),
        "chrf++": round(chrf_score, 2),
    }

    # Log clean TSV
    with open(eval_file, "a") as f:
        f.write(f"{result['bleu']}\t{result['chrf++']}\n")

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
    save_total_limit=2,
    num_train_epochs=20,
    predict_with_generate=True,
    logging_dir="./logs",
    logging_strategy="steps",
    logging_steps=10,
    fp16=True,
    report_to="wandb"
)

# DataCollator automatically pads inputs to max length of the batch
# and sets label padding to -100 (ignoring loss)
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
if __name__ == "__main__":
    trainer.train()
    
    # Save Final
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)

#     {'eval_loss': 0.971784770488739, 'eval_bleu': 17.84, 'eval_chrf++': 68.43, 'eval_runtime': 97.2752, 'eval_samples_per_second': 58.895, 'eval_steps_per_second': 3.691, 'epoch': 20.0}                   
# {'train_runtime': 3614.5986, 'train_samples_per_second': 73.961, 'train_steps_per_second': 4.626, 'train_loss': 0.5208542583528104, 'epoch': 20.0}   