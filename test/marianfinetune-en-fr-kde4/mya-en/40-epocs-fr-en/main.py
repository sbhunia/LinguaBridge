import os
import logging
import pandas as pd
from datasets import Dataset, concatenate_datasets
from transformers import (
    MarianTokenizer, 
    MarianMTModel, 
    Seq2SeqTrainer, 
    Seq2SeqTrainingArguments, 
    DataCollatorForSeq2Seq
)
import evaluate
import numpy as np
import nltk
from statistics import mean

# -------------------------
# CONFIG
# -------------------------
model_name = "Helsinki-NLP/opus-mt-fr-en"
save_dir = "./marian_finetuned_myaamia"
log_file = "./training.log"
eval_file = "./eval_results.tsv"

# Ensure wordnet is downloaded for METEOR
nltk.download("wordnet", quiet=True)
nltk.download('omw-1.4', quiet=True) # Often needed for meteor

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
    # Ensure paths are correct relative to where you run this script
    df = pd.read_csv(path, sep="\t")
    df = df.dropna(subset=["mia", "en"])
    df["mia"] = df["mia"].astype(str)
    df["en"] = df["en"].astype(str)
    return Dataset.from_pandas(df)

# Adjust paths as necessary
datasets = [
    load_tsv("../../../../data/basic_forms.tsv"),
    load_tsv("../../../../data/command_forms.tsv"),
    load_tsv("../../../../data/sentences.tsv"),
    load_tsv("../../../../data/story_book.tsv"),
]

full_dataset = concatenate_datasets(datasets)
full_dataset = full_dataset.train_test_split(test_size=0.3, seed=42)

# -------------------------
# TOKENIZER & MODEL
# -------------------------
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# --- CRITICAL STEP: CHECK FOR SPECIAL CHARACTERS ---
# If Myaamia has characters not in French (like 'š' or 'č'), 
# uncomment the lines below to add them.
new_tokens = ["š", "č", "’"] 
num_added_toks = tokenizer.add_tokens(new_tokens)
if num_added_toks > 0:
    model.resize_token_embeddings(len(tokenizer))
    print(f"Added {num_added_toks} new tokens to vocabulary.")

# -------------------------
# PREPROCESSING
# -------------------------
def preprocess(batch):
    # Source: Myaamia
    # We do NOT pad here. We let the DataCollator pad dynamically per batch.
    # This speeds up training significantly.
    model_inputs = tokenizer(batch["mia"], max_length=128, truncation=True)
    
    # Target: English
    # No prefix needed for single-purpose fine-tuning
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(batch["en"], max_length=128, truncation=True)
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = full_dataset.map(preprocess, batched=True)

# -------------------------
# METRICS
# -------------------------
metric_bleu = evaluate.load("sacrebleu")
metric_chrf = evaluate.load("chrf") # ChrF is excellent for unseen/morphological languages

def postprocess_text(preds, labels):
    preds = [p.strip() for p in preds]
    labels = [[l.strip()] for l in labels]
    return preds, labels

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    if isinstance(preds, tuple):
        preds = preds[0]
        
    # Replace -100 (ignored index) with pad token id for decoding
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    # 1. BLEU
    bleu_result = metric_bleu.compute(predictions=decoded_preds, references=decoded_labels)

    # 3. ChrF (Character F-score)
    # Often better than BERTScore for low-resource/unknown languages
    chrf_result = metric_chrf.compute(predictions=decoded_preds, references=decoded_labels)

    result = {
        "bleu": round(bleu_result["score"], 4),
        "chrf": round(chrf_result["score"], 4)
    }

    # Log results
    with open(eval_file, "a") as f:
        f.write(str(result) + "\n")

    return result

# -------------------------
# TRAINING ARGUMENTS
# -------------------------
training_args = Seq2SeqTrainingArguments(
    output_dir=save_dir,
    eval_strategy="epoch",               
    save_strategy="epoch",
    learning_rate=2e-5,          # Slightly higher LR is often okay for fine-tuning
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=2,          # Save fewer checkpoints to save disk space
    num_train_epochs=40,
    predict_with_generate=True,
    logging_dir="./logs",
    logging_strategy="steps",
    logging_steps=10,
    fp16=True,                   # Use mixed precision (faster)
    group_by_length=True,        # Sorts samples by length for efficient padding
)

# DataCollator handles dynamic padding
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
    
    # Save the final model explicitly
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)

#     {'eval_loss': 0.9383237361907959, 'eval_bleu': 66.2928, 'eval_chrf': 73.9809, 'eval_runtime': 76.6686, 'eval_samples_per_second': 74.724, 'eval_steps_per_second': 4.682, 'epoch': 40.0}                                                                                                      
# {'train_runtime': 5677.9505, 'train_samples_per_second': 94.168, 'train_steps_per_second': 5.889, 'train_loss': 0.34333550598200535, 'epoch': 40.0}                                                                                                                                           
# 100%|██████████████████████████████████████████████████████████████████████████████████████████████████