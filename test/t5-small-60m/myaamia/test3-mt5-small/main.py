import os
# Ensure this matches your actual GPU ID availability
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import logging
import pandas as pd
import numpy as np
import wandb

from datasets import Dataset, concatenate_datasets
# Using Fast tokenizer for speed
from transformers import (
    MT5ForConditionalGeneration,
    MT5TokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)
import evaluate

# -------------------------
# CONFIG
# -------------------------
model_name = "google/mt5-small"
save_dir = "./mt5_finetuned_myaamia"
log_file = "./training.log"
eval_file = "./mt5_small_eval_results.tsv"

# Weights & Biases Project Name
os.environ["WANDB_PROJECT"] = "myaamia-translation-mt5"

# -------------------------
# LOGGER
# -------------------------
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger("").addHandler(console)
logging.info("Starting mT5 fine-tuning script.")

# -------------------------
# LOAD DATA
# -------------------------
def load_tsv(path):
    try:
        # Check if file exists first
        if not os.path.exists(path):
            logging.error(f"Data file not found at: {path}")
            return None
            
        df = pd.read_csv(path, sep="\t")
        df = df.dropna(subset=["mia", "en"])  # Remove rows with missing translations
        
        # Ensure strict string type
        df["mia"] = df["mia"].astype(str).str.strip()
        df["en"] = df["en"].astype(str).str.strip()
        
        # Remove empty strings
        df = df[df["mia"] != ""]
        df = df[df["en"] != ""]
        
        logging.info(f"Loaded {len(df)} rows from {path}")
        return Dataset.from_pandas(df)
    except Exception as e:
        logging.error(f"Error loading {path}: {e}")
        return None

# Paths to your data
datasets = [
    load_tsv("../../../../data/basic_forms.tsv"),
    load_tsv("../../../../data/command_forms.tsv"),
    load_tsv("../../../../data/sentences.tsv"),
    load_tsv("../../../../data/story_book.tsv"),
]

# Filter out None results
datasets = [ds for ds in datasets if ds is not None]

if not datasets:
    logging.error("No data was successfully loaded. Exiting.")
    exit()

full_dataset = concatenate_datasets(datasets)
logging.info(f"Total combined dataset size: {len(full_dataset)}")

# Split Data
full_dataset = full_dataset.train_test_split(test_size=0.3, seed=42)
logging.info(f"Train size: {len(full_dataset['train'])}, Test size: {len(full_dataset['test'])}")

# -------------------------
# MODEL & TOKENIZER
# -------------------------
logging.info(f"Loading model: {model_name}")
model = MT5ForConditionalGeneration.from_pretrained(model_name)

logging.info(f"Loading tokenizer: {model_name}")
# Use Fast tokenizer
tokenizer = MT5TokenizerFast.from_pretrained(model_name)

# -------------------------
# PREPROCESSING
# -------------------------
def preprocess(batch):
    # Prefix helps T5-style models understand the task
    prefix = "translate English to Myaamia: "
    
    src_texts = [f"{prefix}{text}" for text in batch["en"]]
    tgt_texts = batch["mia"]

    # Tokenize inputs
    model_inputs = tokenizer(src_texts, max_length=256, truncation=True)

    # Tokenize targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(tgt_texts, max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

logging.info("Tokenizing dataset...")
tokenized_dataset = full_dataset.map(preprocess, batched=True)
logging.info("Tokenization complete.")

# -------------------------
# METRICS (SacreBLEU & ChrF only)
# -------------------------
metric_bleu = evaluate.load("sacrebleu")
metric_chrf = evaluate.load("chrf")

def postprocess_text(preds, labels):
    preds = [p.strip() for p in preds]
    labels = [[l.strip()] for l in labels]
    return preds, labels

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    
    # Handle tuple predictions (if model outputs logits + past_key_values)
    if isinstance(preds, tuple):
        preds = preds[0]

    # If predict_with_generate=True, preds are token IDs.
    # Take argmax only if we erroneously receive logits (3 dimensions).
    if isinstance(preds, np.ndarray) and preds.ndim == 3:
        preds = np.argmax(preds, axis=-1)

    # Replace -100 (ignore index) with pad token
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    # Decode
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    # 1. SacreBLEU
    try:
        bleu_result = metric_bleu.compute(predictions=decoded_preds, references=decoded_labels)
        bleu_score = bleu_result["score"]
    except Exception as e:
        logging.warning(f"SacreBLEU calculation failed: {e}")
        bleu_score = 0.0

    # 2. chrF++
    try:
        chrf_result = metric_chrf.compute(predictions=decoded_preds, references=decoded_labels, word_order=2)
        chrf_score = chrf_result["score"]
    except Exception as e:
        logging.warning(f"chrF++ calculation failed: {e}")
        chrf_score = 0.0

    result = {
        "bleu": round(bleu_score, 4),
        "chrf++": round(chrf_score, 4),
    }

    # Log to TSV
    try:
        # Write header if file doesn't exist
        if not os.path.exists(eval_file):
            with open(eval_file, "w") as f:
                f.write("BLEU\tChrF++\n")
        
        with open(eval_file, "a") as f:
            f.write(f"{result['bleu']}\t{result['chrf++']}\n")
    except Exception as e:
        logging.warning(f"Could not write to eval_file: {e}")

    return result

# -------------------------
# TRAINING ARGS
# -------------------------
training_args = Seq2SeqTrainingArguments(
    output_dir=save_dir,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    warmup_ratio=0.06,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=40,
    predict_with_generate=True,
    logging_dir="./logs",
    logging_strategy="steps",
    logging_steps=10,
    fp16=False,
    report_to="wandb",
    run_name="mt5-myaamia-finetune",
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
# EXECUTION
# -------------------------
if __name__ == "__main__":
    logging.info("Starting training...")
    trainer.train()
    logging.info("Training complete.")

    # Save final model
    final_path = os.path.join(save_dir, "final_model")
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    logging.info(f"Final model saved to {final_path}")

    # Finish wandb
    wandb.finish()