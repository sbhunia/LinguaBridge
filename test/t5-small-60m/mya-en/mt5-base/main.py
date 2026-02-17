import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
import logging
import pandas as pd
from datasets import Dataset, concatenate_datasets
# ➤ CHANGED: Import T5 model and tokenizer instead of Marian
from transformers import MT5ForConditionalGeneration, MT5Tokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
import evaluate
import numpy as np
import nltk
import bert_score
from statistics import mean
import wandb  # ➤ ADDED: Import wandb

# -------------------------
# CONFIG
# -------------------------
# ➤ CHANGED: Model name is now mt5-small
model_name = "google/mt5-base"
# ➤ CHANGED: Save directory is updated
save_dir = "./mt5_finetuned_myaamia"
log_file = "./training.log"
eval_file = "./eval_results.tsv"
nltk.download("wordnet", quiet=True)

# ➤ ADDED: Set the project name for Weights & Biases
os.environ["WANDB_PROJECT"] = "mt5-base"

# -------------------------
# LOGGER
# -------------------------
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.info("Starting mT5 fine-tuning script.")

# -------------------------
# LOAD DATA
# -------------------------
def load_tsv(path):
    try:
        df = pd.read_csv(path, sep="\t")
        df = df.dropna(subset=["mia", "en"])  # Remove rows with missing translations
        df["mia"] = df["mia"].astype(str)
        df["en"] = df["en"].astype(str)
        logging.info(f"Loaded {len(df)} rows from {path}")
        return Dataset.from_pandas(df)
    except FileNotFoundError:
        logging.error(f"Data file not found at: {path}")
        return None
    except Exception as e:
        logging.error(f"Error loading {path}: {e}")
        return None

datasets = [
    load_tsv("../../../../data/basic_forms.tsv"),
    load_tsv("../../../../data/command_forms.tsv"),
    load_tsv("../../../../data/sentences.tsv"),
    load_tsv("../../../../data/story_book.tsv"),
]

# Filter out any None results from failed loads
datasets = [ds for ds in datasets if ds is not None]

if not datasets:
    logging.error("No data was successfully loaded. Exiting.")
    exit()

full_dataset = concatenate_datasets(datasets)
logging.info(f"Total combined dataset size: {len(full_dataset)}")
full_dataset = full_dataset.train_test_split(test_size=0.3, seed=42)  # 70% train, 30% val
logging.info(f"Train size: {len(full_dataset['train'])}, Test size: {len(full_dataset['test'])}")

# -------------------------
# MODEL
# -------------------------
# ➤ CHANGED: Using MT5ForConditionalGeneration
logging.info(f"Loading model: {model_name}")
model = MT5ForConditionalGeneration.from_pretrained(model_name)

# -------------------------
# TOKENIZER
# -------------------------
# ➤ CHANGED: Using MT5Tokenizer
logging.info(f"Loading tokenizer: {model_name}")
tokenizer = MT5Tokenizer.from_pretrained(model_name)

# ➤ CHANGED: Removed the Marian-style '>>mia<<' token logic.
# T5 models do not use special language tokens in this way.
# Instead, we use a text prefix (e.g., "translate English to Myaamia:")
# to tell the model what task to perform.
#
# --- REMOVED BLOCK ---
# new_token = ">>mia<<"
# if new_token not in tokenizer.get_vocab():
#     tokenizer.add_tokens([new_token])
#     model.resize_token_embeddings(len(tokenizer))
# --- END REMOVED BLOCK ---


# ➤ CHANGED: Preprocessing function updated for T5.
def preprocess(batch):
    # T5 models need a task prefix. We'll tell it to translate.
    # The source language is English, the target is Myaamia.
    prefix = "translate Myaamia to English: "
    
    # Source: English (with prefix)
    src_texts = [f"{prefix}{text}" for text in batch["mia"]]
    # Target: Myaamia
    tgt_texts = batch["en"]

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
# METRIC (sacreBLEU, METEOR, BERTScore, chrF++, COMET)
# -------------------------
# ➤ ADDED: Note for installing COMET
# You may need to install the COMET metric dependencies:
# pip install unbabel-comet
metric = evaluate.load("sacrebleu")
# ➤ ADDED: Load chrF
chrf_metric = evaluate.load("chrf")

def postprocess_text(preds, labels):
    preds = [p.strip() for p in preds]
    labels = [[l.strip()] for l in labels]
    return preds, labels

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    if isinstance(preds, tuple):
        preds = preds[0]

    if preds.ndim == 3:
        preds = np.argmax(preds, axis=-1)

    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    # Replace -100 (ignore index) with pad token for decoding
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    # preds = np.clip(preds, 0, tokenizer.vocab_size - 1)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    # sacreBLEU
    bleu_result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    bleu_score = bleu_result["score"]

    # ➤ ADDED: chrF++ (word_order=2)
    try:
        chrf_result = chrf_metric.compute(predictions=decoded_preds, references=decoded_labels, word_order=2)
        chrf_score = chrf_result["score"]
    except Exception as e:
        logging.warning(f"chrF++ calculation failed: {e}")
        chrf_score = 0.0


    # Combine results
    result = {
        "bleu": round(bleu_score, 4),
        "chrf++": round(chrf_score, 4), 
    }

    # Log to file
    try:
        with open(eval_file, "a") as f:
            f.write(f"Epoch result: {result}\n")
    except Exception as e:
        logging.warning(f"Could not write to eval_file: {e}")

    return result

# -------------------------
# TRAINING
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
    logging_steps=5,
    fp16=False,  
    report_to="wandb",
    run_name="mt5-myaamia-finetune" # ➤ ADDED: Optional run name for wandb
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
logging.info("Starting training...")
trainer.train()
logging.info("Training complete.")

# Save the final model
trainer.save_model(os.path.join(save_dir, "final_model"))
tokenizer.save_pretrained(os.path.join(save_dir, "final_model"))
logging.info(f"Final model saved to {os.path.join(save_dir, 'final_model')}")

# ➤ ADDED: Finish the wandb run
wandb.finish()