import os
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
model_name = "google/mt5-small"
# ➤ CHANGED: Save directory is updated
save_dir = "./mt5_finetuned_myaamia"
log_file = "./training.log"
eval_file = "./eval_results.tsv"
nltk.download("wordnet", quiet=True)

# ➤ ADDED: Set the project name for Weights & Biases
os.environ["WANDB_PROJECT"] = "myaamia-translation-mt5"

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
    load_tsv("../../../data/basic_forms.tsv"),
    load_tsv("../../../data/command_forms.tsv"),
    load_tsv("../../../data/sentences.tsv"),
    load_tsv("../../../data/story_book.tsv"),
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
# VERIFY DATA SPLITS
# -------------------------
print("\n--- Verifying Dataset Splits ---")
print(f"Train set size: {len(full_dataset['train'])}")
print(f"Test set size: {len(full_dataset['test'])}")

print("\nShowing 3 samples from TRAIN set:")
for i in range(3):
    sample = full_dataset['train'][i]
    print(f"  EN: {sample['en']}")
    print(f"  MIA: {sample['mia']}")

print("\nShowing 3 samples from TEST set:")
for i in range(3):
    sample = full_dataset['test'][i]
    print(f"  EN: {sample['en']}")
    print(f"  MIA: {sample['mia']}")

# Check for accidental overlap (optional but good)
train_en_samples = set(full_dataset['train']['en'])
test_en_samples = set(full_dataset['test']['en'])
overlap = train_en_samples.intersection(test_en_samples)
print(f"\nOverlap between train and test 'en' samples: {len(overlap)}")
if len(overlap) > 0:
    print("  WARNING: Overlap detected. This is usually not intended.")
print("--- End of Dataset Verification ---\n")
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
    prefix = "translate English to Myaamia: "
    
    # Source: English (with prefix)
    src_texts = [f"{prefix}{text}" for text in batch["en"]]
    # Target: Myaamia
    tgt_texts = batch["mia"]

    model_inputs = tokenizer(src_texts, max_length=128, truncation=True)

    # Tokenize targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(tgt_texts, max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

logging.info("Tokenizing dataset...")
tokenized_dataset = full_dataset.map(preprocess, batched=True)
logging.info("Tokenization complete.")
# -------------------------
# VERIFY TOKENIZATION
# -------------------------
print("\n--- Verifying Tokenization ---")
print("Checking one sample from the tokenized TRAIN set:")

# Get the first tokenized training sample
sample = tokenized_dataset["train"][0]

# Get the corresponding original sample for comparison
original_sample = full_dataset["train"][0]

print(f"\nOriginal EN: {original_sample['en']}")
print(f"Original MIA: {original_sample['mia']}")

print("\n--- Decoded from Tokenizer ---")
# Decode the input_ids (what the model sees)
decoded_input = tokenizer.decode(sample["input_ids"], skip_special_tokens=True)
print(f"Decoded Input: {decoded_input}")

# Decode the labels (what the model is trained to predict)
decoded_label = tokenizer.decode(sample["labels"], skip_special_tokens=True)
print(f"Decoded Label: {decoded_label}")

# Check if it matches expectations
expected_prefix = "translate English to Myaamia: "
if decoded_input == f"{expected_prefix}{original_sample['en']}" and \
   decoded_label == original_sample['mia']:
    print("\nVerification SUCCESS: Input has correct prefix and label matches original.")
else:
    print("\nVerification FAILED: Decoded text does not match expected output.")
    
print("--- End of Tokenization Verification ---\n")
# -------------------------
# METRIC (sacreBLEU, METEOR, BERTScore, chrF++, COMET)
# -------------------------
# ➤ ADDED: Note for installing COMET
# You may need to install the COMET metric dependencies:
# pip install unbabel-comet
metric = evaluate.load("sacrebleu")
meteor_metric = evaluate.load("meteor")
# ➤ ADDED: Load chrF and COMET metrics
chrf_metric = evaluate.load("chrf")
comet_metric = evaluate.load("comet")

def postprocess_text(preds, labels):
    preds = [p.strip() for p in preds]
    labels = [[l.strip()] for l in labels]
    return preds, labels

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    if isinstance(preds, tuple):
        preds = preds[0]
    
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    # Replace -100 (ignore index) with pad token for decoding
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    # sacreBLEU
    bleu_result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    bleu_score = bleu_result["score"]

    # METEOR
    try:
        # Note: meteor needs a flat list of references, not a list of lists
        meteor_result = meteor_metric.compute(predictions=decoded_preds, references=[ref[0] for ref in decoded_labels])
        meteor_score = meteor_result["meteor"]
    except Exception as e:
        logging.warning(f"METEOR calculation failed: {e}")
        meteor_score = 0.0

    # BERTScore
    # ➤ CHANGED: Removed lang="fr".
    # For a low-resource language like Myaamia, it's best to NOT specify a
    # lang code, which makes BERTScore fall back to its multilingual model.
    try:
        P, R, F1 = bert_score.score(decoded_preds, [ref[0] for ref in decoded_labels],
                                    rescale_with_baseline=True)
        bert_f1 = float(F1.mean())
    except Exception as e:
        logging.warning(f"BERTScore calculation failed: {e}")
        bert_f1 = 0.0

    # ➤ ADDED: chrF++ (word_order=2)
    try:
        chrf_result = chrf_metric.compute(predictions=decoded_preds, references=decoded_labels, word_order=2)
        chrf_score = chrf_result["score"]
    except Exception as e:
        logging.warning(f"chrF++ calculation failed: {e}")
        chrf_score = 0.0

    # ➤ ADDED: COMET
    # COMET requires the source text. We fetch it from the test split.
    try:
        source_texts = full_dataset["test"]["en"]
        # Ensure lengths match
        if len(source_texts) != len(decoded_preds):
             logging.warning(f"COMET: Mismatch in source ({len(source_texts)}) and pred ({len(decoded_preds)}) lengths. Skipping.")
             comet_score = 0.0
        else:
            comet_result = comet_metric.compute(predictions=decoded_preds, references=[ref[0] for ref in decoded_labels], sources=source_texts)
            comet_score = comet_result["mean_score"]
    except Exception as e:
        logging.warning(f"COMET calculation failed: {e}")
        comet_score = 0.0


    # Combine results
    result = {
        "bleu": round(bleu_score, 4),
        "meteor": round(meteor_score, 4),
        "bert_score": round(bert_f1, 4),
        "chrf++": round(chrf_score, 4), # ➤ ADDED
        "comet": round(comet_score, 4)   # ➤ ADDED
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
    learning_rate=1e-5,
    warmup_ratio=0.06,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=20,
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