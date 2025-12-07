import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5" # Keep this to respect your environment
import logging
import pandas as pd
from datasets import Dataset, concatenate_datasets
from transformers import MT5ForConditionalGeneration, MT5Tokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
import evaluate
import numpy as np
import nltk
import bert_score
from statistics import mean
import wandb

# -------------------------
# CONFIG
# -------------------------
# Same model name, but we will not train it
model_name = "google/mt5-base"
# A new log file for the base model evaluation
log_file = "./base_model_eval.log"
eval_file = "./base_model_eval_results.tsv"
nltk.download("wordnet", quiet=True)

# We will not initialize a wandb run for the base evaluation

# -------------------------
# LOGGER
# -------------------------
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.info("Starting mT5 Base Model Evaluation script.")

# -------------------------
# LOAD DATA (Identical to Training Script)
# -------------------------
def load_tsv(path):
    try:
        df = pd.read_csv(path, sep="\t")
        df = df.dropna(subset=["mia", "en"])
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

datasets = [ds for ds in datasets if ds is not None]

if not datasets:
    logging.error("No data was successfully loaded. Exiting.")
    exit()

full_dataset = concatenate_datasets(datasets)
logging.info(f"Total combined dataset size: {len(full_dataset)}")

# **CRITICAL:** Use the SAME seed (42) to ensure the exact same split
full_dataset = full_dataset.train_test_split(test_size=0.3, seed=42)
logging.info(f"Train size: {len(full_dataset['train'])}, Test size: {len(full_dataset['test'])}")
test_dataset = full_dataset["test"] # Only need the test split for evaluation

# -------------------------
# MODEL & TOKENIZER
# -------------------------
logging.info(f"Loading model: {model_name}")
# Load the model, but we don't need the training arguments
model = MT5ForConditionalGeneration.from_pretrained(model_name)

logging.info(f"Loading tokenizer: {model_name}")
tokenizer = MT5Tokenizer.from_pretrained(model_name)

# -------------------------
# PREPROCESSING (Identical to Training Script)
# -------------------------
def preprocess(batch):
    prefix = "translate English to Myaamia: "
    src_texts = [f"{prefix}{text}" for text in batch["en"]]
    tgt_texts = batch["mia"]

    model_inputs = tokenizer(src_texts, max_length=256, truncation=True)

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(tgt_texts, max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

logging.info("Tokenizing test dataset...")
# Only map the test set for evaluation
tokenized_test_dataset = test_dataset.map(preprocess, batched=True)
logging.info("Tokenization complete.")

# -------------------------
# METRIC (Identical to Training Script)
# -------------------------
metric = evaluate.load("sacrebleu")
meteor_metric = evaluate.load("meteor")
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
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    # sacreBLEU
    bleu_result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    bleu_score = bleu_result["score"]

    # METEOR (using NLTK implementation from your original code)
    meteor_scores = [
        nltk.translate.meteor_score.single_meteor_score(ref[0].split(), hyp.split())
        for hyp, ref in zip(decoded_preds, decoded_labels)
    ]
    meteor_score = mean(meteor_scores)

    # BERTScore
    try:
        # Note: bert_score.score takes a list of predictions and a list of references (where each reference is a string)
        P, R, F1 = bert_score.score(decoded_preds, [ref[0] for ref in decoded_labels],
                                    rescale_with_baseline=True)
        bert_f1 = float(F1.mean())
    except Exception as e:
        logging.warning(f"BERTScore calculation failed: {e}")
        bert_f1 = 0.0

    # chrF++
    try:
        chrf_result = chrf_metric.compute(predictions=decoded_preds, references=decoded_labels, word_order=2)
        chrf_score = chrf_result["score"]
    except Exception as e:
        logging.warning(f"chrF++ calculation failed: {e}")
        chrf_score = 0.0

    # Combine results
    result = {
        "bleu": round(bleu_score, 4),
        "meteor": round(meteor_score, 4),
        "bert_score": round(bert_f1, 4),
        "chrf++": round(chrf_score, 4),
    }

    # Log to file
    try:
        with open(eval_file, "w") as f: # Use 'w' to overwrite the result
            f.write(f"Base Model Evaluation Result: {result}\n")
    except Exception as e:
        logging.warning(f"Could not write to eval_file: {e}")

    return result

# -------------------------
# EVALUATION SETUP
# -------------------------

# Define minimal arguments, only needed for the trainer object
eval_args = Seq2SeqTrainingArguments(
    output_dir="./base_eval_output", # Temporary directory
    per_device_eval_batch_size=16,
    predict_with_generate=True,
    fp16=False,
    report_to="none", # Do not report to wandb for the baseline run
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# The Trainer is still used for its convenient .evaluate() method
evaluator = Seq2SeqTrainer(
    model=model,
    args=eval_args,
    # Pass an empty train dataset
    train_dataset=full_dataset["train"], 
    eval_dataset=tokenized_test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# -------------------------
# EVALUATE
# -------------------------
logging.info("Starting base model evaluation...")
# Run the evaluation
evaluation_results = evaluator.evaluate()

logging.info("Base Model Evaluation Complete.")
logging.info(f"Results: {evaluation_results}")

# -------------------------
# CLEANUP (Optional, but good practice)
# -------------------------
# Remove the temporary output directory
import shutil
if os.path.exists("./base_eval_output"):
    shutil.rmtree("./base_eval_output")
logging.info("Temporary output directory cleaned up.")

# End of script