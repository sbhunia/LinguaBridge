import os
# -------------------------
# ENV SETUP
# -------------------------
# Ensure this matches your actual available GPU ID. If you have 1 GPU, use "0".
os.environ["CUDA_VISIBLE_DEVICES"] = "5" 
os.environ["WANDB_PROJECT"] = "mt5-myaamia-lora"

import logging
import pandas as pd
import numpy as np
import torch
import nltk
import evaluate
from datasets import Dataset, concatenate_datasets
from transformers import (
    MT5ForConditionalGeneration,
    AutoTokenizer, # Faster than MT5Tokenizer
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
from peft import LoraConfig, get_peft_model, TaskType

# -------------------------
# CONFIG
# -------------------------
model_name = "google/mt5-large"
save_dir = "./mt5_finetuned_myaamia"
log_file = "./training.log"
eval_file = "./mt5_large_eval_results.tsv"

# NLTK Setup
try:
    nltk.download("wordnet", quiet=True)
    nltk.download("punkt", quiet=True)
    nltk.download("omw-1.4", quiet=True)
except Exception:
    pass

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

logging.info("Starting mT5 fine-tuning script with LoRA.")

# -------------------------
# LOAD DATA
# -------------------------
def load_tsv(path):
    try:
        df = pd.read_csv(path, sep="\t")
        # Normalize column names to lowercase just in case
        df.columns = [c.lower() for c in df.columns]
        
        if "mia" not in df.columns or "en" not in df.columns:
            logging.warning(f"Skipping {path}: Missing 'mia' or 'en' columns.")
            return None
            
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

# Update paths relative to your script
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
full_dataset = full_dataset.shuffle(seed=42)
full_dataset = full_dataset.train_test_split(test_size=0.1, seed=42)
logging.info(f"Train size: {len(full_dataset['train'])}, Test size: {len(full_dataset['test'])}")

# -------------------------
# TOKENIZER
# -------------------------
logging.info(f"Loading tokenizer: {model_name}")
# Use AutoTokenizer to get the 'Fast' version (Rust-based) automatically
tokenizer = AutoTokenizer.from_pretrained(model_name)

# -------------------------
# MODEL & LORA SETUP
# -------------------------
logging.info(f"Loading base model: {model_name}")

# Critical for mT5: float16 causes NaN loss. Use bfloat16 (Ampere GPUs) or float32.
use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
torch_dtype = torch.bfloat16 if use_bf16 else torch.float32

logging.info(f"Using dtype: {torch_dtype} (Bfloat16 supported: {use_bf16})")

model = MT5ForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch_dtype
)

# Enable gradient checkpointing for memory efficiency
model.gradient_checkpointing_enable()
model.config.use_cache = False 

# LoRA Configuration
lora_config = LoraConfig(
    r=16, 
    lora_alpha=32,
    # mT5 uses Gated GeLU, so we target the specific dense layers
    target_modules=["q", "v", "wi_0", "wi_1", "wo", "k"], 
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# -------------------------
# PREPROCESSING
# -------------------------
def preprocess(batch):
    # Prefix helps the model distinguish this task, even if not pre-trained with it
    prefix = "translate English to Myaamia: "
    src_texts = [f"{prefix}{text}" for text in batch["en"]]
    tgt_texts = batch["mia"]
    
    model_inputs = tokenizer(src_texts, max_length=128, truncation=True)
    labels = tokenizer(text_target=tgt_texts, max_length=128, truncation=True)
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

logging.info("Tokenizing dataset...")

# Dynamically find columns to remove so it doesn't crash on hidden columns
column_names = full_dataset["train"].column_names
tokenized_dataset = full_dataset.map(
    preprocess, 
    batched=True, 
    remove_columns=column_names
)
logging.info("Tokenization complete.")

# -------------------------
# METRICS
# -------------------------
sacrebleu = evaluate.load("sacrebleu")
chrf_metric = evaluate.load("chrf")

# Initialize Eval File with Headers
with open(eval_file, "w") as f:
    f.write("bleu\tchrf++\tgen_len\n")

def postprocess_text(preds, labels):
    preds = [p.strip() for p in preds]
    labels = [[l.strip()] for l in labels]
    return preds, labels

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    if isinstance(preds, tuple):
        preds = preds[0]

    # Replace -100 with pad token for decoding
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    bleu_result = sacrebleu.compute(predictions=decoded_preds, references=decoded_labels)
    try:
        chrf_result = chrf_metric.compute(predictions=decoded_preds, references=decoded_labels, word_order=2)
        chrf_score = chrf_result["score"]
    except Exception:
        chrf_score = 0.0

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    
    result = {
        "bleu": round(bleu_result["score"], 4),
        "chrf++": round(chrf_score, 4),
        "gen_len": round(np.mean(prediction_lens), 4),
    }

    try:
        with open(eval_file, "a") as f:
            f.write(f"{result['bleu']}\t{result['chrf++']}\t{result['gen_len']}\n")
    except Exception as e:
        logging.warning(f"Could not write to eval file: {e}")


    return result

# -------------------------
# TRAINER
# -------------------------
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    label_pad_token_id=-100,
    pad_to_multiple_of=8
)

training_args = Seq2SeqTrainingArguments(
    output_dir=save_dir,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=3e-4,
    num_train_epochs=40,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,     # Load the best checkpoint when finished
    metric_for_best_model="chrf++",  # Optimize for chrf++ (better for morphology)
    greater_is_better=True,
    predict_with_generate=True,
    logging_dir="./logs",
    logging_strategy="steps",
    logging_steps=10,
    report_to="wandb",
    fp16=not use_bf16,
    bf16=use_bf16,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    # callbacks=[EarlyStoppingCallback(early_stopping_patience=5)] # Stop if no improvement for 5 evaluations
)

# -------------------------
# EXECUTION
# -------------------------
if __name__ == "__main__":
    logging.info("Starting training...")
    trainer.train()

    logging.info(f"Saving LoRA adapter to {save_dir}")
    trainer.save_model(save_dir)
    tokenizer.save_pretrained(save_dir)

    logging.info("Training complete. NOTE: This saved only the adapter. To use for inference:")
    logging.info("model = PeftModel.from_pretrained(MT5ForConditionalGeneration.from_pretrained('google/mt5-large'), save_dir)")


#     {'eval_loss': 0.8620008826255798, 'eval_bleu': 16.5019, 'eval_chrf++': 62.7011, 'eval_gen_len': 7.0754, 'eval_runtime': 93.909, 'eval_samples_per_second': 20.339, 'eval_steps_per_second': 1.278, 'epoch': 40.0}          
# {'train_runtime': 37340.5294, 'train_samples_per_second': 18.41, 'train_steps_per_second': 1.152, 'train_loss': 1.0406380686066872, 'epoch': 40.0}                                                                         