import os
# Set target GPU (GPU 4) - Ensure this is done before importing torch/transformers
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
os.environ["WANDB_PROJECT"] = "mt5-myaamia-lora"

import logging
import pandas as pd
import numpy as np
import torch
import nltk
import evaluate
from statistics import mean
from datasets import Dataset, concatenate_datasets
from transformers import (
    MT5ForConditionalGeneration,
    MT5Tokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType

# -------------------------
# CONFIG
# -------------------------
model_name = "google/mt5-large"
save_dir = "./mt5_finetuned_myaamia"
log_file = "./training.log"

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
# Also print to console
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger("").addHandler(console)

logging.info("Starting mT5 fine-tuning script with LoRA (Corrected Version).")

# -------------------------
# LOAD DATA
# -------------------------
def load_tsv(path):
    try:
        df = pd.read_csv(path, sep="\t")
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

# Update these paths relative to your script location
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
# Shuffle and split
full_dataset = full_dataset.shuffle(seed=42)
full_dataset = full_dataset.train_test_split(test_size=0.1, seed=42)
logging.info(f"Train size: {len(full_dataset['train'])}, Test size: {len(full_dataset['test'])}")

# -------------------------
# TOKENIZER
# -------------------------
logging.info(f"Loading tokenizer: {model_name}")
tokenizer = MT5Tokenizer.from_pretrained(model_name)

# -------------------------
# MODEL & LORA SETUP
# -------------------------
logging.info(f"Loading base model: {model_name}")

# FIX: T5/mT5 is unstable in float16. Use bfloat16 if GPU supports it (Ampere+), else float32.
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
    target_modules=["q", "v", "wi_0", "wi_1", "wo", "k"], 
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM
)

# Wrap model with LoRA (automatically handles requires_grad)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: 15,925,248 || all params: 1,245,506,560 || trainable%: 1.2786

# -------------------------
# PREPROCESSING
# -------------------------
def preprocess(batch):
    prefix = "translate English to Myaamia: "
    src_texts = [f"{prefix}{text}" for text in batch["en"]]
    tgt_texts = batch["mia"]
    
    # Tokenize inputs
    model_inputs = tokenizer(src_texts, max_length=128, truncation=True)
    
    # Tokenize targets (Using text_target= is the modern API replacement for as_target_tokenizer)
    labels = tokenizer(text_target=tgt_texts, max_length=128, truncation=True)
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

logging.info("Tokenizing dataset...")
tokenized_dataset = full_dataset.map(
    preprocess, 
    batched=True, 
    remove_columns=["en", "mia", "__index_level_0__"] if "__index_level_0__" in full_dataset["train"].column_names else ["en", "mia"]
)
logging.info("Tokenization complete.")

# -------------------------
# METRICS & EVALUATION
# -------------------------
# Load metrics once
sacrebleu = evaluate.load("sacrebleu")
meteor = evaluate.load("meteor")
bertscore = evaluate.load("bertscore")

def postprocess_text(preds, labels):
    preds = [p.strip() for p in preds]
    labels = [[l.strip()] for l in labels]
    return preds, labels

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    
    if isinstance(preds, tuple):
        preds = preds[0]

    # --- FIX STARTS HERE ---
    # 1. Replace -100 in PREDICTIONS with pad_token_id
    # SentencePiece crashes if it sees -100, so we must mask it out.
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)

    # 2. Decode predictions
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    
    # 3. Replace -100 in LABELS with pad_token_id
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # --- FIX ENDS HERE ---

    # Post-process
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    # Calculate Metrics
    # SacreBLEU
    bleu_result = sacrebleu.compute(predictions=decoded_preds, references=decoded_labels)
    
    # METEOR
    meteor_result = meteor.compute(predictions=decoded_preds, references=decoded_labels)

    # BERTScore
    try:
        bert_result = bertscore.compute(
            predictions=decoded_preds, 
            references=decoded_labels, 
            lang="en" 
        )
        f1_mean = mean(bert_result["f1"])
    except Exception as e:
        logging.error(f"BERTScore failed: {e}")
        f1_mean = 0.0

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]

    return {
        "bleu": bleu_result["score"],
        "meteor": meteor_result["meteor"],
        "bertscore_f1": f1_mean,
        "gen_len": np.mean(prediction_lens),
    }

# -------------------------
# TRAINER SETUP
# -------------------------
# Data Collator (Required to pad sequences dynamically and set padding tokens to -100)
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    label_pad_token_id=-100,
    pad_to_multiple_of=8
)

training_args = Seq2SeqTrainingArguments(
    output_dir=save_dir,
    per_device_train_batch_size=8,   # Adjust based on VRAM
    per_device_eval_batch_size=8,
    learning_rate=3e-4,              # LoRA usually likes higher LR than full fine-tuning
    num_train_epochs=10,             # LoRA converges faster, but needs enough epochs
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    predict_with_generate=True,      # Essential for computing BLEU/METEOR
    generation_max_length=128,
    logging_steps=50,
    report_to="wandb",
    load_best_model_at_end=True,
    metric_for_best_model="bleu",
    fp16=not use_bf16,               # Use fp16 ONLY if bf16 is false
    bf16=use_bf16,                   # Prefer bf16 for T5
)

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
logging.info("Starting training...")
trainer.train()

logging.info(f"Saving model to {save_dir}")
trainer.save_model(save_dir)
tokenizer.save_pretrained(save_dir)

logging.info("Training complete.")

# {'eval_loss': 1.2457020282745361, 'eval_bleu': 2.24390947539603, 'eval_meteor': 0.15456146536632928, 'eval_bertscore_f1': 0.9025402134001567, 'eval_gen_len': 8.851832460732984, 'eval_runtime': 301.354, 'eval_samples_per_second': 6.338, 'eval_steps_per_second': 0.793, 'epoch': 10.0}
# {'train_runtime': 20067.4543, 'train_samples_per_second': 8.564, 'train_steps_per_second': 1.071, 'train_loss': 2.2366395880200796, 'epoch': 10.0} 