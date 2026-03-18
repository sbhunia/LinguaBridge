import os
import pandas as pd
import logging
import torch
import numpy as np
import evaluate
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    TrainerCallback
)
import json
from peft import LoraConfig
from trl import SFTTrainer

# Configure logging
logging.basicConfig(level=logging.INFO)

# --- 1. Configuration ---
model_id = "meta-llama/Llama-3.2-1B-Instruct"
# NEW: Updated output directory for the translation task
output_dir = "llama-3.2-1b-mia-en-translation"
eval_log_file = os.path.join(output_dir, "evaluation_metrics.jsonl")

# Set target GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

# --- 2. Load Tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

# --- 3. Load Model ---
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    device_map="auto"
)

# --- 4. Load Local Dataset ---
def load_tsv(path):
    """Loads a TSV file into a Hugging Face Dataset."""
    try:
        df = pd.read_csv(path, sep="\t")
        df = df.dropna(subset=["mia", "en"])  # Remove rows with missing translations
        df["mia"] = df["mia"].astype(str).str.strip()
        df["en"] = df["en"].astype(str).str.strip()
        logging.info(f"Loaded {len(df)} rows from {path}")
        return Dataset.from_pandas(df)
    except FileNotFoundError:
        logging.error(f"Data file not found at: {path}")
        return None
    except Exception as e:
        logging.error(f"Error loading {path}: {e}")
        return None

# ⬇️ *** ADJUST THESE PATHS TO MATCH YOUR FILE SYSTEM ***
data_files = [
    "../../../data/basic_forms.tsv",
    "../../../data/command_forms.tsv",
    "../../../data/sentences.tsv",
    "../../../data/story_book.tsv",
]

# Load all datasets
loaded_datasets = [load_tsv(path) for path in data_files]

# Filter out any that failed to load (None)
valid_datasets = [ds for ds in loaded_datasets if ds is not None]

if not valid_datasets:
    raise ValueError("No valid datasets were loaded. Please check your file paths.")

# Combine all valid datasets into one
full_dataset = concatenate_datasets(valid_datasets)

# Shuffle the dataset before splitting
full_dataset = full_dataset.shuffle(seed=42)

# Create train/test split
dataset_splits = full_dataset.train_test_split(test_size=0.1)
train_data = dataset_splits["train"]
test_data = dataset_splits["test"]

logging.info(f"Total examples: {len(full_dataset)}")
logging.info(f"Training examples: {len(train_data)}")
logging.info(f"Test examples: {len(test_data)}")

# --- 5. Formatting Function for Translation ---
def format_translation_prompt(example):
    """
    Formats the dataset example for a translation task
    and returns a single, formatted string.
    """
    system_prompt = "You are an expert translator. Translate the given Myaamia text into English."
    
    user_text = example.get('mia', '')
    assistant_text = example.get('en', '')

    # Prevent training on empty/corrupt examples
    if not user_text or not assistant_text:
        return ""  # Return an empty string to be skipped

    # Create the list of messages
    messages = [
        {"role": "system", "content": system_prompt},
        # Simple instruction for the model
        {"role": "user", "content": f"translate myaamia to english: {user_text}"},
        {"role": "assistant", "content": assistant_text}
    ]
    
    # Apply the chat template
    try:
        formatted_string = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=False # This is crucial for training
        )
        return formatted_string
    except Exception as e:
        print(f"Error applying chat template: {e}")
        return ""

# --- 6. Configure PEFT (LoRA) (Unchanged) ---
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# --- 7. Define Evaluation Metrics ---
sacrebleu = evaluate.load("sacrebleu")
chrf = evaluate.load("chrf")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    # Handle logits
    pred_logits = predictions[0] if isinstance(predictions, tuple) else predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    # Replace ignored index
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    # Decode
    decoded_preds = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = [p.strip() for p in decoded_preds]
    decoded_labels = [l.strip() for l in decoded_labels]

    # Remove empty labels
    preds = []
    refs = []

    for p, l in zip(decoded_preds, decoded_labels):
        if l:
            preds.append(p)
            refs.append([l])

    if len(preds) == 0:
        return {"sacrebleu": 0.0, "chrf++": 0.0}

    bleu_result = sacrebleu.compute(predictions=preds, references=refs)
    chrf_result = chrf.compute(predictions=preds, references=refs)

    return {
        "sacrebleu": bleu_result["score"],
        "chrf++": chrf_result["score"]
    }

class EvalLoggerCallback(TrainerCallback):
    def __init__(self, output_file):
        self.output_file = output_file

        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)

        # create file and header if it doesn't exist
        if not os.path.exists(self.output_file):
            with open(self.output_file, "w") as f:
                pass

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return

        record = {
            "epoch": state.epoch,
            **metrics
        }

        with open(self.output_file, "a") as f:
            f.write(json.dumps(record) + "\n")

# --- 8. Configure Training Arguments (Updated) ---
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=4,  # Enabled evaluation
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=5,  # You may want to increase this for a new task
    lr_scheduler_type="cosine",
    logging_steps=25,
    eval_strategy="epoch",        # Enabled evaluation
    # eval_steps=100,             # Enabled evaluation
    save_strategy="epoch",
    save_steps=100,
    report_to="wandb",
    bf16=True,
    # bf16_full_eval=True,
    save_total_limit=2,
    # load_best_model_at_end=True,  # Enabled
    # metric_for_best_model="bleu", # Changed to BLEU
    # greater_is_better=True,       # BLEU score is better when higher
)

# --- 9. SFTTrainer Initialization (Updated) ---
# This uses the older API arguments (processing_class)
# that your original script was set up for.
trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,  # Using your original fix
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data,
    peft_config=peft_config,
    formatting_func=format_translation_prompt, # Use the NEW function
    compute_metrics=compute_metrics,           # Add the metrics function
    callbacks=[EvalLoggerCallback(eval_log_file)]
)

# --- 10. Start Training ---
print("Starting translation model training...")
trainer.train()

# --- 11. Save the final adapter model ---
print("Saving model...")
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

print("Training complete. Model and tokenizer saved to:", output_dir)