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
    TrainingArguments
)
from peft import LoraConfig
from trl import SFTTrainer

# Configure logging
logging.basicConfig(level=logging.INFO)

# --- 1. Configuration ---
model_id = "meta-llama/Llama-3.2-1B-Instruct"
# NEW: Updated output directory for the translation task
output_dir = "llama-3.2-1b-en-mia-translation"

# Set target GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,4"

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
    "../../../../data/basic_forms.tsv",
    "../../../../data/command_forms.tsv",
    "../../../../data/sentences.tsv",
    "../../../../data/story_book.tsv",
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
    system_prompt = "You are an expert translator. Translate the given English text into Myaamia."
    
    user_text = example.get('en', '')
    assistant_text = example.get('mia', '')

    # Prevent training on empty/corrupt examples
    if not user_text or not assistant_text:
        return ""  # Return an empty string to be skipped

    # Create the list of messages
    messages = [
        {"role": "system", "content": system_prompt},
        # Simple instruction for the model
        {"role": "user", "content": f"translate english to myaamia: {user_text}"},
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

# --- 7. Define Evaluation Metrics (Changed to BLEU) ---
bleu = evaluate.load("bleu")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    
    # predictions are logits, let's argmax to get token ids
    pred_logits = predictions[0] if isinstance(predictions, tuple) else predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    # Decode predictions
    decoded_preds = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    
    # Prepare labels: replace -100 (ignored index) with pad_token_id for decoding
    label_ids = np.where(labels == -100, tokenizer.pad_token_id, labels)
    decoded_labels = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    
    # Clean up and format for BLEU
    decoded_preds = [pred.strip() for pred in decoded_preds]
    # BLEU expects references to be a list of lists (for multiple possible references)
    decoded_labels = [[label.strip()] for label in decoded_labels] 

    # Filter out empty pairs
    final_preds = []
    final_labels = []
    for pred, label_list in zip(decoded_preds, decoded_labels):
        if label_list[0]: # If the label is not an empty string
            final_preds.append(pred)
            final_labels.append(label_list)

    if not final_preds or not final_labels:
         return {"bleu": 0.0}

    result = bleu.compute(predictions=final_preds, references=final_labels)
    
    return {"bleu": result["bleu"]}

# --- 8. Configure Training Arguments (Updated) ---
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=1,  # Enabled evaluation
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=1,  # You may want to increase this for a new task
    lr_scheduler_type="cosine",
    logging_steps=25,
    # eval_strategy="steps",        # Enabled evaluation
    # eval_steps=100,             # Enabled evaluation
    save_strategy="steps",
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
    # eval_dataset=test_data,
    peft_config=peft_config,
    formatting_func=format_translation_prompt, # Use the NEW function
    # compute_metrics=compute_metrics,           # Add the metrics function
)

# --- 10. Start Training ---
print("Starting translation model training...")
trainer.train()

# --- 11. Save the final adapter model ---
print("Saving model...")
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

print("Training complete. Model and tokenizer saved to:", output_dir)