import os
import pandas as pd
import logging
import torch
import numpy as np
import evaluate
from datasets import Dataset, concatenate_datasets
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import PeftModel  # We need this to load the LoRA adapter
from tqdm import tqdm  # For a nice progress bar

# Configure logging
logging.basicConfig(level=logging.INFO)

# --- 1. Configuration ---
# ⬇️ *** TODO: SET THIS TO YOUR CHECKPOINT PATH *** ⬇️
# Example: "llama-3.2-1b-en-mia-translation/checkpoint-200"
ADAPTER_PATH = "/home/chaudhy/LinguaBridge/test/llama-3/myaamia/test2/llama-3.2-1b-en-mia-translation"

# This is the base model your adapter was trained on
MODEL_ID = "meta-llama/Llama-3.2-1B"

# Set which GPU to use for evaluation
EVAL_GPU = "3"

# Set a batch size for evaluation. Adjust this based on your VRAM.
# 4, 8, or 16 are good values to try.
EVAL_BATCH_SIZE = 8

# --- 2. Set Target GPU ---
os.environ["CUDA_VISIBLE_DEVICES"] = EVAL_GPU
DEVICE = "cuda"

# --- 3. Data Loading Function (Copied from training script) ---
def load_tsv(path):
    """Loads a TSV file into a Hugging Face Dataset."""
    try:
        df = pd.read_csv(path, sep="\t")
        df = df.dropna(subset=["mia", "en"])
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

# --- 4. Load the *Exact Same* Test Split ---
# This logic is identical to your training script to ensure
# we get the *exact same* 1,910 test examples.

data_files = [
    "../../../data/basic_forms.tsv",
    "../../../data/command_forms.tsv",
    "../../../data/sentences.tsv",
    "../../../data/story_book.tsv",
]

loaded_datasets = [load_tsv(path) for path in data_files]
valid_datasets = [ds for ds in loaded_datasets if ds is not None]

if not valid_datasets:
    raise ValueError("No valid datasets were loaded. Please check your file paths.")

full_dataset = concatenate_datasets(valid_datasets)
full_dataset = full_dataset.shuffle(seed=42) # The crucial step

# Re-create the 90/10 split
dataset_splits = full_dataset.train_test_split(test_size=0.1)
test_data = dataset_splits["test"]

logging.info(f"Successfully loaded {len(test_data)} unique test examples.")

# --- 5. Load Model and Tokenizer ---
logging.info(f"Loading base model: {MODEL_ID}")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,
    device_map={"": DEVICE} # Load the whole model onto our target GPU
)

logging.info(f"Loading adapter and tokenizer from: {ADAPTER_PATH}")
# Load the adapter on top of the base model
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
model = model.eval() # Set model to evaluation mode

# --- 6. Prompt Creation Function (Inference) ---
def create_inference_prompt(english_text):
    """
    Creates the prompt for the model to translate,
    but does *not* include the answer.
    """
    system_prompt = "You are an expert translator. Translate the given English text into Myaamia."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"translate english to myaamia: {english_text}"},
    ]
    
    # apply_chat_template will add the <|start_header_id|>assistant<|end_header_id|>
    # tokens, telling the model to start generating
    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    return prompt

# --- 7. Run Evaluation Loop ---
logging.info("Starting evaluation...")
bleu = evaluate.load("bleu")
all_predictions = []
all_references = []

# Process the test set in batches
for i in tqdm(range(0, len(test_data), EVAL_BATCH_SIZE), desc="Evaluating Batches"):
    batch = test_data[i : i + EVAL_BATCH_SIZE]
    
    # Get the ground-truth Myaamia translations
    references = batch['mia']
    
    # Create the input prompts for the model
    prompts = [create_inference_prompt(en_text) for en_text in batch['en']]
    
    # Tokenize the batch of prompts
    inputs = tokenizer(
        prompts, 
        return_tensors="pt", 
        padding=True, 
        truncation=True
    ).to(DEVICE)

    # Generate the model's translations
    with torch.no_grad():
        output_tokens = model.generate(
            **inputs,
            max_new_tokens=75,  # Max length of a Myaamia translation
            pad_token_id=tokenizer.eos_token_id # Important for generation
        )
    
    # We need to decode *only* the newly generated tokens
    input_length = inputs.input_ids.shape[1]
    new_tokens = output_tokens[:, input_length:]
    
    # Decode the predictions
    predictions = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
    
    # Clean and store
    cleaned_preds = [pred.strip() for pred in predictions]
    cleaned_refs = [[ref.strip()] for ref in references]
    
    all_predictions.extend(cleaned_preds)
    all_references.extend(cleaned_refs)


print(f"  Checkpoint: {ADAPTER_PATH}")
print(f"  Test Examples: {len(all_predictions)}")
# --- 8. Compute and Print Final BLEU Score ---
logging.info("Calculating final BLEU score...")
result = bleu.compute(predictions=all_predictions, references=all_references)

sacrebleu = evaluate.load("sacrebleu")
sacrebleu_results = sacrebleu.compute(predictions=all_predictions, references=all_references)
print(f"SacreBLEU Score: {sacrebleu_results['score']:.4f}")

# Load chrF metric
# ChrF++ is chrF with word_order=2 (defaults to chrF if word_order=0)
chrf = evaluate.load("chrf") 
chrf_results = chrf.compute(
    predictions=all_predictions, 
    references=all_references, 
    word_order=2, # This activates ChrF++
    beta=2.0 # Standard beta for ChrF++
)
print(f"ChrF++ Score: {chrf_results['score']:.4f}")


print(f"  BLEU Score: {result['bleu'] * 100:.2f}")
print("---------------------------")


# SacreBLEU Score: 0.0104
# ChrF++ Score: 4.8728
#   BLEU Score: 0.00