import os
import pandas as pd
import logging
import torch
import evaluate
from datasets import Dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)

# --- 1. Configuration ---
# ⬇️ *** TODO: SET THIS TO YOUR MERGED MODEL PATH *** ⬇️
# This folder should contain config.json, model.safetensors, tokenizer.json, etc.
MERGED_MODEL_PATH = "meta-llama/Llama-3.2-1B-Instruct"

# Set which GPU to use for evaluation
# EVAL_GPU = "3"
EVAL_BATCH_SIZE = 8

# --- 2. Set Target GPU ---
# os.environ["CUDA_VISIBLE_DEVICES"] = EVAL_GPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- 3. Data Loading Function ---
def load_tsv(path):
    """Loads a TSV file into a Hugging Face Dataset."""
    try:
        df = pd.read_csv(path, sep="\t")
        df = df.dropna(subset=["mia", "en"])
        
        # Use .loc to avoid Pandas warnings
        df.loc[:, "mia"] = df["mia"].astype(str).str.strip()
        df.loc[:, "en"] = df["en"].astype(str).str.strip()
        
        logging.info(f"Loaded {len(df)} rows from {path}")
        return Dataset.from_pandas(df)
    except Exception as e:
        logging.error(f"Error loading {path}: {e}")
        return None

# --- 4. Load the Test Split ---
data_files = [
    "../../../data/basic_forms.tsv",
    "../../../data/command_forms.tsv",
    "../../../data/sentences.tsv",
    "../../../data/story_book.tsv",
]

loaded_datasets = [load_tsv(path) for path in data_files]
valid_datasets = [ds for ds in loaded_datasets if ds is not None]

if not valid_datasets:
    raise ValueError("No valid datasets were loaded.")

full_dataset = concatenate_datasets(valid_datasets)
full_dataset = full_dataset.shuffle(seed=42) 

# Re-create the 90/10 split
dataset_splits = full_dataset.train_test_split(test_size=0.1)
test_data = dataset_splits["test"]

logging.info(f"Successfully loaded {len(test_data)} unique test examples.")

# --- 5. Load Merged Model and Tokenizer ---
logging.info(f"Loading merged model from: {MERGED_MODEL_PATH}")

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MERGED_MODEL_PATH)

# CRITICAL: Set padding side to LEFT for batched inference
tokenizer.padding_side = "left"

# Ensure pad_token is set (Llama 3 often defaults to None or EOS)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load Model
model = AutoModelForCausalLM.from_pretrained(
    MERGED_MODEL_PATH,
    dtype=torch.bfloat16,
    device_map={"": DEVICE} # Load directly to target GPU
)

model.eval() 

# --- 6. Prompt Creation Function ---
def create_inference_prompt(english_text):
    system_prompt = "You are an expert translator. Translate the given English text into Myaamia."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"translate english to myaamia: {english_text}"},
    ]
    
    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    return prompt

# --- 7. Run Evaluation Loop ---
logging.info("Starting evaluation...")
all_predictions = []
all_references = []

for i in tqdm(range(0, len(test_data), EVAL_BATCH_SIZE), desc="Evaluating Batches"):
    batch = test_data[i : i + EVAL_BATCH_SIZE]
    references = batch['mia']
    
    # Prepare inputs
    prompts = [create_inference_prompt(en) for en in batch['en']]
    
    inputs = tokenizer(
        prompts, 
        return_tensors="pt", 
        padding=True, 
        truncation=True
    ).to(DEVICE)

    # Generate
    with torch.no_grad():
        output_tokens = model.generate(
            **inputs,
            max_new_tokens=75,
            pad_token_id=tokenizer.eos_token_id,
            stop_strings=["<|eot_id|>", "\n"], # Prevent rambling
            tokenizer=tokenizer
        )
    
    # Decode only new tokens
    input_length = inputs.input_ids.shape[1]
    new_tokens = output_tokens[:, input_length:]
    predictions = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
    
    # Store results
    all_predictions.extend([p.strip() for p in predictions])
    all_references.extend([[r.strip()] for r in references])

# --- 8. Compute Metrics ---
logging.info("Calculating final scores...")

bleu = evaluate.load("bleu")
sacrebleu = evaluate.load("sacrebleu")
chrf = evaluate.load("chrf")

bleu_score = bleu.compute(predictions=all_predictions, references=all_references)
sacrebleu_score = sacrebleu.compute(predictions=all_predictions, references=all_references)
chrf_score = chrf.compute(predictions=all_predictions, references=all_references, word_order=2)

# --- 9. Print and Save Results ---
model_name = os.path.basename(MERGED_MODEL_PATH)
output_filename = f"eval_results_merged_{model_name}_base-1B.txt"

result_text = (
    "\n" + "="*30 + "\n"
    f"RESULTS FOR MERGED MODEL: {model_name}\n"
    + "="*30 + "\n"
    f"Test Examples: {len(all_predictions)}\n"
    f"BLEU:          {bleu_score['bleu'] * 100:.2f}\n"
    f"SacreBLEU:     {sacrebleu_score['score']:.4f}\n"
    f"ChrF++:        {chrf_score['score']:.4f}\n"
    + "="*30 + "\n"
)

print(result_text)

with open(output_filename, "w", encoding="utf-8") as f:
    f.write(result_text)

logging.info(f"Results saved to {output_filename}")