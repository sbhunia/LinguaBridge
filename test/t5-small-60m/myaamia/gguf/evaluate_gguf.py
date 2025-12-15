import os
import pandas as pd
import logging
import subprocess
import evaluate
from datasets import Dataset, concatenate_datasets
from tqdm import tqdm
import re

# Configure logging
logging.basicConfig(level=logging.INFO)

# --- 1. Configuration ---
# ⬇️ *** VERIFY YOUR PATHS HERE *** ⬇️
GGUF_MODEL_PATH = "/home/yogesh/LinguaBridge/test/llama-3/llama.cpp/mt5_large_Q4_K_M.gguf"
LLAMA_CLI_PATH = "/home/yogesh/LinguaBridge/test/llama-3/llama.cpp/build/bin/llama-cli"

# --- 2. Data Loading ---
def load_tsv(path):
    try:
        df = pd.read_csv(path, sep="\t")
        df = df.dropna(subset=["mia", "en"])
        df["mia"] = df["mia"].astype(str)
        df["en"] = df["en"].astype(str)
        return Dataset.from_pandas(df)
    except FileNotFoundError:
        logging.error(f"Data file not found at: {path}")
        return None

data_files = [
    "../../../../data/basic_forms.tsv",
    "../../../../data/command_forms.tsv",
    "../../../../data/sentences.tsv",
    "../../../../data/story_book.tsv",
]

datasets = [load_tsv(path) for path in data_files]
datasets = [ds for ds in datasets if ds is not None]

if not datasets:
    raise ValueError("No valid datasets loaded.")

full_dataset = concatenate_datasets(datasets)
dataset_splits = full_dataset.train_test_split(test_size=0.3, seed=42)
test_data = dataset_splits["test"]

logging.info(f"Loaded {len(test_data)} test examples.")

# --- 3. Robust T5 Inference Function ---
def run_t5_inference(text_input):
    """
    Runs llama-cli WITHOUT suppressing logs, then cleans the output in Python.
    """
    prompt = f"translate English to Myaamia: {text_input}"
    
    # REMOVED: --log-disable (since it hides output)
    # ADDED: --no-display-prompt (keeps the prompt out of the output)
    command = [
        LLAMA_CLI_PATH,
        "-m", GGUF_MODEL_PATH,
        "-p", prompt,
        "-n", "128",
        "--temp", "0.1",
        "--no-display-prompt", 
        "-c", "512" 
    ]

    try:
        # Capture stdout (text) and stderr (logs) separately
        result = subprocess.run(
            command,
            capture_output=True, 
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        
        # In many llama-cli versions, the actual text is in stdout, logs in stderr.
        # But sometimes they mix. We focus on STDOUT first.
        raw_output = result.stdout.strip()
        
        # --- CLEANING LOGIC ---
        # 1. Remove the specific "[end of text]" marker
        clean_output = raw_output.replace("[end of text]", "")
        
        # 2. Split into lines and filter out system logs just in case they leaked into stdout
        lines = clean_output.split('\n')
        filtered_lines = []
        for line in lines:
            # Skip lines that look like logs
            if (line.startswith("sampler") or 
                line.startswith("generate:") or 
                line.startswith("llama_perf") or 
                line.startswith("llama_memory") or
                line.startswith("system_info") or
                line.strip() == ""):
                continue
            filtered_lines.append(line)
            
        # Join whatever is left (usually just one line: "wiiyaahkweeta")
        final_prediction = " ".join(filtered_lines).strip()
        
        return final_prediction

    except Exception as e:
        logging.error(f"Error running model: {e}")
        return ""

# --- 4. Evaluation Loop ---
logging.info("Starting evaluation...")

all_predictions = []
all_references = []

# Debug print for the first item to ensure it's working
print("\n--- TEST RUN (First Item) ---")
first_ex = test_data[0]
print(f"Input: {first_ex['en']}")
pred = run_t5_inference(first_ex['en'])
print(f"Cleaned Prediction: '{pred}'")
print("-" * 30 + "\n")

# Main Loop
for example in tqdm(test_data, desc="Evaluating"):
    prediction = run_t5_inference(example['en'])
    all_predictions.append(prediction)
    all_references.append([example['mia']])

# --- 5. Metrics & Saving ---
logging.info("Calculating scores...")

sacrebleu = evaluate.load("sacrebleu")
sacrebleu_results = sacrebleu.compute(predictions=all_predictions, references=all_references)

chrf = evaluate.load("chrf") 
chrf_results = chrf.compute(
    predictions=all_predictions, 
    references=all_references, 
    word_order=2,
    beta=2.0
)

model_name = os.path.basename(GGUF_MODEL_PATH)
output_filename = f"eval_results_mt5_large_gguf_fixed_2.txt"

result_text = (
    "\n" + "="*30 + "\n"
    f"RESULTS FOR: {model_name}\n"
    + "="*30 + "\n"
    f"Test Examples: {len(all_predictions)}\n"
    f"SacreBLEU:     {sacrebleu_results['score']:.4f}\n"
    f"ChrF++:        {chrf_results['score']:.4f}\n"
    + "="*30 + "\n"
)

print(result_text)
with open(output_filename, "w", encoding="utf-8") as f:
    f.write(result_text)