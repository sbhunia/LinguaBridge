import os
import pandas as pd
import logging
from llama_cpp import Llama
import evaluate
from datasets import Dataset, concatenate_datasets
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)

# --- 1. Configuration ---
# ⬇️ *** TODO: SET THIS TO YOUR GGUF MODEL PATH *** ⬇️
# Example: "/path/to/your/model-q4_k_m.gguf"
GGUF_MODEL_PATH = "/home/yogesh/LinguaBridge/test/llama-3/llama.cpp/llama3_1b_finetune_Q4_K_M.gguf"

# GPU Settings for llama-cpp-python
# -1 loads all layers to GPU. If you run out of VRAM, lower this number (e.g., 20, 30).
N_GPU_LAYERS = -1 
CONTEXT_WINDOW = 2048 

# --- 2. Data Loading Function (Identical to original) ---
def load_tsv(path):
    """Loads a TSV file into a Hugging Face Dataset."""
    try:
        df = pd.read_csv(path, sep="\t")
        df = df.dropna(subset=["mia", "en"])
        df.loc[:, "mia"] = df["mia"].astype(str).str.strip()
        df.loc[:, "en"] = df["en"].astype(str).str.strip()
        logging.info(f"Loaded {len(df)} rows from {path}")
        return Dataset.from_pandas(df)
    except FileNotFoundError:
        logging.error(f"Data file not found at: {path}")
        return None
    except Exception as e:
        logging.error(f"Error loading {path}: {e}")
        return None

# --- 3. Load the *Exact Same* Test Split ---
# We replicate the exact shuffling logic to ensure the test set is identical
data_files = [
    "../../../../data/basic_forms.tsv",
    "../../../../data/command_forms.tsv",
    "../../../../data/sentences.tsv",
    "../../../../data/story_book.tsv",
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

# --- 4. Load GGUF Model ---
logging.info(f"Loading GGUF model from: {GGUF_MODEL_PATH}")

# Initialize the GGUF model
llm = Llama(
    model_path=GGUF_MODEL_PATH,
    n_gpu_layers=N_GPU_LAYERS, # Offload to GPU
    n_ctx=CONTEXT_WINDOW,      # Context window
    verbose=False              # Reduce noise in logs
)

# --- 5. Run Evaluation Loop ---
logging.info("Starting evaluation...")

all_predictions = []
all_references = []

# Define the System Prompt
system_prompt = "You are an expert translator. Translate the given English text into Myaamia."

# Iterate through test data one by one (Standard for GGUF)
for example in tqdm(test_data, desc="Evaluating GGUF"):
    english_text = example['en']
    reference_text = example['mia']

    # Create the chat message structure
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"translate english to myaamia: {english_text}"}
    ]

    # Generate prediction
    # create_chat_completion automatically handles the prompt formatting (System/User/Assistant tags)
    response = llm.create_chat_completion(
        messages=messages,
        max_tokens=128,    # Limit generation length
        temperature=0.1,  # Low temperature for deterministic/greedy-like decoding
        top_p=0.9,
        stop=["<|eot_id|>", "\n"] # Stop tokens to prevent rambling
    )

    # Extract the text content from the response
    prediction = response['choices'][0]['message']['content'].strip()

    all_predictions.append(prediction)
    all_references.append([reference_text]) # References must be a list of lists for metrics

# --- 6. Compute Metrics ---
logging.info("Calculating final scores...")

# 1. SacreBLEU
sacrebleu = evaluate.load("sacrebleu")
sacrebleu_results = sacrebleu.compute(predictions=all_predictions, references=all_references)

# 2. ChrF++
chrf = evaluate.load("chrf") 
chrf_results = chrf.compute(
    predictions=all_predictions, 
    references=all_references, 
    word_order=2, # This activates ChrF++
    beta=2.0      # Standard beta for ChrF++
)

# --- 7. Print and Save Results ---
logging.info("Saving results...")

# Define the output filename (e.g., based on the model name)
model_name = os.path.basename(GGUF_MODEL_PATH)
output_filename = f"eval_results_{model_name}.txt"

# Create the formatted result string
result_text = (
    "\n" + "="*30 + "\n"
    f"RESULTS FOR GGUF: {model_name}\n"
    + "="*30 + "\n"
    f"Test Examples: {len(all_predictions)}\n"
    f"SacreBLEU:     {sacrebleu_results['score']:.4f}\n"
    f"ChrF++:        {chrf_results['score']:.4f}\n"
    + "="*30 + "\n"
)

# 1. Print to console
print(result_text)

# 2. Save to file
with open(output_filename, "w", encoding="utf-8") as f:
    f.write(result_text)

logging.info(f"Results saved to {output_filename}")

# Results:
#  ""/home/yogesh/LinguaBridge/test/llama-3/llama.cpp/llama3_1b_finetune_Q4_K_M.gguf""