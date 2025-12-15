import os
import pandas as pd
import logging
import evaluate
from datasets import Dataset, concatenate_datasets
from tqdm import tqdm
from transformers import AutoTokenizer
from optimum.executorch import ExecuTorchModelForSeq2SeqLM

# Configure logging
logging.basicConfig(level=logging.INFO)

# --- 1. Configuration ---
# ⬇️ *** VERIFY YOUR PATHS HERE *** ⬇️
# Point this to the DIRECTORY containing your .pte file (e.g., "mt5_large_myaamia_quantized")
EXPORTED_MODEL_DIR = "mt5_large_myaamia_quantized" 

# --- 2. Load Model & Tokenizer ---
logging.info(f"Loading ExecuTorch model from: {EXPORTED_MODEL_DIR}")

try:
    # Load the Tokenizer (Optimum usually copies tokenizer files to the export dir)
    tokenizer = AutoTokenizer.from_pretrained(EXPORTED_MODEL_DIR)
    
    # Load the ExecuTorch Model
    model = ExecuTorchModelForSeq2SeqLM.from_pretrained(EXPORTED_MODEL_DIR)
    
except Exception as e:
    raise RuntimeError(f"Failed to load model/tokenizer from {EXPORTED_MODEL_DIR}. Error: {e}")

# --- 3. Data Loading ---
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

# --- 4. Inference Function (Python API) ---
def run_t5_inference(text_input):
    """
    Runs inference using the loaded ExecuTorch Python model.
    """
    try:
        # 1. Prepare Prompt
        prompt = f"translate English to Myaamia: {text_input}"
        
        # 2. Tokenize
        # return_tensors="pt" creates PyTorch tensors
        inputs = tokenizer(prompt, return_tensors="pt")
        
        # 3. Generate
        # ExecuTorchModelForSeq2SeqLM supports the standard .generate() API
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            num_beams=1, # Greedy search is standard for simple evaluations, change if needed
            do_sample=False
        )
        
        # 4. Decode
        # skip_special_tokens=True removes <pad>, </s>, etc. automatically
        output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        return output_text.strip()

    except Exception as e:
        logging.error(f"Error running model inference on input '{text_input}': {e}")
        return ""

# --- 5. Evaluation Loop ---
logging.info("Starting evaluation...")

all_predictions = []
all_references = []

# Debug print for the first item to ensure it's working
print("\n--- TEST RUN (First Item) ---")
first_ex = test_data[0]
print(f"Input: {first_ex['en']}")
pred = run_t5_inference(first_ex['en'])
print(f"Prediction: '{pred}'")
print("-" * 30 + "\n")

# Main Loop
for example in tqdm(test_data, desc="Evaluating"):
    prediction = run_t5_inference(example['en'])
    all_predictions.append(prediction)
    all_references.append([example['mia']])

# --- 6. Metrics & Saving ---
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

# Use directory name as model identifier
model_name = os.path.basename(os.path.normpath(EXPORTED_MODEL_DIR))
output_filename = f"eval_results_executorch_{model_name}.txt"

result_text = (
    "\n" + "="*30 + "\n"
    f"RESULTS FOR: {model_name} (.pte)\n"
    + "="*30 + "\n"
    f"Test Examples: {len(all_predictions)}\n"
    f"SacreBLEU:     {sacrebleu_results['score']:.4f}\n"
    f"ChrF++:        {chrf_results['score']:.4f}\n"
    + "="*30 + "\n"
)

print(result_text)
with open(output_filename, "w", encoding="utf-8") as f:
    f.write(result_text)