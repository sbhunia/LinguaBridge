import os
import pandas as pd
import logging
import evaluate
from datasets import Dataset, concatenate_datasets
from tqdm import tqdm
from transformers import AutoTokenizer
from optimum.executorch import ExecuTorchModelForCausalLM
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)

# --- 1. Configuration ---
# ⬇️ *** VERIFY YOUR PATHS HERE *** ⬇️
# Point this to the DIRECTORY containing your 'model.pte' and tokenizer files
EXECUTORCH_MODEL_DIR = "llama3_1b_executorch_output" 

# --- 2. Load Model & Tokenizer ---
logging.info(f"Loading ExecuTorch Llama model from: {EXECUTORCH_MODEL_DIR}")

try:
    # Load Tokenizer (Required for chat templates)
    tokenizer = AutoTokenizer.from_pretrained(EXECUTORCH_MODEL_DIR)
    # Ensure padding side is left for decoder-only generation
    tokenizer.padding_side = "left" 
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load ExecuTorch Model
    # Note: We use CausalLM, not Seq2SeqLM
    model = ExecuTorchModelForCausalLM.from_pretrained(EXECUTORCH_MODEL_DIR)
    
except Exception as e:
    raise RuntimeError(f"Failed to load model/tokenizer. Error: {e}")

# --- 3. Data Loading (Identical to previous scripts) ---
def load_tsv(path):
    try:
        df = pd.read_csv(path, sep="\t")
        df = df.dropna(subset=["mia", "en"])
        df["mia"] = df["mia"].astype(str).str.strip()
        df["en"] = df["en"].astype(str).str.strip()
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
valid_datasets = [ds for ds in datasets if ds is not None]

if not valid_datasets:
    raise ValueError("No valid datasets loaded.")

full_dataset = concatenate_datasets(valid_datasets)
full_dataset = full_dataset.shuffle(seed=42) # Ensure identical split
dataset_splits = full_dataset.train_test_split(test_size=0.1)
test_data = dataset_splits["test"]

logging.info(f"Loaded {len(test_data)} test examples.")

# --- 4. Inference Function (Llama 3 Chat Logic) ---
def run_llama_inference(text_input):
    """
    Runs inference using ExecuTorch + Llama 3 Chat Template.
    """
    system_prompt = "You are an expert translator. Translate the given English text into Myaamia."
    
    # 1. Apply Chat Template
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"translate english to myaamia: {text_input}"}
    ]
    
    # This converts the messages into Llama 3 input IDs (including <|begin_of_text|>, etc.)
    input_ids = tokenizer.apply_chat_template(
        messages, 
        add_generation_prompt=True, 
        return_tensors="pt"
    )

    try:
        # 2. Generate
        generated_ids = model.generate(
            input_ids,
            max_new_tokens=128,
            do_sample=False,   # Greedy decoding (deterministic)
            temperature=None,  # Not needed for greedy
            top_p=None,
            pad_token_id=tokenizer.eos_token_id
        )

        # 3. Decode & Clean
        # Llama generation includes the input prompt in the result. We need to strip it.
        input_length = input_ids.shape[1]
        new_tokens = generated_ids[0][input_length:]
        
        output_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        return output_text.strip()

    except Exception as e:
        logging.error(f"Error evaluating input '{text_input}': {e}")
        return ""

# --- 5. Evaluation Loop ---
logging.info("Starting evaluation...")

all_predictions = []
all_references = []

# Debug print for first item
print("\n--- TEST RUN (First Item) ---")
first_ex = test_data[0]
print(f"Input: {first_ex['en']}")
pred = run_llama_inference(first_ex['en'])
print(f"Prediction: '{pred}'")
print("-" * 30 + "\n")

for example in tqdm(test_data, desc="Evaluating Llama .pte"):
    prediction = run_llama_inference(example['en'])
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

model_name = os.path.basename(os.path.normpath(EXECUTORCH_MODEL_DIR))
output_filename = f"eval_results_executorch_llama_{model_name}.txt"

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